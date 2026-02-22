#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "gigavector/gv_crypto.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

#define TMP_PLAIN_PATH "/tmp/gv_test_crypto_plain.bin"
#define TMP_ENC_PATH   "/tmp/gv_test_crypto_enc.bin"
#define TMP_DEC_PATH   "/tmp/gv_test_crypto_dec.bin"

static void cleanup_temp_files(void) {
    unlink(TMP_PLAIN_PATH);
    unlink(TMP_ENC_PATH);
    unlink(TMP_DEC_PATH);
}

/* ── Test: config init ─────────────────────────────────────────────────── */
static int test_config_init(void) {
    GV_CryptoConfig cfg;
    memset(&cfg, 0xFF, sizeof(cfg));
    gv_crypto_config_init(&cfg);
    ASSERT(cfg.algorithm == GV_CRYPTO_AES_256_CBC, "default algorithm should be AES-256-CBC");
    ASSERT(cfg.kdf == GV_KDF_PBKDF2, "default KDF should be PBKDF2");
    ASSERT(cfg.kdf_iterations == 100000, "default iterations should be 100000");
    return 0;
}

/* ── Test: create and destroy ──────────────────────────────────────────── */
static int test_create_destroy(void) {
    GV_CryptoContext *ctx = gv_crypto_create(NULL);
    ASSERT(ctx != NULL, "crypto context creation with defaults");
    gv_crypto_destroy(ctx);
    /* NULL safety */
    gv_crypto_destroy(NULL);
    return 0;
}

/* ── Test: generate key ────────────────────────────────────────────────── */
static int test_generate_key(void) {
    GV_CryptoKey key;
    memset(&key, 0, sizeof(key));

    int rc = gv_crypto_generate_key(&key);
    ASSERT(rc == 0, "generate random key");

    /* Key should not be all zeros */
    int all_zero = 1;
    for (int i = 0; i < 32; i++) {
        if (key.key[i] != 0) { all_zero = 0; break; }
    }
    ASSERT(!all_zero, "generated key should not be all zeros");

    gv_crypto_wipe_key(&key);

    /* After wipe, key bytes should be zeroed */
    all_zero = 1;
    for (int i = 0; i < 32; i++) {
        if (key.key[i] != 0) { all_zero = 0; break; }
    }
    ASSERT(all_zero, "wiped key should be all zeros");

    return 0;
}

/* ── Test: derive key from password ────────────────────────────────────── */
static int test_derive_key(void) {
    GV_CryptoContext *ctx = gv_crypto_create(NULL);
    ASSERT(ctx != NULL, "context creation");

    unsigned char salt[16];
    int rc = gv_crypto_generate_salt(salt, sizeof(salt));
    ASSERT(rc == 0, "generate salt");

    GV_CryptoKey key;
    rc = gv_crypto_derive_key(ctx, "mypassword", 10, salt, sizeof(salt), &key);
    ASSERT(rc == 0, "derive key from password");

    /* Derive again with same inputs -- should get same key */
    GV_CryptoKey key2;
    rc = gv_crypto_derive_key(ctx, "mypassword", 10, salt, sizeof(salt), &key2);
    ASSERT(rc == 0, "derive key again");
    ASSERT(memcmp(key.key, key2.key, 32) == 0, "same password+salt should produce same key");

    gv_crypto_wipe_key(&key);
    gv_crypto_wipe_key(&key2);
    gv_crypto_destroy(ctx);
    return 0;
}

/* ── Test: encrypt and decrypt ─────────────────────────────────────────── */
static int test_encrypt_decrypt(void) {
    GV_CryptoContext *ctx = gv_crypto_create(NULL);
    ASSERT(ctx != NULL, "context creation");

    GV_CryptoKey key;
    gv_crypto_generate_key(&key);
    gv_crypto_generate_iv(key.iv);

    const unsigned char plaintext[] = "Hello, GigaVector encryption!";
    size_t pt_len = sizeof(plaintext) - 1;

    unsigned char ciphertext[256];
    size_t ct_len = 0;
    int rc = gv_crypto_encrypt(ctx, &key, plaintext, pt_len, ciphertext, &ct_len);
    ASSERT(rc == 0, "encrypt data");
    ASSERT(ct_len > 0, "ciphertext length should be > 0");
    ASSERT(memcmp(ciphertext, plaintext, pt_len) != 0, "ciphertext should differ from plaintext");

    unsigned char decrypted[256];
    size_t dec_len = 0;
    rc = gv_crypto_decrypt(ctx, &key, ciphertext, ct_len, decrypted, &dec_len);
    ASSERT(rc == 0, "decrypt data");
    ASSERT(dec_len == pt_len, "decrypted length should match original");
    ASSERT(memcmp(decrypted, plaintext, pt_len) == 0, "decrypted data should match original");

    gv_crypto_wipe_key(&key);
    gv_crypto_destroy(ctx);
    return 0;
}

/* ── Test: constant time compare ───────────────────────────────────────── */
static int test_constant_time_compare(void) {
    unsigned char a[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    unsigned char b[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    unsigned char c[16] = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

    int eq = gv_crypto_constant_time_compare(a, b, 16);
    ASSERT(eq == 0, "identical arrays should compare equal (return 0)");

    int neq = gv_crypto_constant_time_compare(a, c, 16);
    ASSERT(neq != 0, "different arrays should compare non-equal (return non-zero)");

    return 0;
}

/* ── Test: HMAC-SHA256 ─────────────────────────────────────────────────── */
static int test_hmac_sha256(void) {
    const unsigned char key[] = "secret-key";
    const unsigned char data[] = "message to authenticate";
    unsigned char hmac1[32], hmac2[32];

    int rc = gv_crypto_hmac_sha256(key, sizeof(key) - 1, data, sizeof(data) - 1, hmac1);
    ASSERT(rc == 0, "compute HMAC-SHA256");

    /* Same inputs should produce same HMAC */
    rc = gv_crypto_hmac_sha256(key, sizeof(key) - 1, data, sizeof(data) - 1, hmac2);
    ASSERT(rc == 0, "compute HMAC-SHA256 again");
    ASSERT(memcmp(hmac1, hmac2, 32) == 0, "same key+data should produce same HMAC");

    /* Different data should produce different HMAC */
    const unsigned char data2[] = "different message";
    unsigned char hmac3[32];
    rc = gv_crypto_hmac_sha256(key, sizeof(key) - 1, data2, sizeof(data2) - 1, hmac3);
    ASSERT(rc == 0, "compute HMAC-SHA256 with different data");
    ASSERT(memcmp(hmac1, hmac3, 32) != 0, "different data should produce different HMAC");

    return 0;
}

/* ── Test: algorithm string ────────────────────────────────────────────── */
static int test_algorithm_string(void) {
    const char *s;

    s = gv_crypto_algorithm_string(GV_CRYPTO_NONE);
    ASSERT(s != NULL, "algorithm string for NONE should not be NULL");

    s = gv_crypto_algorithm_string(GV_CRYPTO_AES_256_CBC);
    ASSERT(s != NULL, "algorithm string for AES-256-CBC should not be NULL");

    s = gv_crypto_algorithm_string(GV_CRYPTO_AES_256_GCM);
    ASSERT(s != NULL, "algorithm string for AES-256-GCM should not be NULL");

    return 0;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing config init...",             test_config_init},
        {"Testing create/destroy...",          test_create_destroy},
        {"Testing generate key...",            test_generate_key},
        {"Testing derive key...",              test_derive_key},
        {"Testing encrypt/decrypt...",         test_encrypt_decrypt},
        {"Testing constant time compare...",   test_constant_time_compare},
        {"Testing HMAC-SHA256...",             test_hmac_sha256},
        {"Testing algorithm string...",        test_algorithm_string},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    cleanup_temp_files();
    return passed == n ? 0 : 1;
}
