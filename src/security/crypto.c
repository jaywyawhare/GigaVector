/**
 * @file crypto.c
 * @brief Encryption implementation.
 *
 * This is a portable implementation. For production, consider using
 * OpenSSL or libsodium for better performance and security auditing.
 */

#include "security/crypto.h"
#include "core/memory.h"
#include "security/auth.h"  /* For SHA-256 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#ifdef __linux__
#include <sys/random.h>
#elif defined(_WIN32)
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#endif

#ifdef GV_HAVE_OPENSSL
#include <openssl/evp.h>
#endif

/* AES-256-GCM authenticated tag length (bytes). */
#define GCM_TAG_LEN 16
/* Per-message GCM nonce (96-bit) and CBC IV lengths. A FRESH random value is
 * generated for every crypto_encrypt() call and prepended to the output, so the
 * (key, nonce) pair is never reused — the old code took the nonce from the fixed
 * key->iv, which is catastrophic for GCM across multiple messages. */
#define GCM_NONCE_LEN 12
#define CBC_IV_LEN    16
/* Max bytes crypto_encrypt adds over the plaintext: IV/nonce prefix + tag/pad.
 * (GCM: 12 nonce + 16 tag = 28; CBC: 16 IV + up-to-16 pad = 32.) */
#define CRYPTO_MAX_OVERHEAD 48

struct GV_CryptoContext {
    GV_CryptoConfig config;
};

struct GV_CryptoStream {
    GV_CryptoContext *ctx;
    GV_CryptoKey key;
    int encrypting;
    unsigned char buffer[16];   /* partial input block accumulator */
    size_t buffer_len;
    unsigned char prev[16];     /* CBC chaining block (holds the IV initially) */
    int iv_done;                /* encrypt: IV emitted; decrypt: IV consumed */
    unsigned char held[16];     /* decrypt: previous plaintext block, held back so
                                 * final() can strip the PKCS7 padding of the LAST block */
    int have_held;
};

/* AES S-box */
static const unsigned char sbox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

/* Inverse S-box */
static const unsigned char rsbox[256] = {
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
};

/* Rcon */
static const unsigned char Rcon[11] = {
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};

/* GF(2^8) multiplication */
static unsigned char gmul(unsigned char a, unsigned char b) {
    unsigned char p = 0;
    for (int i = 0; i < 8; i++) {
        if (b & 1) p ^= a;
        int hi_bit_set = a & 0x80;
        a <<= 1;
        if (hi_bit_set) a ^= 0x1b;
        b >>= 1;
    }
    return p;
}

/* Key expansion for AES-256 */
static void aes256_key_expansion(const unsigned char *key, unsigned char *roundkeys) {
    unsigned char temp[4];
    int i = 0;

    /* First 32 bytes are the key */
    memcpy(roundkeys, key, 32);
    i = 8;

    while (i < 60) {
        temp[0] = roundkeys[(i - 1) * 4 + 0];
        temp[1] = roundkeys[(i - 1) * 4 + 1];
        temp[2] = roundkeys[(i - 1) * 4 + 2];
        temp[3] = roundkeys[(i - 1) * 4 + 3];

        if (i % 8 == 0) {
            /* RotWord + SubWord + Rcon */
            unsigned char t = temp[0];
            temp[0] = sbox[temp[1]] ^ Rcon[i / 8];
            temp[1] = sbox[temp[2]];
            temp[2] = sbox[temp[3]];
            temp[3] = sbox[t];
        } else if (i % 8 == 4) {
            /* SubWord only */
            temp[0] = sbox[temp[0]];
            temp[1] = sbox[temp[1]];
            temp[2] = sbox[temp[2]];
            temp[3] = sbox[temp[3]];
        }

        roundkeys[i * 4 + 0] = roundkeys[(i - 8) * 4 + 0] ^ temp[0];
        roundkeys[i * 4 + 1] = roundkeys[(i - 8) * 4 + 1] ^ temp[1];
        roundkeys[i * 4 + 2] = roundkeys[(i - 8) * 4 + 2] ^ temp[2];
        roundkeys[i * 4 + 3] = roundkeys[(i - 8) * 4 + 3] ^ temp[3];
        i++;
    }
}

/* AES add round key */
static void add_round_key(unsigned char state[16], const unsigned char *roundkey) {
    for (int i = 0; i < 16; i++) {
        state[i] ^= roundkey[i];
    }
}

/* AES sub bytes */
static void sub_bytes(unsigned char state[16]) {
    for (int i = 0; i < 16; i++) {
        state[i] = sbox[state[i]];
    }
}

static void inv_sub_bytes(unsigned char state[16]) {
    for (int i = 0; i < 16; i++) {
        state[i] = rsbox[state[i]];
    }
}

/* AES shift rows */
static void shift_rows(unsigned char state[16]) {
    unsigned char temp;
    /* Row 1: shift left 1 */
    temp = state[1]; state[1] = state[5]; state[5] = state[9]; state[9] = state[13]; state[13] = temp;
    /* Row 2: shift left 2 */
    temp = state[2]; state[2] = state[10]; state[10] = temp;
    temp = state[6]; state[6] = state[14]; state[14] = temp;
    /* Row 3: shift left 3 (= right 1) */
    temp = state[15]; state[15] = state[11]; state[11] = state[7]; state[7] = state[3]; state[3] = temp;
}

static void inv_shift_rows(unsigned char state[16]) {
    unsigned char temp;
    /* Row 1: shift right 1 */
    temp = state[13]; state[13] = state[9]; state[9] = state[5]; state[5] = state[1]; state[1] = temp;
    /* Row 2: shift right 2 */
    temp = state[2]; state[2] = state[10]; state[10] = temp;
    temp = state[6]; state[6] = state[14]; state[14] = temp;
    /* Row 3: shift right 3 (= left 1) */
    temp = state[3]; state[3] = state[7]; state[7] = state[11]; state[11] = state[15]; state[15] = temp;
}

/* AES mix columns */
static void mix_columns(unsigned char state[16]) {
    for (int i = 0; i < 4; i++) {
        unsigned char a = state[i * 4 + 0];
        unsigned char b = state[i * 4 + 1];
        unsigned char c = state[i * 4 + 2];
        unsigned char d = state[i * 4 + 3];
        state[i * 4 + 0] = gmul(a, 2) ^ gmul(b, 3) ^ c ^ d;
        state[i * 4 + 1] = a ^ gmul(b, 2) ^ gmul(c, 3) ^ d;
        state[i * 4 + 2] = a ^ b ^ gmul(c, 2) ^ gmul(d, 3);
        state[i * 4 + 3] = gmul(a, 3) ^ b ^ c ^ gmul(d, 2);
    }
}

static void inv_mix_columns(unsigned char state[16]) {
    for (int i = 0; i < 4; i++) {
        unsigned char a = state[i * 4 + 0];
        unsigned char b = state[i * 4 + 1];
        unsigned char c = state[i * 4 + 2];
        unsigned char d = state[i * 4 + 3];
        state[i * 4 + 0] = gmul(a, 0x0e) ^ gmul(b, 0x0b) ^ gmul(c, 0x0d) ^ gmul(d, 0x09);
        state[i * 4 + 1] = gmul(a, 0x09) ^ gmul(b, 0x0e) ^ gmul(c, 0x0b) ^ gmul(d, 0x0d);
        state[i * 4 + 2] = gmul(a, 0x0d) ^ gmul(b, 0x09) ^ gmul(c, 0x0e) ^ gmul(d, 0x0b);
        state[i * 4 + 3] = gmul(a, 0x0b) ^ gmul(b, 0x0d) ^ gmul(c, 0x09) ^ gmul(d, 0x0e);
    }
}

/* AES-256 encrypt one block */
static void aes256_encrypt_block(const unsigned char in[16], unsigned char out[16],
                                  const unsigned char *roundkeys) {
    unsigned char state[16];
    memcpy(state, in, 16);

    add_round_key(state, roundkeys);

    for (int round = 1; round < 14; round++) {
        sub_bytes(state);
        shift_rows(state);
        mix_columns(state);
        add_round_key(state, roundkeys + round * 16);
    }

    sub_bytes(state);
    shift_rows(state);
    add_round_key(state, roundkeys + 14 * 16);

    memcpy(out, state, 16);
}

/* AES-256 decrypt one block */
static void aes256_decrypt_block(const unsigned char in[16], unsigned char out[16],
                                  const unsigned char *roundkeys) {
    unsigned char state[16];
    memcpy(state, in, 16);

    add_round_key(state, roundkeys + 14 * 16);

    for (int round = 13; round >= 1; round--) {
        inv_shift_rows(state);
        inv_sub_bytes(state);
        add_round_key(state, roundkeys + round * 16);
        inv_mix_columns(state);
    }

    inv_shift_rows(state);
    inv_sub_bytes(state);
    add_round_key(state, roundkeys);

    memcpy(out, state, 16);
}

int gv_secure_random_bytes(unsigned char *buf, size_t len) {
#if defined(_WIN32)
    NTSTATUS st = BCryptGenRandom(NULL, (PUCHAR)buf, (ULONG)len, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    if (BCRYPT_SUCCESS(st)) return 0;
#elif defined(__linux__)
    ssize_t r = getrandom(buf, len, 0);
    if (r >= 0 && (size_t)r == len) return 0;
#endif
#if !defined(_WIN32)
    FILE *fp = fopen("/dev/urandom", "rb");
    if (fp) {
        size_t n = fread(buf, 1, len, fp);
        fclose(fp);
        if (n == len) return 0;
    }
#endif
    fprintf(stderr, "GigaVector: FATAL: could not obtain cryptographic randomness\n");
    return -1;
}

static const GV_CryptoConfig DEFAULT_CONFIG = {
    .algorithm = GV_CRYPTO_AES_256_CBC,
    .kdf = GV_KDF_PBKDF2,
    .kdf_iterations = 100000
};

void crypto_config_init(GV_CryptoConfig *config) {
    if (!config) return;
    *config = DEFAULT_CONFIG;
}

GV_CryptoContext *crypto_create(const GV_CryptoConfig *config) {
    GV_CryptoContext *ctx = gv_calloc(1, sizeof(GV_CryptoContext));
    if (!ctx) return NULL;
    ctx->config = config ? *config : DEFAULT_CONFIG;
    return ctx;
}

void crypto_destroy(GV_CryptoContext *ctx) {
    if (!ctx) return;
    memset(ctx, 0, sizeof(*ctx));
    gv_free(ctx);
}

int crypto_derive_key(GV_CryptoContext *ctx, const char *password,
                          size_t password_len, const unsigned char *salt,
                          size_t salt_len, GV_CryptoKey *key) {
    if (!ctx || !password || !salt || !key) return -1;

    memset(key, 0, sizeof(*key));

    /* PBKDF2-HMAC-SHA256 */
    unsigned char U[32], T[32];
    uint32_t iterations = ctx->config.kdf_iterations;
    if (iterations == 0) iterations = 100000;

    /* Derive 32 bytes for key + generate random IV */
    unsigned char dk[32];

    memset(T, 0, 32);

    /* Counter (big-endian) */
    unsigned char counter[4] = {0, 0, 0, 1};

    /* First iteration: HMAC(password, salt || counter) */
    unsigned char *msg = gv_alloc(salt_len + 4);
    if (!msg) return -1;
    memcpy(msg, salt, salt_len);
    memcpy(msg + salt_len, counter, 4);

    crypto_hmac_sha256((unsigned char *)password, password_len, msg, salt_len + 4, U);
    memcpy(T, U, 32);

    /* Remaining iterations */
    for (uint32_t i = 1; i < iterations; i++) {
        crypto_hmac_sha256((unsigned char *)password, password_len, U, 32, U);
        for (int j = 0; j < 32; j++) {
            T[j] ^= U[j];
        }
    }

    gv_free(msg);
    memcpy(dk, T, 32);

    memcpy(key->key, dk, 32);
    if (gv_secure_random_bytes(key->iv, 16) != 0) return -1;

    return 0;
}

int crypto_generate_key(GV_CryptoKey *key) {
    if (!key) return -1;
    if (gv_secure_random_bytes(key->key, 32) != 0) return -1;
    if (gv_secure_random_bytes(key->iv, 16) != 0) {
        crypto_wipe_key(key);
        return -1;
    }
    return 0;
}

int crypto_generate_iv(unsigned char *iv) {
    if (!iv) return -1;
    return gv_secure_random_bytes(iv, 16);
}

int crypto_generate_salt(unsigned char *salt, size_t salt_len) {
    if (!salt || salt_len == 0) return -1;
    return gv_secure_random_bytes(salt, salt_len);
}

void crypto_wipe_key(GV_CryptoKey *key) {
    if (!key) return;
    /*
     * Secure wipe.  Rely SOLELY on the volatile write: a plain memset() to a
     * buffer that is not read again afterwards is dead-store-eliminable and
     * may be optimized away by the compiler.  Writing through a volatile
     * pointer forces the store to actually happen.
     */
    volatile unsigned char *p = (volatile unsigned char *)key;
    for (size_t i = 0; i < sizeof(*key); i++) {
        p[i] = 0;
    }
}

#ifdef GV_HAVE_OPENSSL
/*
 * AES-256-GCM authenticated encryption using OpenSSL.
 *
 * Output layout (self-contained, since the public API has no separate nonce/tag
 * parameters): [12-byte nonce][ciphertext bytes ...][16-byte GCM tag].
 * A fresh random 96-bit nonce is generated per call (never key->iv), so the
 * (key, nonce) pair is never reused. Returns 0 on success, -1 on failure.
 */
static int gcm_encrypt_openssl(const GV_CryptoKey *key,
                               const unsigned char *plaintext, size_t plaintext_len,
                               unsigned char *ciphertext, size_t *ciphertext_len) {
    unsigned char nonce[GCM_NONCE_LEN];
    if (gv_secure_random_bytes(nonce, GCM_NONCE_LEN) != 0) return -1;

    EVP_CIPHER_CTX *c = EVP_CIPHER_CTX_new();
    if (!c) return -1;
    int rc = -1;
    int outl = 0;
    size_t total = 0;

    if (EVP_EncryptInit_ex(c, EVP_aes_256_gcm(), NULL, NULL, NULL) != 1) goto done;
    if (EVP_CIPHER_CTX_ctrl(c, EVP_CTRL_GCM_SET_IVLEN, GCM_NONCE_LEN, NULL) != 1) goto done;
    if (EVP_EncryptInit_ex(c, NULL, NULL, key->key, nonce) != 1) goto done;

    /* Prepend the nonce so the decryptor can recover it. */
    memcpy(ciphertext, nonce, GCM_NONCE_LEN);
    total = GCM_NONCE_LEN;

    if (plaintext_len > 0) {
        if (EVP_EncryptUpdate(c, ciphertext + total, &outl, plaintext,
                              (int)plaintext_len) != 1) goto done;
        total += (size_t)outl;
    }
    if (EVP_EncryptFinal_ex(c, ciphertext + total, &outl) != 1) goto done;
    total += (size_t)outl;

    if (EVP_CIPHER_CTX_ctrl(c, EVP_CTRL_GCM_GET_TAG, GCM_TAG_LEN,
                            ciphertext + total) != 1) goto done;
    total += GCM_TAG_LEN;

    *ciphertext_len = total;
    rc = 0;
done:
    EVP_CIPHER_CTX_free(c);
    return rc;
}

static int gcm_decrypt_openssl(const GV_CryptoKey *key,
                               const unsigned char *ciphertext, size_t ciphertext_len,
                               unsigned char *plaintext, size_t *plaintext_len) {
    /* Layout: [12 nonce][ct][16 tag]. */
    if (ciphertext_len < GCM_NONCE_LEN + GCM_TAG_LEN) return -1;
    const unsigned char *nonce = ciphertext;
    const unsigned char *ct    = ciphertext + GCM_NONCE_LEN;
    size_t ct_len = ciphertext_len - GCM_NONCE_LEN - GCM_TAG_LEN;
    const unsigned char *tag = ct + ct_len;

    EVP_CIPHER_CTX *c = EVP_CIPHER_CTX_new();
    if (!c) return -1;
    int rc = -1;
    int outl = 0;
    size_t total = 0;

    if (EVP_DecryptInit_ex(c, EVP_aes_256_gcm(), NULL, NULL, NULL) != 1) goto done;
    if (EVP_CIPHER_CTX_ctrl(c, EVP_CTRL_GCM_SET_IVLEN, GCM_NONCE_LEN, NULL) != 1) goto done;
    if (EVP_DecryptInit_ex(c, NULL, NULL, key->key, nonce) != 1) goto done;

    if (ct_len > 0) {
        if (EVP_DecryptUpdate(c, plaintext, &outl, ct,
                              (int)ct_len) != 1) goto done;
        total += (size_t)outl;
    }
    /* Set expected tag, then finalize: fails if authentication is invalid. */
    if (EVP_CIPHER_CTX_ctrl(c, EVP_CTRL_GCM_SET_TAG, GCM_TAG_LEN,
                            (void *)tag) != 1) goto done;
    if (EVP_DecryptFinal_ex(c, plaintext + total, &outl) != 1) goto done;
    total += (size_t)outl;

    *plaintext_len = total;
    rc = 0;
done:
    EVP_CIPHER_CTX_free(c);
    return rc;
}
#endif /* GV_HAVE_OPENSSL */

int crypto_encrypt(GV_CryptoContext *ctx, const GV_CryptoKey *key,
                       const unsigned char *plaintext, size_t plaintext_len,
                       unsigned char *ciphertext, size_t *ciphertext_len) {
    if (!ctx || !key || !plaintext || !ciphertext || !ciphertext_len) return -1;

    /*
     * Dispatch on the configured algorithm.  GCM is authenticated; CBC in this
     * portable build is unauthenticated (no MAC).  We must never silently do
     * unauthenticated CBC when the caller asked for authenticated GCM.
     */
    if (ctx->config.algorithm == GV_CRYPTO_AES_256_GCM) {
#ifdef GV_HAVE_OPENSSL
        return gcm_encrypt_openssl(key, plaintext, plaintext_len,
                                   ciphertext, ciphertext_len);
#else
        /* Authenticated GCM is unavailable without OpenSSL: fail closed
         * rather than fall back to unauthenticated CBC. */
        (void)plaintext_len;
        return -1;
#endif
    }
    if (ctx->config.algorithm != GV_CRYPTO_AES_256_CBC) {
        return -1;  /* unknown/unsupported algorithm */
    }

    /* Fresh random IV per message, prepended to the output; the old code used the
     * fixed key->iv, which leaks first-block equality across messages. */
    unsigned char iv[CBC_IV_LEN];
    if (gv_secure_random_bytes(iv, CBC_IV_LEN) != 0) return -1;

    unsigned char roundkeys[240];
    aes256_key_expansion(key->key, roundkeys);

    /* Calculate padded length (PKCS7) */
    size_t pad_len = 16 - (plaintext_len % 16);
    size_t total_len = plaintext_len + pad_len;
    *ciphertext_len = CBC_IV_LEN + total_len;   /* IV prefix + ciphertext */

    /* Prepend the IV; ciphertext blocks are written after it. */
    memcpy(ciphertext, iv, CBC_IV_LEN);
    unsigned char prev_block[16];
    memcpy(prev_block, iv, CBC_IV_LEN);

    size_t pos = 0;
    while (pos < total_len) {
        unsigned char block[16];
        size_t block_len = (pos < plaintext_len) ? (plaintext_len - pos) : 0;
        if (block_len > 16) block_len = 16;

        if (block_len > 0) memcpy(block, plaintext + pos, block_len);

        /* PKCS7 padding fills the remainder of the block (or an entire block) */
        for (size_t i = block_len; i < 16; i++) {
            block[i] = (unsigned char)pad_len;
        }

        /* XOR with previous ciphertext block */
        for (int i = 0; i < 16; i++) {
            block[i] ^= prev_block[i];
        }

        aes256_encrypt_block(block, ciphertext + CBC_IV_LEN + pos, roundkeys);
        memcpy(prev_block, ciphertext + CBC_IV_LEN + pos, 16);

        pos += 16;
    }

    return 0;
}

int crypto_decrypt(GV_CryptoContext *ctx, const GV_CryptoKey *key,
                       const unsigned char *ciphertext, size_t ciphertext_len,
                       unsigned char *plaintext, size_t *plaintext_len) {
    if (!ctx || !key || !ciphertext || !plaintext || !plaintext_len) return -1;

    if (ctx->config.algorithm == GV_CRYPTO_AES_256_GCM) {
#ifdef GV_HAVE_OPENSSL
        /* Authenticated: fails if the tag does not verify. */
        return gcm_decrypt_openssl(key, ciphertext, ciphertext_len,
                                   plaintext, plaintext_len);
#else
        return -1;  /* GCM unsupported without OpenSSL: fail closed */
#endif
    }
    if (ctx->config.algorithm != GV_CRYPTO_AES_256_CBC) {
        return -1;  /* unknown/unsupported algorithm */
    }

    /* Layout: [16 IV][ciphertext]. */
    if (ciphertext_len < CBC_IV_LEN) return -1;
    const unsigned char *iv = ciphertext;
    const unsigned char *ct = ciphertext + CBC_IV_LEN;
    size_t ct_len = ciphertext_len - CBC_IV_LEN;
    if (ct_len == 0 || ct_len % 16 != 0) return -1;

    unsigned char roundkeys[240];
    aes256_key_expansion(key->key, roundkeys);

    /* CBC mode decryption, seeded with the prepended IV. */
    unsigned char prev_block[16];
    memcpy(prev_block, iv, CBC_IV_LEN);

    for (size_t pos = 0; pos < ct_len; pos += 16) {
        unsigned char block[16];
        aes256_decrypt_block(ct + pos, block, roundkeys);

        /* XOR with previous ciphertext block */
        for (int i = 0; i < 16; i++) {
            block[i] ^= prev_block[i];
        }

        memcpy(plaintext + pos, block, 16);
        memcpy(prev_block, ct + pos, 16);
    }

    /* Remove and validate PKCS7 padding */
    unsigned char pad = plaintext[ct_len - 1];
    if (pad == 0 || pad > 16) {
        return -1;
    }
    for (unsigned char pi = 1; pi < pad; pi++) {
        if (plaintext[ct_len - 1 - pi] != pad) {
            return -1;
        }
    }
    *plaintext_len = ct_len - pad;

    return 0;
}

#define FILE_BUFFER_SIZE (64 * 1024)

int crypto_encrypt_file(GV_CryptoContext *ctx, const GV_CryptoKey *key,
                            const char *input_path, const char *output_path) {
    if (!ctx || !key || !input_path || !output_path) return -1;

    FILE *fin = fopen(input_path, "rb");
    if (!fin) return -1;

    FILE *fout = fopen(output_path, "wb");
    if (!fout) {
        fclose(fin);
        return -1;
    }

    unsigned char *buffer = gv_alloc(FILE_BUFFER_SIZE);
    unsigned char *cipher = gv_alloc(FILE_BUFFER_SIZE + CRYPTO_MAX_OVERHEAD);
    if (!buffer || !cipher) {
        gv_free(buffer);
        gv_free(cipher);
        fclose(fin);
        fclose(fout);
        return -1;
    }

    /* Each 64 KiB plaintext chunk is encrypted independently (crypto_encrypt now
     * embeds its own nonce/IV + tag/pad, so a chunk's ciphertext is larger than
     * its plaintext by a variable amount). Length-FRAME every chunk so the
     * decryptor reads exactly cipher_len bytes — the old code re-read fixed 64 KiB
     * chunks, which desynced after the first chunk for any file > 64 KiB. */
    int rc = 0;
    size_t nread;
    while ((nread = fread(buffer, 1, FILE_BUFFER_SIZE, fin)) > 0) {
        size_t cipher_len = 0;
        if (crypto_encrypt(ctx, key, buffer, nread, cipher, &cipher_len) != 0) { rc = -1; break; }
        unsigned char lenbuf[4];
        lenbuf[0] = (unsigned char)(cipher_len & 0xFF);
        lenbuf[1] = (unsigned char)((cipher_len >> 8) & 0xFF);
        lenbuf[2] = (unsigned char)((cipher_len >> 16) & 0xFF);
        lenbuf[3] = (unsigned char)((cipher_len >> 24) & 0xFF);
        if (fwrite(lenbuf, 1, 4, fout) != 4 ||
            fwrite(cipher, 1, cipher_len, fout) != cipher_len) { rc = -1; break; }
    }
    if (ferror(fin)) rc = -1;

    gv_free(buffer);
    gv_free(cipher);
    if (fclose(fout) != 0) rc = -1;
    fclose(fin);

    return rc;
}

int crypto_decrypt_file(GV_CryptoContext *ctx, const GV_CryptoKey *key,
                            const char *input_path, const char *output_path) {
    if (!ctx || !key || !input_path || !output_path) return -1;

    FILE *fin = fopen(input_path, "rb");
    if (!fin) return -1;

    FILE *fout = fopen(output_path, "wb");
    if (!fout) {
        fclose(fin);
        return -1;
    }

    size_t cap = FILE_BUFFER_SIZE + CRYPTO_MAX_OVERHEAD;
    unsigned char *cbuf  = gv_alloc(cap);
    unsigned char *plain = gv_alloc(cap);   /* plaintext is never larger than its ciphertext */
    if (!cbuf || !plain) {
        gv_free(cbuf);
        gv_free(plain);
        fclose(fin);
        fclose(fout);
        return -1;
    }

    /* Read each length-framed chunk: [4-byte LE cipher_len][cipher bytes]. */
    int rc = 0;
    size_t got;
    unsigned char lenbuf[4];
    while ((got = fread(lenbuf, 1, 4, fin)) == 4) {
        uint32_t clen = (uint32_t)lenbuf[0] | ((uint32_t)lenbuf[1] << 8) |
                        ((uint32_t)lenbuf[2] << 16) | ((uint32_t)lenbuf[3] << 24);
        if (clen == 0 || clen > cap) { rc = -1; break; }   /* corrupt/oversized frame */
        if (fread(cbuf, 1, clen, fin) != clen) { rc = -1; break; }
        size_t plain_len = 0;
        if (crypto_decrypt(ctx, key, cbuf, clen, plain, &plain_len) != 0) { rc = -1; break; }
        if (fwrite(plain, 1, plain_len, fout) != plain_len) { rc = -1; break; }
    }
    if (got != 0 && got != 4) rc = -1;   /* trailing partial length header = corruption */
    if (ferror(fin)) rc = -1;

    gv_free(cbuf);
    gv_free(plain);
    if (fclose(fout) != 0) rc = -1;
    fclose(fin);

    return rc;
}

GV_CryptoStream *crypto_stream_create(GV_CryptoContext *ctx,
                                          const GV_CryptoKey *key,
                                          int encrypting) {
    if (!ctx || !key) return NULL;

    GV_CryptoStream *stream = gv_calloc(1, sizeof(GV_CryptoStream));
    if (!stream) return NULL;

    stream->ctx = ctx;
    stream->key = *key;
    stream->encrypting = encrypting;
    stream->buffer_len = 0;
    stream->iv_done = 0;
    stream->have_held = 0;
    if (encrypting) {
        /* Fresh random IV per stream (emitted to the output before any ciphertext
         * on the first update/final). Never the fixed key->iv, which would reuse
         * the IV across streams. */
        if (gv_secure_random_bytes(stream->prev, 16) != 0) { gv_free(stream); return NULL; }
    }

    return stream;
}

int crypto_stream_update(GV_CryptoStream *stream,
                             const unsigned char *input, size_t input_len,
                             unsigned char *output, size_t *output_len) {
    if (!stream || (!input && input_len) || !output || !output_len) return -1;

    *output_len = 0;

    unsigned char roundkeys[240];
    aes256_key_expansion(stream->key.key, roundkeys);

    /* Encrypt: emit the random IV once, before any ciphertext. */
    if (stream->encrypting && !stream->iv_done) {
        memcpy(output, stream->prev, 16);
        *output_len = 16;
        stream->iv_done = 1;
    }

    size_t processed = 0;
    while (processed < input_len) {
        size_t space = 16 - stream->buffer_len;
        size_t to_copy = input_len - processed;
        if (to_copy > space) to_copy = space;

        memcpy(stream->buffer + stream->buffer_len, input + processed, to_copy);
        stream->buffer_len += to_copy;
        processed += to_copy;

        if (stream->buffer_len == 16) {
            if (stream->encrypting) {
                for (int i = 0; i < 16; i++) stream->buffer[i] ^= stream->prev[i];
                aes256_encrypt_block(stream->buffer, output + *output_len, roundkeys);
                memcpy(stream->prev, output + *output_len, 16);  /* chain */
                *output_len += 16;
            } else if (!stream->iv_done) {
                /* Decrypt: the FIRST block of the input stream is the IV. */
                memcpy(stream->prev, stream->buffer, 16);
                stream->iv_done = 1;
            } else {
                unsigned char cipher[16], plain[16];
                memcpy(cipher, stream->buffer, 16);
                aes256_decrypt_block(stream->buffer, plain, roundkeys);
                for (int i = 0; i < 16; i++) plain[i] ^= stream->prev[i];
                memcpy(stream->prev, cipher, 16);  /* chain */
                /* Hold back one plaintext block so final() can strip the PKCS7
                 * padding from the genuine last block; emit the prior held block. */
                if (stream->have_held) {
                    memcpy(output + *output_len, stream->held, 16);
                    *output_len += 16;
                }
                memcpy(stream->held, plain, 16);
                stream->have_held = 1;
            }
            stream->buffer_len = 0;
        }
    }

    return 0;
}

int crypto_stream_final(GV_CryptoStream *stream,
                            unsigned char *output, size_t *output_len) {
    if (!stream || !output || !output_len) return -1;

    *output_len = 0;

    unsigned char roundkeys[240];
    aes256_key_expansion(stream->key.key, roundkeys);

    if (stream->encrypting) {
        /* Emit the IV first if no update() ever did (empty plaintext). */
        if (!stream->iv_done) {
            memcpy(output, stream->prev, 16);
            *output_len = 16;
            stream->iv_done = 1;
        }
        /* PKCS7-pad the final (possibly empty) partial block, then CBC-encrypt. */
        size_t pad_len = 16 - stream->buffer_len;
        for (size_t i = stream->buffer_len; i < 16; i++) {
            stream->buffer[i] = (unsigned char)pad_len;
        }
        for (int i = 0; i < 16; i++) stream->buffer[i] ^= stream->prev[i];
        aes256_encrypt_block(stream->buffer, output + *output_len, roundkeys);
        *output_len += 16;
    } else {
        /* A leftover partial block means the ciphertext was truncated. */
        if (stream->buffer_len != 0) return -1;
        /* The held block is the last plaintext block; strip and validate its
         * PKCS7 padding. Without a held block the stream carried no data block
         * (IV only / empty) — invalid, since encrypt always emits ≥1 padded block. */
        if (!stream->have_held) return -1;
        unsigned char pad = stream->held[15];
        if (pad == 0 || pad > 16) return -1;
        for (unsigned char pi = 1; pi < pad; pi++) {
            if (stream->held[16 - 1 - pi] != pad) return -1;
        }
        size_t out = 16 - (size_t)pad;
        if (out > 0) memcpy(output, stream->held, out);
        *output_len = out;
    }

    return 0;
}

void crypto_stream_destroy(GV_CryptoStream *stream) {
    if (!stream) return;
    crypto_wipe_key(&stream->key);
    memset(stream, 0, sizeof(*stream));
    gv_free(stream);
}

int crypto_hmac_sha256(const unsigned char *key, size_t key_len,
                           const unsigned char *data, size_t data_len,
                           unsigned char *hmac) {
    if (!key || !data || !hmac) return -1;

    unsigned char k_ipad[64], k_opad[64];

    unsigned char key_block[64];
    memset(key_block, 0, 64);

    if (key_len > 64) {
        auth_sha256(key, key_len, key_block);
    } else {
        memcpy(key_block, key, key_len);
    }

    for (int i = 0; i < 64; i++) {
        k_ipad[i] = key_block[i] ^ 0x36;
        k_opad[i] = key_block[i] ^ 0x5c;
    }

    /* Inner hash: H(K XOR ipad || data) */
    unsigned char *inner_data = gv_alloc(64 + data_len);
    if (!inner_data) return -1;
    memcpy(inner_data, k_ipad, 64);
    memcpy(inner_data + 64, data, data_len);

    unsigned char inner_hash[32];
    auth_sha256(inner_data, 64 + data_len, inner_hash);
    gv_free(inner_data);

    /* Outer hash: H(K XOR opad || inner_hash) */
    unsigned char outer_data[96];
    memcpy(outer_data, k_opad, 64);
    memcpy(outer_data + 64, inner_hash, 32);

    auth_sha256(outer_data, 96, hmac);

    return 0;
}

/*
 * Constant-time byte comparison.
 *
 * Compares exactly `len` bytes of `a` and `b` in time independent of WHERE
 * they first differ (no early exit), to avoid leaking secret contents via a
 * timing side channel — important when comparing MACs/tags/tokens.
 *
 * Return semantics (unchanged): 0 iff the two buffers are equal over `len`
 * bytes; non-zero otherwise.  Note this is the OPPOSITE polarity of memcmp's
 * "0 == equal but non-zero conveys ordering"; here non-zero only means
 * "differ".  Callers must treat only 0 as a match.  Note also that `len`
 * itself is not secret: this does not hide the length being compared.
 */
int crypto_constant_time_compare(const unsigned char *a,
                                     const unsigned char *b, size_t len) {
    unsigned char result = 0;
    for (size_t i = 0; i < len; i++) {
        result |= a[i] ^ b[i];
    }
    return result;
}

const char *crypto_algorithm_string(GV_CryptoAlgorithm algorithm) {
    switch (algorithm) {
        case GV_CRYPTO_NONE: return "none";
        case GV_CRYPTO_AES_256_CBC: return "AES-256-CBC";
        case GV_CRYPTO_AES_256_GCM: return "AES-256-GCM";
        default: return "unknown";
    }
}

static int b64url_val(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '-') return 62;
    if (c == '_') return 63;
    return -1;
}

int crypto_base64url_decode(const char *in, size_t in_len,
                            unsigned char *out, size_t *out_len) {
    if (!in || !out || !out_len) return -1;
    while (in_len > 0 && in[in_len - 1] == '=') in_len--;

    size_t max_out = (in_len * 3) / 4 + 1;
    if (*out_len < max_out) { *out_len = max_out; return -1; }

    size_t i = 0, j = 0;
    while (i < in_len) {
        int s[4] = {0, 0, 0, 0};
        size_t n = 0;
        for (n = 0; n < 4 && i < in_len; n++, i++) {
            s[n] = b64url_val((unsigned char)in[i]);
            if (s[n] < 0) return -1;
        }
        uint32_t triple = ((uint32_t)s[0] << 18) | ((uint32_t)s[1] << 12) |
                          ((uint32_t)s[2] << 6) | (uint32_t)s[3];
        if (n >= 2) out[j++] = (unsigned char)((triple >> 16) & 0xFF);
        if (n >= 3) out[j++] = (unsigned char)((triple >> 8) & 0xFF);
        if (n >= 4) out[j++] = (unsigned char)(triple & 0xFF);
    }
    *out_len = j;
    return 0;
}
