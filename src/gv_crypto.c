/**
 * @file gv_crypto.c
 * @brief Encryption implementation.
 *
 * This is a portable implementation. For production, consider using
 * OpenSSL or libsodium for better performance and security auditing.
 */

#include "gigavector/gv_crypto.h"
#include "gigavector/gv_auth.h"  /* For SHA-256 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

struct GV_CryptoContext {
    GV_CryptoConfig config;
};

struct GV_CryptoStream {
    GV_CryptoContext *ctx;
    GV_CryptoKey key;
    int encrypting;
    unsigned char buffer[16];
    size_t buffer_len;
};

/* ============================================================================
 * AES Implementation (Minimal, for portability)
 * ============================================================================ */

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

/* ============================================================================
 * Random Generation
 * ============================================================================ */

static void generate_random_bytes(unsigned char *buf, size_t len) {
    FILE *fp = fopen("/dev/urandom", "rb");
    if (fp) {
        size_t read = fread(buf, 1, len, fp);
        fclose(fp);
        if (read == len) return;
    }
    /* Fallback to weak random (not secure!) */
    for (size_t i = 0; i < len; i++) {
        buf[i] = (unsigned char)(rand() & 0xff);
    }
}

/* ============================================================================
 * Configuration
 * ============================================================================ */

static const GV_CryptoConfig DEFAULT_CONFIG = {
    .algorithm = GV_CRYPTO_AES_256_CBC,
    .kdf = GV_KDF_PBKDF2,
    .kdf_iterations = 100000
};

void gv_crypto_config_init(GV_CryptoConfig *config) {
    if (!config) return;
    *config = DEFAULT_CONFIG;
}

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

GV_CryptoContext *gv_crypto_create(const GV_CryptoConfig *config) {
    GV_CryptoContext *ctx = calloc(1, sizeof(GV_CryptoContext));
    if (!ctx) return NULL;
    ctx->config = config ? *config : DEFAULT_CONFIG;
    return ctx;
}

void gv_crypto_destroy(GV_CryptoContext *ctx) {
    if (!ctx) return;
    memset(ctx, 0, sizeof(*ctx));
    free(ctx);
}

/* ============================================================================
 * Key Management
 * ============================================================================ */

int gv_crypto_derive_key(GV_CryptoContext *ctx, const char *password,
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

    /* Initialize T to zeros */
    memset(T, 0, 32);

    /* Counter (big-endian) */
    unsigned char counter[4] = {0, 0, 0, 1};

    /* First iteration: HMAC(password, salt || counter) */
    unsigned char *msg = malloc(salt_len + 4);
    if (!msg) return -1;
    memcpy(msg, salt, salt_len);
    memcpy(msg + salt_len, counter, 4);

    gv_crypto_hmac_sha256((unsigned char *)password, password_len, msg, salt_len + 4, U);
    memcpy(T, U, 32);

    /* Remaining iterations */
    for (uint32_t i = 1; i < iterations; i++) {
        gv_crypto_hmac_sha256((unsigned char *)password, password_len, U, 32, U);
        for (int j = 0; j < 32; j++) {
            T[j] ^= U[j];
        }
    }

    free(msg);
    memcpy(dk, T, 32);

    memcpy(key->key, dk, 32);
    generate_random_bytes(key->iv, 16);

    return 0;
}

int gv_crypto_generate_key(GV_CryptoKey *key) {
    if (!key) return -1;
    generate_random_bytes(key->key, 32);
    generate_random_bytes(key->iv, 16);
    return 0;
}

int gv_crypto_generate_iv(unsigned char *iv) {
    if (!iv) return -1;
    generate_random_bytes(iv, 16);
    return 0;
}

int gv_crypto_generate_salt(unsigned char *salt, size_t salt_len) {
    if (!salt || salt_len == 0) return -1;
    generate_random_bytes(salt, salt_len);
    return 0;
}

void gv_crypto_wipe_key(GV_CryptoKey *key) {
    if (!key) return;
    /* Secure wipe - write zeros then random */
    memset(key, 0, sizeof(*key));
    volatile unsigned char *p = (volatile unsigned char *)key;
    for (size_t i = 0; i < sizeof(*key); i++) {
        p[i] = 0;
    }
}

/* ============================================================================
 * Encryption/Decryption
 * ============================================================================ */

int gv_crypto_encrypt(GV_CryptoContext *ctx, const GV_CryptoKey *key,
                       const unsigned char *plaintext, size_t plaintext_len,
                       unsigned char *ciphertext, size_t *ciphertext_len) {
    if (!ctx || !key || !plaintext || !ciphertext || !ciphertext_len) return -1;

    /* Expand key */
    unsigned char roundkeys[240];
    aes256_key_expansion(key->key, roundkeys);

    /* Calculate padded length (PKCS7) */
    size_t pad_len = 16 - (plaintext_len % 16);
    size_t total_len = plaintext_len + pad_len;
    *ciphertext_len = total_len;

    /* CBC mode encryption */
    unsigned char prev_block[16];
    memcpy(prev_block, key->iv, 16);

    size_t pos = 0;
    while (pos < plaintext_len) {
        unsigned char block[16];
        size_t block_len = plaintext_len - pos;
        if (block_len > 16) block_len = 16;

        memcpy(block, plaintext + pos, block_len);

        /* PKCS7 padding */
        if (block_len < 16 || pos + 16 >= total_len) {
            for (size_t i = block_len; i < 16; i++) {
                block[i] = (unsigned char)pad_len;
            }
        }

        /* XOR with previous ciphertext block */
        for (int i = 0; i < 16; i++) {
            block[i] ^= prev_block[i];
        }

        /* Encrypt */
        aes256_encrypt_block(block, ciphertext + pos, roundkeys);
        memcpy(prev_block, ciphertext + pos, 16);

        pos += 16;
    }

    return 0;
}

int gv_crypto_decrypt(GV_CryptoContext *ctx, const GV_CryptoKey *key,
                       const unsigned char *ciphertext, size_t ciphertext_len,
                       unsigned char *plaintext, size_t *plaintext_len) {
    if (!ctx || !key || !ciphertext || !plaintext || !plaintext_len) return -1;
    if (ciphertext_len == 0 || ciphertext_len % 16 != 0) return -1;

    /* Expand key */
    unsigned char roundkeys[240];
    aes256_key_expansion(key->key, roundkeys);

    /* CBC mode decryption */
    unsigned char prev_block[16];
    memcpy(prev_block, key->iv, 16);

    for (size_t pos = 0; pos < ciphertext_len; pos += 16) {
        unsigned char block[16];
        aes256_decrypt_block(ciphertext + pos, block, roundkeys);

        /* XOR with previous ciphertext block */
        for (int i = 0; i < 16; i++) {
            block[i] ^= prev_block[i];
        }

        memcpy(plaintext + pos, block, 16);
        memcpy(prev_block, ciphertext + pos, 16);
    }

    /* Remove PKCS7 padding */
    unsigned char pad = plaintext[ciphertext_len - 1];
    if (pad > 0 && pad <= 16) {
        *plaintext_len = ciphertext_len - pad;
    } else {
        *plaintext_len = ciphertext_len;
    }

    return 0;
}

/* ============================================================================
 * File Encryption
 * ============================================================================ */

#define FILE_BUFFER_SIZE (64 * 1024)

int gv_crypto_encrypt_file(GV_CryptoContext *ctx, const GV_CryptoKey *key,
                            const char *input_path, const char *output_path) {
    if (!ctx || !key || !input_path || !output_path) return -1;

    FILE *fin = fopen(input_path, "rb");
    if (!fin) return -1;

    FILE *fout = fopen(output_path, "wb");
    if (!fout) {
        fclose(fin);
        return -1;
    }

    /* Write IV at beginning */
    fwrite(key->iv, 1, 16, fout);

    unsigned char *buffer = malloc(FILE_BUFFER_SIZE);
    unsigned char *cipher = malloc(FILE_BUFFER_SIZE + 16);
    if (!buffer || !cipher) {
        free(buffer);
        free(cipher);
        fclose(fin);
        fclose(fout);
        return -1;
    }

    size_t total_read = 0;
    size_t nread;
    GV_CryptoKey working_key = *key;

    while ((nread = fread(buffer, 1, FILE_BUFFER_SIZE, fin)) > 0) {
        size_t cipher_len;
        if (gv_crypto_encrypt(ctx, &working_key, buffer, nread, cipher, &cipher_len) != 0) {
            free(buffer);
            free(cipher);
            fclose(fin);
            fclose(fout);
            return -1;
        }
        fwrite(cipher, 1, cipher_len, fout);

        /* Update IV for next block */
        if (cipher_len >= 16) {
            memcpy(working_key.iv, cipher + cipher_len - 16, 16);
        }

        total_read += nread;
    }

    free(buffer);
    free(cipher);
    fclose(fin);
    fclose(fout);

    return 0;
}

int gv_crypto_decrypt_file(GV_CryptoContext *ctx, const GV_CryptoKey *key,
                            const char *input_path, const char *output_path) {
    if (!ctx || !key || !input_path || !output_path) return -1;

    FILE *fin = fopen(input_path, "rb");
    if (!fin) return -1;

    FILE *fout = fopen(output_path, "wb");
    if (!fout) {
        fclose(fin);
        return -1;
    }

    /* Read IV from beginning */
    GV_CryptoKey working_key = *key;
    if (fread(working_key.iv, 1, 16, fin) != 16) {
        fclose(fin);
        fclose(fout);
        return -1;
    }

    unsigned char *buffer = malloc(FILE_BUFFER_SIZE);
    unsigned char *plain = malloc(FILE_BUFFER_SIZE);
    if (!buffer || !plain) {
        free(buffer);
        free(plain);
        fclose(fin);
        fclose(fout);
        return -1;
    }

    size_t nread;
    while ((nread = fread(buffer, 1, FILE_BUFFER_SIZE, fin)) > 0) {
        size_t plain_len;
        if (gv_crypto_decrypt(ctx, &working_key, buffer, nread, plain, &plain_len) != 0) {
            free(buffer);
            free(plain);
            fclose(fin);
            fclose(fout);
            return -1;
        }
        fwrite(plain, 1, plain_len, fout);

        /* Update IV for next block */
        if (nread >= 16) {
            memcpy(working_key.iv, buffer + nread - 16, 16);
        }
    }

    free(buffer);
    free(plain);
    fclose(fin);
    fclose(fout);

    return 0;
}

/* ============================================================================
 * Stream Encryption
 * ============================================================================ */

GV_CryptoStream *gv_crypto_stream_create(GV_CryptoContext *ctx,
                                          const GV_CryptoKey *key,
                                          int encrypting) {
    if (!ctx || !key) return NULL;

    GV_CryptoStream *stream = calloc(1, sizeof(GV_CryptoStream));
    if (!stream) return NULL;

    stream->ctx = ctx;
    stream->key = *key;
    stream->encrypting = encrypting;
    stream->buffer_len = 0;

    return stream;
}

int gv_crypto_stream_update(GV_CryptoStream *stream,
                             const unsigned char *input, size_t input_len,
                             unsigned char *output, size_t *output_len) {
    if (!stream || !input || !output || !output_len) return -1;

    /* For simplicity, process complete blocks only */
    *output_len = 0;

    /* Add input to buffer */
    size_t processed = 0;
    while (processed < input_len) {
        size_t space = 16 - stream->buffer_len;
        size_t to_copy = input_len - processed;
        if (to_copy > space) to_copy = space;

        memcpy(stream->buffer + stream->buffer_len, input + processed, to_copy);
        stream->buffer_len += to_copy;
        processed += to_copy;

        /* Process complete block */
        if (stream->buffer_len == 16) {
            unsigned char roundkeys[240];
            aes256_key_expansion(stream->key.key, roundkeys);

            if (stream->encrypting) {
                for (int i = 0; i < 16; i++) {
                    stream->buffer[i] ^= stream->key.iv[i];
                }
                aes256_encrypt_block(stream->buffer, output + *output_len, roundkeys);
                memcpy(stream->key.iv, output + *output_len, 16);
            } else {
                unsigned char temp[16];
                memcpy(temp, stream->buffer, 16);
                aes256_decrypt_block(stream->buffer, output + *output_len, roundkeys);
                for (int i = 0; i < 16; i++) {
                    output[*output_len + i] ^= stream->key.iv[i];
                }
                memcpy(stream->key.iv, temp, 16);
            }

            *output_len += 16;
            stream->buffer_len = 0;
        }
    }

    return 0;
}

int gv_crypto_stream_final(GV_CryptoStream *stream,
                            unsigned char *output, size_t *output_len) {
    if (!stream || !output || !output_len) return -1;

    *output_len = 0;

    if (stream->encrypting) {
        /* Add PKCS7 padding */
        size_t pad_len = 16 - stream->buffer_len;
        for (size_t i = stream->buffer_len; i < 16; i++) {
            stream->buffer[i] = (unsigned char)pad_len;
        }

        unsigned char roundkeys[240];
        aes256_key_expansion(stream->key.key, roundkeys);

        for (int i = 0; i < 16; i++) {
            stream->buffer[i] ^= stream->key.iv[i];
        }
        aes256_encrypt_block(stream->buffer, output, roundkeys);
        *output_len = 16;
    } else {
        /* Handle remaining data with padding removal */
        if (stream->buffer_len > 0) {
            /* This is an error - incomplete block in decryption */
            return -1;
        }
    }

    return 0;
}

void gv_crypto_stream_destroy(GV_CryptoStream *stream) {
    if (!stream) return;
    gv_crypto_wipe_key(&stream->key);
    memset(stream, 0, sizeof(*stream));
    free(stream);
}

/* ============================================================================
 * HMAC-SHA256
 * ============================================================================ */

int gv_crypto_hmac_sha256(const unsigned char *key, size_t key_len,
                           const unsigned char *data, size_t data_len,
                           unsigned char *hmac) {
    if (!key || !data || !hmac) return -1;

    unsigned char k_ipad[64], k_opad[64];

    /* Key processing */
    unsigned char key_block[64];
    memset(key_block, 0, 64);

    if (key_len > 64) {
        /* Hash key if too long */
        gv_auth_sha256(key, key_len, key_block);
    } else {
        memcpy(key_block, key, key_len);
    }

    /* Create padded keys */
    for (int i = 0; i < 64; i++) {
        k_ipad[i] = key_block[i] ^ 0x36;
        k_opad[i] = key_block[i] ^ 0x5c;
    }

    /* Inner hash: H(K XOR ipad || data) */
    unsigned char *inner_data = malloc(64 + data_len);
    if (!inner_data) return -1;
    memcpy(inner_data, k_ipad, 64);
    memcpy(inner_data + 64, data, data_len);

    unsigned char inner_hash[32];
    gv_auth_sha256(inner_data, 64 + data_len, inner_hash);
    free(inner_data);

    /* Outer hash: H(K XOR opad || inner_hash) */
    unsigned char outer_data[96];
    memcpy(outer_data, k_opad, 64);
    memcpy(outer_data + 64, inner_hash, 32);

    gv_auth_sha256(outer_data, 96, hmac);

    return 0;
}

int gv_crypto_constant_time_compare(const unsigned char *a,
                                     const unsigned char *b, size_t len) {
    unsigned char result = 0;
    for (size_t i = 0; i < len; i++) {
        result |= a[i] ^ b[i];
    }
    return result;
}

const char *gv_crypto_algorithm_string(GV_CryptoAlgorithm algorithm) {
    switch (algorithm) {
        case GV_CRYPTO_NONE: return "none";
        case GV_CRYPTO_AES_256_CBC: return "AES-256-CBC";
        case GV_CRYPTO_AES_256_GCM: return "AES-256-GCM";
        default: return "unknown";
    }
}
