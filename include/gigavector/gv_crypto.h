#ifndef GIGAVECTOR_GV_CRYPTO_H
#define GIGAVECTOR_GV_CRYPTO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_crypto.h
 * @brief Encryption for GigaVector data at rest.
 *
 * Provides AES-256 encryption for database files and sensitive data.
 */

/**
 * @brief Encryption algorithm.
 */
typedef enum {
    GV_CRYPTO_NONE = 0,             /**< No encryption. */
    GV_CRYPTO_AES_256_CBC = 1,      /**< AES-256 CBC mode. */
    GV_CRYPTO_AES_256_GCM = 2       /**< AES-256 GCM mode (authenticated). */
} GV_CryptoAlgorithm;

/**
 * @brief Key derivation function.
 */
typedef enum {
    GV_KDF_NONE = 0,                /**< Use key directly (must be 32 bytes). */
    GV_KDF_PBKDF2 = 1,              /**< PBKDF2-HMAC-SHA256. */
    GV_KDF_SCRYPT = 2               /**< scrypt (memory-hard). */
} GV_KDFType;

/**
 * @brief Crypto configuration.
 */
typedef struct {
    GV_CryptoAlgorithm algorithm;   /**< Encryption algorithm. */
    GV_KDFType kdf;                 /**< Key derivation function. */
    uint32_t kdf_iterations;        /**< KDF iterations (PBKDF2). */
} GV_CryptoConfig;

/**
 * @brief Encryption key.
 */
typedef struct {
    unsigned char key[32];          /**< 256-bit key. */
    unsigned char iv[16];           /**< Initialization vector. */
} GV_CryptoKey;

/**
 * @brief Opaque crypto context handle.
 */
typedef struct GV_CryptoContext GV_CryptoContext;

/* Configuration */

/**
 * @brief Initialize crypto configuration with defaults.
 *
 * Default values:
 * - algorithm: GV_CRYPTO_AES_256_CBC
 * - kdf: GV_KDF_PBKDF2
 * - kdf_iterations: 100000
 *
 * @param config Configuration to initialize.
 */
void gv_crypto_config_init(GV_CryptoConfig *config);

/* Context Lifecycle */

/**
 * @brief Create a crypto context.
 *
 * @param config Crypto configuration (NULL for defaults).
 * @return Crypto context, or NULL on error.
 */
GV_CryptoContext *gv_crypto_create(const GV_CryptoConfig *config);

/**
 * @brief Destroy a crypto context.
 *
 * Securely wipes keys from memory.
 *
 * @param ctx Crypto context (safe to call with NULL).
 */
void gv_crypto_destroy(GV_CryptoContext *ctx);

/* Key Management */

/**
 * @brief Derive a key from a password.
 *
 * @param ctx Crypto context.
 * @param password Password string.
 * @param password_len Password length.
 * @param salt Salt bytes (16 bytes recommended).
 * @param salt_len Salt length.
 * @param key Output key structure.
 * @return 0 on success, -1 on error.
 */
int gv_crypto_derive_key(GV_CryptoContext *ctx, const char *password,
                          size_t password_len, const unsigned char *salt,
                          size_t salt_len, GV_CryptoKey *key);

/**
 * @brief Generate a random key.
 *
 * @param key Output key structure.
 * @return 0 on success, -1 on error.
 */
int gv_crypto_generate_key(GV_CryptoKey *key);

/**
 * @brief Generate a random IV.
 *
 * @param iv Output IV buffer (16 bytes).
 * @return 0 on success, -1 on error.
 */
int gv_crypto_generate_iv(unsigned char *iv);

/**
 * @brief Generate random salt.
 *
 * @param salt Output salt buffer.
 * @param salt_len Salt length (16 bytes recommended).
 * @return 0 on success, -1 on error.
 */
int gv_crypto_generate_salt(unsigned char *salt, size_t salt_len);

/**
 * @brief Securely wipe key from memory.
 *
 * @param key Key to wipe.
 */
void gv_crypto_wipe_key(GV_CryptoKey *key);

/* Encryption/Decryption */

/**
 * @brief Encrypt data.
 *
 * @param ctx Crypto context.
 * @param key Encryption key.
 * @param plaintext Input data.
 * @param plaintext_len Input length.
 * @param ciphertext Output buffer (must be plaintext_len + 16 bytes for padding).
 * @param ciphertext_len Output length.
 * @return 0 on success, -1 on error.
 */
int gv_crypto_encrypt(GV_CryptoContext *ctx, const GV_CryptoKey *key,
                       const unsigned char *plaintext, size_t plaintext_len,
                       unsigned char *ciphertext, size_t *ciphertext_len);

/**
 * @brief Decrypt data.
 *
 * @param ctx Crypto context.
 * @param key Decryption key.
 * @param ciphertext Input data.
 * @param ciphertext_len Input length.
 * @param plaintext Output buffer (can be same size as ciphertext).
 * @param plaintext_len Output length.
 * @return 0 on success, -1 on error.
 */
int gv_crypto_decrypt(GV_CryptoContext *ctx, const GV_CryptoKey *key,
                       const unsigned char *ciphertext, size_t ciphertext_len,
                       unsigned char *plaintext, size_t *plaintext_len);

/* File Encryption */

/**
 * @brief Encrypt a file.
 *
 * @param ctx Crypto context.
 * @param key Encryption key.
 * @param input_path Input file path.
 * @param output_path Output file path.
 * @return 0 on success, -1 on error.
 */
int gv_crypto_encrypt_file(GV_CryptoContext *ctx, const GV_CryptoKey *key,
                            const char *input_path, const char *output_path);

/**
 * @brief Decrypt a file.
 *
 * @param ctx Crypto context.
 * @param key Decryption key.
 * @param input_path Input file path.
 * @param output_path Output file path.
 * @return 0 on success, -1 on error.
 */
int gv_crypto_decrypt_file(GV_CryptoContext *ctx, const GV_CryptoKey *key,
                            const char *input_path, const char *output_path);

/* Stream Encryption */

/**
 * @brief Opaque encryption stream handle.
 */
typedef struct GV_CryptoStream GV_CryptoStream;

/**
 * @brief Create an encryption stream.
 *
 * @param ctx Crypto context.
 * @param key Encryption key.
 * @param encrypting 1 for encryption, 0 for decryption.
 * @return Stream handle, or NULL on error.
 */
GV_CryptoStream *gv_crypto_stream_create(GV_CryptoContext *ctx,
                                          const GV_CryptoKey *key,
                                          int encrypting);

/**
 * @brief Process data through stream.
 *
 * @param stream Stream handle.
 * @param input Input data.
 * @param input_len Input length.
 * @param output Output buffer.
 * @param output_len Output length.
 * @return 0 on success, -1 on error.
 */
int gv_crypto_stream_update(GV_CryptoStream *stream,
                             const unsigned char *input, size_t input_len,
                             unsigned char *output, size_t *output_len);

/**
 * @brief Finalize stream processing.
 *
 * @param stream Stream handle.
 * @param output Output buffer for final block.
 * @param output_len Output length.
 * @return 0 on success, -1 on error.
 */
int gv_crypto_stream_final(GV_CryptoStream *stream,
                            unsigned char *output, size_t *output_len);

/**
 * @brief Destroy stream.
 *
 * @param stream Stream handle (safe to call with NULL).
 */
void gv_crypto_stream_destroy(GV_CryptoStream *stream);

/* Utility Functions */

/**
 * @brief Compute HMAC-SHA256.
 *
 * @param key HMAC key.
 * @param key_len Key length.
 * @param data Input data.
 * @param data_len Data length.
 * @param hmac Output buffer (32 bytes).
 * @return 0 on success, -1 on error.
 */
int gv_crypto_hmac_sha256(const unsigned char *key, size_t key_len,
                           const unsigned char *data, size_t data_len,
                           unsigned char *hmac);

/**
 * @brief Compare two byte arrays in constant time.
 *
 * Prevents timing attacks.
 *
 * @param a First array.
 * @param b Second array.
 * @param len Length to compare.
 * @return 0 if equal, non-zero otherwise.
 */
int gv_crypto_constant_time_compare(const unsigned char *a,
                                     const unsigned char *b, size_t len);

/**
 * @brief Get algorithm name.
 *
 * @param algorithm Algorithm enum.
 * @return Algorithm name string.
 */
const char *gv_crypto_algorithm_string(GV_CryptoAlgorithm algorithm);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_CRYPTO_H */
