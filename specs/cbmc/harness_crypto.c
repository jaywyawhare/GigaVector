/*
 * CBMC harness: the self-describing ciphertext framing (src/security/crypto.c).
 *
 * MEMORY.md records that the format is self-describing -- a nonce/IV prefix
 * followed by length-framed chunks -- and that crypto_stream was reimplemented
 * around an IV prefix with PKCS7 hold-back. Any parser reading a length field
 * out of attacker-supplied bytes is a CBMC target.
 *
 * Two properties: encrypt->decrypt is the identity, and decrypting arbitrary
 * bytes is memory-safe (it must reject, not overrun).
  *
 * CBMC-SOURCES: src/security/crypto.c src/core/memory.c
 * CBMC-UNWIND: 64
*/
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "security/crypto.h"

#if defined(__CPROVER__) || defined(__CPROVER)
uint64_t nondet_u64(void);
size_t   nondet_size(void);
unsigned char nondet_uchar(void);
float    nondet_float(void);
#  define ASSUME(c) __CPROVER_assume(c)
#else
static uint64_t g_seed = 0x9E3779B97F4A7C15ULL;
static uint64_t nondet_u64(void) {
    g_seed ^= g_seed << 13; g_seed ^= g_seed >> 7; g_seed ^= g_seed << 17;
    return g_seed;
}
static size_t nondet_size(void) { return (size_t)(nondet_u64() % 8); }
static unsigned char nondet_uchar(void) { return (unsigned char)(nondet_u64() & 0xFF); }
static float nondet_float(void) {
    return (float)((int)(nondet_u64() % 2000) - 1000) / 100.0f;
}
#  define ASSUME(c) do { if (!(c)) return 0; } while (0)
#endif

#define PT_MAX 16
#define CT_MAX 128

int main(void) {
    GV_CryptoConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    GV_CryptoContext *ctx = crypto_create(&cfg);
    if (!ctx) return 0;

    GV_CryptoKey key;
    memset(&key, 0, sizeof(key));
    if (crypto_generate_key(&key) != 0) return 0;

    size_t plen = nondet_size();
    ASSUME(plen >= 1 && plen <= PT_MAX);

    unsigned char pt[PT_MAX];
    for (size_t i = 0; i < PT_MAX; i++) pt[i] = nondet_uchar();

    unsigned char ct[CT_MAX];
    size_t clen = CT_MAX;
    if (crypto_encrypt(ctx, &key, pt, plen, ct, &clen) == 0) {
        assert(clen <= CT_MAX);                 /* never writes past the buffer */

        unsigned char out[CT_MAX];
        size_t olen = CT_MAX;
        if (crypto_decrypt(ctx, &key, ct, clen, out, &olen) == 0) {
            assert(olen == plen);               /* round-trip length */
            for (size_t i = 0; i < plen; i++)
                assert(out[i] == pt[i]);        /* ... and content */
        }
    }

    /* Decrypting arbitrary bytes must fail safely rather than overrun: the
     * length prefix is attacker-controlled. */
    unsigned char garbage[PT_MAX];
    for (size_t i = 0; i < PT_MAX; i++) garbage[i] = nondet_uchar();
    unsigned char sink[CT_MAX];
    size_t slen = CT_MAX;
    (void)crypto_decrypt(ctx, &key, garbage, plen, sink, &slen);

    return 0;
}
