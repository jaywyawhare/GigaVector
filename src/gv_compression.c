/**
 * @file gv_compression.c
 * @brief LZ4-style fast metadata/payload compression implementation.
 *
 * Implements a simple LZ77-style compressor with no external dependencies.
 * Uses a hash table of 4-byte sequences for fast match finding and a
 * sliding window for back-references.
 *
 * Compressed block format:
 *   [literal_length (varint)] [literal_bytes]
 *   [match_offset (2 bytes, little-endian)] [match_length (varint)]
 *
 * A literal_length of 0 with match_offset of 0 signals end-of-stream.
 */

#include "gigavector/gv_compression.h"

#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

/*  Constants  */

#define GV_HASH_BITS      14
#define GV_HASH_SIZE      (1 << GV_HASH_BITS)   /* 16384 entries */
#define GV_HASH_MASK      (GV_HASH_SIZE - 1)

#define GV_MIN_MATCH      4      /* Minimum match length (hash is over 4 bytes) */
#define GV_MAX_MATCH      271    /* 15 + 256: max encodable in token + 1 extra byte */
#define GV_WINDOW_SIZE    65535  /* Max back-reference offset (fits in 2 bytes) */

#define GV_SEARCH_DEPTH_LOW   4   /* Levels 1-3 */
#define GV_SEARCH_DEPTH_HIGH  16  /* Levels 4-9 */

#define GV_END_MARGIN     5      /* Stop searching this many bytes before end */

/*  Internal Structures  */

struct GV_Compressor {
    GV_CompressionConfig config;

    /* Hash table for match finding (reused across calls) */
    uint32_t hash_table[GV_HASH_SIZE];

    /* Chain table for deeper search at higher compression levels.
     * Each entry points to the previous position with the same hash. */
    uint32_t chain[GV_WINDOW_SIZE + 1];

    /* Thread-safe statistics */
    _Atomic uint64_t total_compressed;
    _Atomic uint64_t total_decompressed;
    _Atomic uint64_t bytes_in;
    _Atomic uint64_t bytes_out;
};

/*  Hash Function  */

/**
 * @brief Fast 4-byte hash for match finding.
 *
 * Uses a multiplicative hash similar to the one used in LZ4.
 */
static inline uint32_t gv_hash4(const uint8_t *p)
{
    uint32_t v;
    memcpy(&v, p, 4);
    return (v * 2654435761U) >> (32 - GV_HASH_BITS);
}

/*  Variable-Length Integer Encoding  */

/**
 * @brief Encode a length value using a variable-length scheme.
 *
 * If the value is < 255 it is stored as a single byte. Otherwise 255 is stored
 * repeatedly, and the remainder is stored as the final byte.
 *
 * @return Number of bytes written, or 0 if the buffer is too small.
 */
static size_t encode_varint(uint8_t *dst, size_t dst_cap, size_t value)
{
    size_t written = 0;
    while (value >= 255) {
        if (written >= dst_cap) return 0;
        dst[written++] = 255;
        value -= 255;
    }
    if (written >= dst_cap) return 0;
    dst[written++] = (uint8_t)value;
    return written;
}

/**
 * @brief Decode a variable-length integer.
 *
 * @param src       Input buffer.
 * @param src_len   Remaining bytes in input.
 * @param value_out Decoded value.
 * @return Number of bytes consumed, or 0 on error.
 */
static size_t decode_varint(const uint8_t *src, size_t src_len, size_t *value_out)
{
    size_t value = 0;
    size_t consumed = 0;
    while (consumed < src_len) {
        uint8_t byte = src[consumed++];
        value += byte;
        if (byte < 255) {
            *value_out = value;
            return consumed;
        }
    }
    /* Ran out of input without a terminating byte < 255. */
    return 0;
}

/*  Compression  */

/**
 * @brief Find the length of the match between two positions.
 */
static size_t match_length(const uint8_t *a, const uint8_t *b,
                           const uint8_t *a_end)
{
    const uint8_t *a_start = a;
    size_t limit = (size_t)(a_end - a);
    if (limit > GV_MAX_MATCH) {
        limit = GV_MAX_MATCH;
    }
    while ((size_t)(a - a_start) < limit && *a == *b) {
        a++;
        b++;
    }
    return (size_t)(a - a_start);
}

/**
 * @brief Core LZ77-style compression.
 *
 * Algorithm:
 *   1. For each position, compute the 4-byte hash.
 *   2. Look up the hash table (and optionally the chain) for candidate matches.
 *   3. If a match of length >= GV_MIN_MATCH is found, emit any pending literals,
 *      then emit the match (offset + length).
 *   4. Otherwise, accumulate the byte as a literal.
 *   5. At end-of-input, emit remaining literals and a termination marker.
 */
static size_t compress_core(GV_Compressor *comp,
                            const uint8_t *src, size_t src_len,
                            uint8_t *dst, size_t dst_cap)
{
    if (src_len == 0) {
        /* Emit end marker: literal_len=0, offset=0 */
        if (dst_cap < 3) return 0;
        dst[0] = 0;  /* literal length 0 */
        dst[1] = 0;  /* offset low */
        dst[2] = 0;  /* offset high */
        return 3;
    }

    int search_depth = (comp->config.level <= 3)
                       ? GV_SEARCH_DEPTH_LOW
                       : GV_SEARCH_DEPTH_HIGH;

    /* Reset hash and chain tables */
    memset(comp->hash_table, 0, sizeof(comp->hash_table));

    const uint8_t *src_end = src + src_len;
    const uint8_t *match_limit = src_end - GV_END_MARGIN;
    if (src_len <= GV_END_MARGIN) {
        match_limit = src; /* No matches possible for very short input */
    }

    size_t ip = 0;       /* Current input position */
    size_t anchor = 0;    /* Start of un-emitted literals */
    size_t op = 0;        /* Current output position */

    while (ip < src_len) {
        size_t best_len = 0;
        size_t best_off = 0;

        /* Only attempt match finding if we have at least 4 bytes left
         * and we are within the match limit. */
        if (ip + GV_MIN_MATCH <= src_len && src + ip < match_limit) {
            uint32_t h = gv_hash4(src + ip);
            uint32_t candidate = comp->hash_table[h];

            /* Walk the chain for candidate matches */
            uint32_t cur_candidate = candidate;
            for (int d = 0; d < search_depth; d++) {
                if (cur_candidate == 0 && d > 0) break;
                if (d == 0) cur_candidate = candidate;
                else {
                    if (cur_candidate >= GV_WINDOW_SIZE + 1) break;
                    cur_candidate = comp->chain[cur_candidate % (GV_WINDOW_SIZE + 1)];
                }

                if (cur_candidate > ip) continue;
                size_t offset = ip - cur_candidate;
                if (offset == 0 || offset > GV_WINDOW_SIZE) continue;

                size_t ml = match_length(src + ip, src + cur_candidate, src_end);
                if (ml >= GV_MIN_MATCH && ml > best_len) {
                    best_len = ml;
                    best_off = offset;
                    if (best_len >= GV_MAX_MATCH) break;
                }
            }

            /* Update hash table and chain */
            comp->chain[ip % (GV_WINDOW_SIZE + 1)] = comp->hash_table[h];
            comp->hash_table[h] = (uint32_t)ip;
        }

        if (best_len >= GV_MIN_MATCH) {
            /* Emit literals from anchor to ip */
            size_t lit_len = ip - anchor;

            /* Encode literal length */
            size_t n = encode_varint(dst + op, dst_cap - op, lit_len);
            if (n == 0) return 0;
            op += n;

            /* Copy literal bytes */
            if (op + lit_len > dst_cap) return 0;
            memcpy(dst + op, src + anchor, lit_len);
            op += lit_len;

            /* Encode match offset (2 bytes, little-endian) */
            if (op + 2 > dst_cap) return 0;
            dst[op++] = (uint8_t)(best_off & 0xFF);
            dst[op++] = (uint8_t)((best_off >> 8) & 0xFF);

            /* Encode match length (subtract GV_MIN_MATCH to save space) */
            size_t encoded_ml = best_len - GV_MIN_MATCH;
            n = encode_varint(dst + op, dst_cap - op, encoded_ml);
            if (n == 0) return 0;
            op += n;

            /* Update hash for all positions in the match for better future matching */
            for (size_t j = 1; j < best_len && ip + j + GV_MIN_MATCH <= src_len; j++) {
                uint32_t mh = gv_hash4(src + ip + j);
                comp->chain[(ip + j) % (GV_WINDOW_SIZE + 1)] = comp->hash_table[mh];
                comp->hash_table[mh] = (uint32_t)(ip + j);
            }

            ip += best_len;
            anchor = ip;
        } else {
            ip++;
        }
    }

    /* Emit remaining literals */
    size_t lit_len = src_len - anchor;
    if (lit_len > 0) {
        size_t n = encode_varint(dst + op, dst_cap - op, lit_len);
        if (n == 0) return 0;
        op += n;

        if (op + lit_len > dst_cap) return 0;
        memcpy(dst + op, src + anchor, lit_len);
        op += lit_len;

        /* Write zero offset to indicate no match follows these literals */
        if (op + 2 > dst_cap) return 0;
        dst[op++] = 0;
        dst[op++] = 0;
    }

    /* Emit end-of-stream marker: literal_len=0, offset=0 */
    size_t n = encode_varint(dst + op, dst_cap - op, 0);
    if (n == 0) return 0;
    op += n;

    if (op + 2 > dst_cap) return 0;
    dst[op++] = 0;
    dst[op++] = 0;

    return op;
}

/*  Decompression  */

/**
 * @brief Core decompression: sequential decode of compressed blocks.
 *
 * Reads blocks of [literal_length, literal_bytes, match_offset, match_length]
 * until the end-of-stream marker is encountered.
 */
static size_t decompress_core(const uint8_t *src, size_t src_len,
                              uint8_t *dst, size_t dst_cap)
{
    size_t ip = 0;  /* Input position */
    size_t op = 0;  /* Output position */

    while (ip < src_len) {
        /* Decode literal length */
        size_t lit_len = 0;
        size_t n = decode_varint(src + ip, src_len - ip, &lit_len);
        if (n == 0) return 0;
        ip += n;

        /* Copy literal bytes */
        if (lit_len > 0) {
            if (ip + lit_len > src_len) return 0;
            if (op + lit_len > dst_cap) return 0;
            memcpy(dst + op, src + ip, lit_len);
            ip += lit_len;
            op += lit_len;
        }

        /* Read match offset (2 bytes, little-endian) */
        if (ip + 2 > src_len) return 0;
        size_t offset = (size_t)src[ip] | ((size_t)src[ip + 1] << 8);
        ip += 2;

        /* End-of-stream check: literal_len == 0 && offset == 0 */
        if (lit_len == 0 && offset == 0) {
            break;
        }

        if (offset == 0) {
            /* Literals followed by no match -- continue to next block */
            continue;
        }

        /* Decode match length */
        size_t match_len_encoded = 0;
        n = decode_varint(src + ip, src_len - ip, &match_len_encoded);
        if (n == 0) return 0;
        ip += n;

        size_t ml = match_len_encoded + GV_MIN_MATCH;

        /* Validate offset and copy match */
        if (offset > op) return 0;  /* Invalid back-reference */
        if (op + ml > dst_cap) return 0;

        /* Copy match bytes. Use byte-by-byte copy to handle overlapping
         * references (e.g., run-length encoding patterns). */
        size_t match_pos = op - offset;
        for (size_t i = 0; i < ml; i++) {
            dst[op + i] = dst[match_pos + i];
        }
        op += ml;
    }

    return op;
}

/*  Configuration  */

static const GV_CompressionConfig DEFAULT_CONFIG = {
    .type     = GV_COMPRESS_LZ4,
    .level    = 1,
    .min_size = 64
};

void gv_compression_config_init(GV_CompressionConfig *config)
{
    if (!config) return;
    *config = DEFAULT_CONFIG;
}

/*  Lifecycle  */

GV_Compressor *gv_compression_create(const GV_CompressionConfig *config)
{
    GV_Compressor *comp = (GV_Compressor *)calloc(1, sizeof(GV_Compressor));
    if (!comp) return NULL;

    comp->config = config ? *config : DEFAULT_CONFIG;

    /* Clamp compression level to valid range */
    if (comp->config.level < 1) comp->config.level = 1;
    if (comp->config.level > 9) comp->config.level = 9;

    atomic_store(&comp->total_compressed, 0);
    atomic_store(&comp->total_decompressed, 0);
    atomic_store(&comp->bytes_in, 0);
    atomic_store(&comp->bytes_out, 0);

    return comp;
}

void gv_compression_destroy(GV_Compressor *comp)
{
    if (!comp) return;
    free(comp);
}

/*  Public API  */

size_t gv_compress(GV_Compressor *comp, const void *input, size_t input_len,
                   void *output, size_t output_capacity)
{
    if (!comp || !input || !output) return 0;
    if (input_len == 0) return 0;

    const uint8_t *src = (const uint8_t *)input;
    uint8_t *dst = (uint8_t *)output;

    /* If compression type is NONE, or the input is below the minimum size
     * threshold, store uncompressed with a 1-byte header. */
    if (comp->config.type == GV_COMPRESS_NONE || input_len < comp->config.min_size) {
        /* Uncompressed format: [0x00] [raw data] */
        if (output_capacity < 1 + input_len) return 0;
        dst[0] = 0x00;  /* marker: uncompressed */
        memcpy(dst + 1, src, input_len);

        atomic_fetch_add(&comp->total_compressed, 1);
        atomic_fetch_add(&comp->bytes_in, input_len);
        atomic_fetch_add(&comp->bytes_out, 1 + input_len);
        return 1 + input_len;
    }

    /* Compressed format: [0x01] [compressed data] */
    if (output_capacity < 1) return 0;
    dst[0] = 0x01;  /* marker: compressed */

    size_t compressed_size = compress_core(comp, src, input_len,
                                           dst + 1, output_capacity - 1);

    if (compressed_size == 0 || compressed_size + 1 >= input_len + 1) {
        /* Compression did not help -- fall back to uncompressed. */
        if (output_capacity < 1 + input_len) return 0;
        dst[0] = 0x00;
        memcpy(dst + 1, src, input_len);

        atomic_fetch_add(&comp->total_compressed, 1);
        atomic_fetch_add(&comp->bytes_in, input_len);
        atomic_fetch_add(&comp->bytes_out, 1 + input_len);
        return 1 + input_len;
    }

    size_t total = 1 + compressed_size;

    atomic_fetch_add(&comp->total_compressed, 1);
    atomic_fetch_add(&comp->bytes_in, input_len);
    atomic_fetch_add(&comp->bytes_out, total);

    return total;
}

size_t gv_decompress(GV_Compressor *comp, const void *input, size_t input_len,
                     void *output, size_t output_capacity)
{
    if (!comp || !input || !output) return 0;
    if (input_len < 1) return 0;

    const uint8_t *src = (const uint8_t *)input;
    uint8_t *dst = (uint8_t *)output;

    uint8_t marker = src[0];

    if (marker == 0x00) {
        /* Uncompressed: copy raw data after the marker byte */
        size_t raw_len = input_len - 1;
        if (raw_len > output_capacity) return 0;
        memcpy(dst, src + 1, raw_len);

        atomic_fetch_add(&comp->total_decompressed, 1);
        atomic_fetch_add(&comp->bytes_in, input_len);
        atomic_fetch_add(&comp->bytes_out, raw_len);
        return raw_len;
    }

    if (marker == 0x01) {
        /* Compressed data follows */
        size_t decompressed_size = decompress_core(src + 1, input_len - 1,
                                                    dst, output_capacity);
        if (decompressed_size == 0) return 0;

        atomic_fetch_add(&comp->total_decompressed, 1);
        atomic_fetch_add(&comp->bytes_in, input_len);
        atomic_fetch_add(&comp->bytes_out, decompressed_size);
        return decompressed_size;
    }

    /* Unknown marker byte */
    return 0;
}

size_t gv_compress_bound(const GV_Compressor *comp, size_t input_len)
{
    (void)comp;
    /* Worst case: 1-byte marker + input + overhead for varint lengths.
     * Formula: input_len + input_len/255 + 16 (per specification). */
    return input_len + input_len / 255 + 16;
}

int gv_compression_get_stats(const GV_Compressor *comp, GV_CompressionStats *stats)
{
    if (!comp || !stats) return -1;

    stats->total_compressed   = atomic_load(&((GV_Compressor *)comp)->total_compressed);
    stats->total_decompressed = atomic_load(&((GV_Compressor *)comp)->total_decompressed);
    stats->bytes_in           = atomic_load(&((GV_Compressor *)comp)->bytes_in);
    stats->bytes_out          = atomic_load(&((GV_Compressor *)comp)->bytes_out);

    if (stats->bytes_in > 0 && stats->total_compressed > 0) {
        /* Average ratio = total bytes out from compression / total bytes in */
        stats->avg_ratio = (double)stats->bytes_out / (double)stats->bytes_in;
    } else {
        stats->avg_ratio = 1.0;
    }

    return 0;
}
