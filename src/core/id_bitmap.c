/**
 * @file id_bitmap.c
 * @brief Roaring-lite 64-bit id set (see id_bitmap.h).
 *
 * A bitmap is a sorted (by @c key) dynamic array of chunks. Each chunk owns a
 * container over its low-16 values, in one of two representations:
 *   - ARRAY:  sorted uint16_t[] of the low-16 values (cardinality <= 4096).
 *   - BITMAP: dense 8192-byte bitmap covering all 65536 low values.
 * A chunk is promoted ARRAY->BITMAP when its cardinality exceeds 4096; it is
 * never demoted (simplicity), which does not affect correctness.
 */

#include "core/id_bitmap.h"
#include "core/memory.h"

#include <string.h>

/* ---- constants -------------------------------------------------------- */

#define GV_IDBM_CONT_ARRAY  0u
#define GV_IDBM_CONT_BITMAP 1u

/* Array container converts to a bitmap once it grows past this many values. */
#define GV_IDBM_ARRAY_MAX 4096u

/* Bitmap container: 65536 bits = 8192 bytes = 1024 uint64 words. */
#define GV_IDBM_BITMAP_WORDS 1024u
#define GV_IDBM_BITMAP_BYTES (GV_IDBM_BITMAP_WORDS * 8u) /* 8192 */

/* Serialization framing. */
#define GV_IDBM_MAGIC   0x47564942u /* "GVIB" */
#define GV_IDBM_VERSION 1u

/* Sanity caps for deserialize (untrusted input). 2^48 possible keys, but cap
 * chunk count and per-container sizes to bound worst-case allocation. */
#define GV_IDBM_MAX_CHUNKS   ((size_t)1u << 26) /* 67M chunks */
#define GV_IDBM_MAX_ARRAY    65536u             /* a chunk holds <= 65536 vals */

/* ---- structures ------------------------------------------------------- */

typedef struct {
    uint64_t key;       /**< high 48 bits (id >> 16). */
    uint32_t card;      /**< number of low-16 values in this container. */
    uint8_t  type;      /**< GV_IDBM_CONT_ARRAY or GV_IDBM_CONT_BITMAP. */
    union {
        uint16_t *arr;  /**< ARRAY: sorted, @c cap slots allocated. */
        uint64_t *bits; /**< BITMAP: GV_IDBM_BITMAP_WORDS words. */
    } c;
    uint32_t cap;       /**< ARRAY: allocated slots. */
} GV_IdChunk;

struct GV_IdBitmap {
    GV_IdChunk *chunks;
    size_t      nchunks;
    size_t      cap;
};

/* ---- bit helpers ------------------------------------------------------ */

static inline int bitmap_test(const uint64_t *bits, uint16_t v)
{
    return (bits[v >> 6] >> (v & 63u)) & 1u;
}

static inline void bitmap_set(uint64_t *bits, uint16_t v)
{
    bits[v >> 6] |= (uint64_t)1u << (v & 63u);
}

static inline void bitmap_clear_bit(uint64_t *bits, uint16_t v)
{
    bits[v >> 6] &= ~((uint64_t)1u << (v & 63u));
}

/* ---- chunk lookup ----------------------------------------------------- */

/* Binary search for @p key. Returns index if found (*found=1) or the insertion
 * point if not (*found=0). */
static size_t chunk_search(const GV_IdBitmap *b, uint64_t key, int *found)
{
    size_t lo = 0, hi = b->nchunks;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        uint64_t k = b->chunks[mid].key;
        if (k == key) { *found = 1; return mid; }
        if (k < key) lo = mid + 1;
        else hi = mid;
    }
    *found = 0;
    return lo;
}

/* Search within an array container. Returns index if found (*found=1) else
 * insertion point. */
static uint32_t array_search(const uint16_t *arr, uint32_t n, uint16_t v, int *found)
{
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (arr[mid] == v) { *found = 1; return mid; }
        if (arr[mid] < v) lo = mid + 1;
        else hi = mid;
    }
    *found = 0;
    return lo;
}

/* ---- chunk lifecycle -------------------------------------------------- */

static void chunk_fini(GV_IdChunk *ch)
{
    if (ch->type == GV_IDBM_CONT_BITMAP) gv_free(ch->c.bits);
    else gv_free(ch->c.arr);
    ch->c.arr = NULL;
    ch->cap = 0;
    ch->card = 0;
}

/* Ensure room for one more chunk at the tail; returns 0 or -1 on OOM. */
static int bitmap_reserve_chunk(GV_IdBitmap *b)
{
    if (b->nchunks < b->cap) return 0;
    size_t ncap = b->cap ? b->cap * 2 : 8;
    GV_IdChunk *nc = (GV_IdChunk *)gv_realloc(b->chunks, ncap * sizeof(*nc));
    if (!nc) return -1;
    b->chunks = nc;
    b->cap = ncap;
    return 0;
}

/* Insert a fresh empty ARRAY chunk at index @p at (shifting the tail). */
static int chunk_insert_empty(GV_IdBitmap *b, size_t at, uint64_t key)
{
    if (bitmap_reserve_chunk(b) != 0) return -1;
    memmove(&b->chunks[at + 1], &b->chunks[at],
            (b->nchunks - at) * sizeof(b->chunks[0]));
    GV_IdChunk *ch = &b->chunks[at];
    ch->key = key;
    ch->card = 0;
    ch->type = GV_IDBM_CONT_ARRAY;
    ch->cap = 0;
    ch->c.arr = NULL;
    b->nchunks++;
    return 0;
}

static void chunk_remove_at(GV_IdBitmap *b, size_t at)
{
    chunk_fini(&b->chunks[at]);
    memmove(&b->chunks[at], &b->chunks[at + 1],
            (b->nchunks - at - 1) * sizeof(b->chunks[0]));
    b->nchunks--;
}

/* Promote an array container to a bitmap container. Returns 0 or -1 on OOM
 * (leaving the chunk unchanged as an array). */
static int chunk_promote(GV_IdChunk *ch)
{
    uint64_t *bits = (uint64_t *)gv_calloc(GV_IDBM_BITMAP_WORDS, sizeof(uint64_t));
    if (!bits) return -1;
    for (uint32_t i = 0; i < ch->card; i++) bitmap_set(bits, ch->c.arr[i]);
    gv_free(ch->c.arr);
    ch->c.bits = bits;
    ch->type = GV_IDBM_CONT_BITMAP;
    ch->cap = 0;
    return 0;
}

/* Add low value @p v to chunk. Returns 1 if newly added, 0 if already present,
 * -1 on OOM. */
static int chunk_add(GV_IdChunk *ch, uint16_t v)
{
    if (ch->type == GV_IDBM_CONT_BITMAP) {
        if (bitmap_test(ch->c.bits, v)) return 0;
        bitmap_set(ch->c.bits, v);
        ch->card++;
        return 1;
    }

    int found = 0;
    uint32_t pos = array_search(ch->c.arr, ch->card, v, &found);
    if (found) return 0;

    if (ch->card >= GV_IDBM_ARRAY_MAX) {
        if (chunk_promote(ch) != 0) return -1;
        if (bitmap_test(ch->c.bits, v)) return 0; /* defensive */
        bitmap_set(ch->c.bits, v);
        ch->card++;
        return 1;
    }

    if (ch->card >= ch->cap) {
        uint32_t ncap = ch->cap ? ch->cap * 2 : 4;
        if (ncap > GV_IDBM_ARRAY_MAX) ncap = GV_IDBM_ARRAY_MAX;
        uint16_t *na = (uint16_t *)gv_realloc(ch->c.arr, (size_t)ncap * sizeof(uint16_t));
        if (!na) return -1;
        ch->c.arr = na;
        ch->cap = ncap;
    }
    memmove(&ch->c.arr[pos + 1], &ch->c.arr[pos],
            (ch->card - pos) * sizeof(uint16_t));
    ch->c.arr[pos] = v;
    ch->card++;
    return 1;
}

/* Remove low value @p v from chunk. Returns 1 if removed, 0 if absent. */
static int chunk_remove(GV_IdChunk *ch, uint16_t v)
{
    if (ch->type == GV_IDBM_CONT_BITMAP) {
        if (!bitmap_test(ch->c.bits, v)) return 0;
        bitmap_clear_bit(ch->c.bits, v);
        ch->card--;
        return 1;
    }
    int found = 0;
    uint32_t pos = array_search(ch->c.arr, ch->card, v, &found);
    if (!found) return 0;
    memmove(&ch->c.arr[pos], &ch->c.arr[pos + 1],
            (ch->card - pos - 1) * sizeof(uint16_t));
    ch->card--;
    return 1;
}

/* ---- public: lifecycle ------------------------------------------------ */

GV_IdBitmap *gv_id_bitmap_create(void)
{
    GV_IdBitmap *b = (GV_IdBitmap *)gv_calloc(1, sizeof(*b));
    return b; /* NULL on OOM */
}

void gv_id_bitmap_clear(GV_IdBitmap *b)
{
    if (!b) return;
    for (size_t i = 0; i < b->nchunks; i++) chunk_fini(&b->chunks[i]);
    b->nchunks = 0;
}

void gv_id_bitmap_free(GV_IdBitmap *b)
{
    if (!b) return;
    for (size_t i = 0; i < b->nchunks; i++) chunk_fini(&b->chunks[i]);
    gv_free(b->chunks);
    gv_free(b);
}

/* ---- public: single-id ops ------------------------------------------- */

int gv_id_bitmap_add(GV_IdBitmap *b, uint64_t id)
{
    if (!b) return -1;
    uint64_t key = id >> 16;
    uint16_t lo = (uint16_t)(id & 0xFFFFu);
    int found = 0;
    size_t idx = chunk_search(b, key, &found);
    if (!found) {
        if (chunk_insert_empty(b, idx, key) != 0) return -1;
    }
    int r = chunk_add(&b->chunks[idx], lo);
    if (r < 0) {
        /* If we just created an empty chunk, drop it to avoid a dangling
         * zero-card container after OOM. */
        if (!found && b->chunks[idx].card == 0) chunk_remove_at(b, idx);
        return -1;
    }
    return 0;
}

int gv_id_bitmap_remove(GV_IdBitmap *b, uint64_t id)
{
    if (!b) return 0;
    uint64_t key = id >> 16;
    uint16_t lo = (uint16_t)(id & 0xFFFFu);
    int found = 0;
    size_t idx = chunk_search(b, key, &found);
    if (!found) return 0;
    chunk_remove(&b->chunks[idx], lo);
    if (b->chunks[idx].card == 0) chunk_remove_at(b, idx);
    return 0;
}

int gv_id_bitmap_contains(const GV_IdBitmap *b, uint64_t id)
{
    if (!b) return 0;
    uint64_t key = id >> 16;
    uint16_t lo = (uint16_t)(id & 0xFFFFu);
    int found = 0;
    size_t idx = chunk_search(b, key, &found);
    if (!found) return 0;
    const GV_IdChunk *ch = &b->chunks[idx];
    if (ch->type == GV_IDBM_CONT_BITMAP) return bitmap_test(ch->c.bits, lo);
    int f2 = 0;
    (void)array_search(ch->c.arr, ch->card, lo, &f2);
    return f2;
}

uint64_t gv_id_bitmap_cardinality(const GV_IdBitmap *b)
{
    if (!b) return 0;
    uint64_t total = 0;
    for (size_t i = 0; i < b->nchunks; i++) total += b->chunks[i].card;
    return total;
}

/* ---- public: clone ---------------------------------------------------- */

/* Deep-copy the container of @p src into the (uninitialized) @p dst chunk.
 * Returns 0 or -1 on OOM. */
static int chunk_copy(GV_IdChunk *dst, const GV_IdChunk *src)
{
    dst->key = src->key;
    dst->card = src->card;
    dst->type = src->type;
    dst->cap = 0;
    dst->c.arr = NULL;
    if (src->type == GV_IDBM_CONT_BITMAP) {
        uint64_t *bits = (uint64_t *)gv_alloc(GV_IDBM_BITMAP_BYTES);
        if (!bits) return -1;
        memcpy(bits, src->c.bits, GV_IDBM_BITMAP_BYTES);
        dst->c.bits = bits;
    } else if (src->card > 0) {
        uint16_t *arr = (uint16_t *)gv_alloc((size_t)src->card * sizeof(uint16_t));
        if (!arr) return -1;
        memcpy(arr, src->c.arr, (size_t)src->card * sizeof(uint16_t));
        dst->c.arr = arr;
        dst->cap = src->card;
    }
    return 0;
}

GV_IdBitmap *gv_id_bitmap_clone(const GV_IdBitmap *b)
{
    if (!b) return NULL;
    GV_IdBitmap *out = gv_id_bitmap_create();
    if (!out) return NULL;
    if (b->nchunks == 0) return out;

    out->chunks = (GV_IdChunk *)gv_calloc(b->nchunks, sizeof(GV_IdChunk));
    if (!out->chunks) { gv_id_bitmap_free(out); return NULL; }
    out->cap = b->nchunks;

    for (size_t i = 0; i < b->nchunks; i++) {
        if (chunk_copy(&out->chunks[i], &b->chunks[i]) != 0) {
            out->nchunks = i; /* already-copied chunks get freed */
            gv_id_bitmap_free(out);
            return NULL;
        }
        out->nchunks++;
    }
    return out;
}

/* ---- public: set operations ------------------------------------------ */

/* Merge low values of @p src chunk into @p dst chunk (dst |= src). Returns 0 or
 * -1 on OOM. */
static int chunk_or_into(GV_IdChunk *dst, const GV_IdChunk *src)
{
    if (src->type == GV_IDBM_CONT_BITMAP) {
        const uint64_t *sb = src->c.bits;
        for (uint32_t w = 0; w < GV_IDBM_BITMAP_WORDS; w++) {
            uint64_t bits = sb[w];
            while (bits) {
                uint32_t b = (uint32_t)__builtin_ctzll(bits);
                bits &= bits - 1;
                if (chunk_add(dst, (uint16_t)(w * 64u + b)) < 0) return -1;
            }
        }
    } else {
        for (uint32_t i = 0; i < src->card; i++) {
            if (chunk_add(dst, src->c.arr[i]) < 0) return -1;
        }
    }
    return 0;
}

int gv_id_bitmap_or_into(GV_IdBitmap *dst, const GV_IdBitmap *src)
{
    if (!dst) return -1;
    if (!src || src->nchunks == 0) return 0;
    for (size_t i = 0; i < src->nchunks; i++) {
        const GV_IdChunk *s = &src->chunks[i];
        int found = 0;
        size_t idx = chunk_search(dst, s->key, &found);
        if (!found) {
            if (chunk_insert_empty(dst, idx, s->key) != 0) return -1;
        }
        if (chunk_or_into(&dst->chunks[idx], s) != 0) {
            if (!found && dst->chunks[idx].card == 0) chunk_remove_at(dst, idx);
            return -1;
        }
    }
    return 0;
}

GV_IdBitmap *gv_id_bitmap_or(const GV_IdBitmap *a, const GV_IdBitmap *b)
{
    GV_IdBitmap *out;
    if (a) {
        out = gv_id_bitmap_clone(a);
        if (!out) return NULL;
        if (gv_id_bitmap_or_into(out, b) != 0) { gv_id_bitmap_free(out); return NULL; }
    } else if (b) {
        out = gv_id_bitmap_clone(b);
        if (!out) return NULL;
    } else {
        out = gv_id_bitmap_create();
    }
    return out;
}

/* Test membership of low value @p v within a chunk. */
static int chunk_contains(const GV_IdChunk *ch, uint16_t v)
{
    if (ch->type == GV_IDBM_CONT_BITMAP) return bitmap_test(ch->c.bits, v);
    int f = 0;
    (void)array_search(ch->c.arr, ch->card, v, &f);
    return f;
}

/* Intersect two chunks with the same key into fresh chunk @p out (added in
 * ascending order). Returns 0 on success, -1 on OOM. @p out must be empty. */
static int chunk_and_into(GV_IdChunk *out, const GV_IdChunk *x, const GV_IdChunk *y)
{
    /* Iterate the smaller-cardinality side to keep it cheap. */
    const GV_IdChunk *small = (x->card <= y->card) ? x : y;
    const GV_IdChunk *big   = (x->card <= y->card) ? y : x;

    if (small->type == GV_IDBM_CONT_BITMAP) {
        const uint64_t *sb = small->c.bits;
        for (uint32_t w = 0; w < GV_IDBM_BITMAP_WORDS; w++) {
            uint64_t bits = sb[w];
            while (bits) {
                uint32_t bpos = (uint32_t)__builtin_ctzll(bits);
                bits &= bits - 1;
                uint16_t v = (uint16_t)(w * 64u + bpos);
                if (chunk_contains(big, v)) {
                    if (chunk_add(out, v) < 0) return -1;
                }
            }
        }
    } else {
        for (uint32_t i = 0; i < small->card; i++) {
            uint16_t v = small->c.arr[i];
            if (chunk_contains(big, v)) {
                if (chunk_add(out, v) < 0) return -1;
            }
        }
    }
    return 0;
}

GV_IdBitmap *gv_id_bitmap_and(const GV_IdBitmap *a, const GV_IdBitmap *b)
{
    GV_IdBitmap *out = gv_id_bitmap_create();
    if (!out) return NULL;
    if (!a || !b || a->nchunks == 0 || b->nchunks == 0) return out;

    size_t i = 0, j = 0;
    while (i < a->nchunks && j < b->nchunks) {
        uint64_t ka = a->chunks[i].key, kb = b->chunks[j].key;
        if (ka < kb) { i++; continue; }
        if (kb < ka) { j++; continue; }
        /* keys match: append an empty chunk and intersect into it. */
        if (chunk_insert_empty(out, out->nchunks, ka) != 0) {
            gv_id_bitmap_free(out); return NULL;
        }
        GV_IdChunk *oc = &out->chunks[out->nchunks - 1];
        if (chunk_and_into(oc, &a->chunks[i], &b->chunks[j]) != 0) {
            gv_id_bitmap_free(out); return NULL;
        }
        if (oc->card == 0) chunk_remove_at(out, out->nchunks - 1);
        i++; j++;
    }
    return out;
}

/* ---- public: iteration ------------------------------------------------ */

int gv_id_bitmap_iterate(const GV_IdBitmap *b,
                         int (*fn)(uint64_t id, void *ctx), void *ctx)
{
    if (!b || !fn) return 0;
    for (size_t i = 0; i < b->nchunks; i++) {
        const GV_IdChunk *ch = &b->chunks[i];
        uint64_t base = ch->key << 16;
        if (ch->type == GV_IDBM_CONT_BITMAP) {
            const uint64_t *bits = ch->c.bits;
            for (uint32_t w = 0; w < GV_IDBM_BITMAP_WORDS; w++) {
                uint64_t word = bits[w];
                while (word) {
                    uint32_t bpos = (uint32_t)__builtin_ctzll(word);
                    word &= word - 1;
                    uint64_t id = base | (uint64_t)(w * 64u + bpos);
                    int r = fn(id, ctx);
                    if (r) return r;
                }
            }
        } else {
            for (uint32_t k = 0; k < ch->card; k++) {
                uint64_t id = base | (uint64_t)ch->c.arr[k];
                int r = fn(id, ctx);
                if (r) return r;
            }
        }
    }
    return 0;
}

/* ---- serialization ---------------------------------------------------- */
/*
 * Layout (all little-endian):
 *   u32 magic
 *   u32 version
 *   u64 nchunks
 *   repeated nchunks times:
 *     u64 key
 *     u8  type            (0 = array, 1 = bitmap)
 *     u32 card
 *     if array : card * u16 (ascending low values)
 *     if bitmap: 8192 bytes (raw little-endian words)
 *   u32 crc32 over everything preceding it
 */

static void put_u32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v);       p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

static void put_u64(uint8_t *p, uint64_t v)
{
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

static uint32_t get_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t get_u64(const uint8_t *p)
{
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i);
    return v;
}

/* Simple CRC-32 (IEEE, reflected), computed without a static table. */
static uint32_t gv_idbm_crc32(const uint8_t *data, size_t len)
{
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int k = 0; k < 8; k++) {
            uint32_t mask = (uint32_t)-(int32_t)(crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

int gv_id_bitmap_serialize(const GV_IdBitmap *b, uint8_t **out, size_t *out_len)
{
    if (!b || !out || !out_len) return -1;

    /* Compute exact size first. Header (16) + per-chunk + trailing crc (4). */
    size_t total = 4 + 4 + 8; /* magic + version + nchunks */
    for (size_t i = 0; i < b->nchunks; i++) {
        const GV_IdChunk *ch = &b->chunks[i];
        total += 8 + 1 + 4; /* key + type + card */
        if (ch->type == GV_IDBM_CONT_BITMAP) total += GV_IDBM_BITMAP_BYTES;
        else total += (size_t)ch->card * 2u;
    }
    total += 4; /* crc */

    uint8_t *buf = (uint8_t *)gv_alloc(total);
    if (!buf) return -1;

    uint8_t *p = buf;
    put_u32(p, GV_IDBM_MAGIC);   p += 4;
    put_u32(p, GV_IDBM_VERSION); p += 4;
    put_u64(p, (uint64_t)b->nchunks); p += 8;

    for (size_t i = 0; i < b->nchunks; i++) {
        const GV_IdChunk *ch = &b->chunks[i];
        put_u64(p, ch->key); p += 8;
        *p++ = ch->type;
        put_u32(p, ch->card); p += 4;
        if (ch->type == GV_IDBM_CONT_BITMAP) {
            for (uint32_t w = 0; w < GV_IDBM_BITMAP_WORDS; w++) {
                put_u64(p, ch->c.bits[w]); p += 8;
            }
        } else {
            for (uint32_t k = 0; k < ch->card; k++) {
                p[0] = (uint8_t)(ch->c.arr[k]);
                p[1] = (uint8_t)(ch->c.arr[k] >> 8);
                p += 2;
            }
        }
    }

    uint32_t crc = gv_idbm_crc32(buf, (size_t)(p - buf));
    put_u32(p, crc); p += 4;

    /* p should now equal buf + total. */
    *out = buf;
    *out_len = total;
    return 0;
}

GV_IdBitmap *gv_id_bitmap_deserialize(const uint8_t *data, size_t len)
{
    if (!data) return NULL;
    /* Minimum: header (16) + crc (4). */
    if (len < 20) return NULL;

    if (get_u32(data) != GV_IDBM_MAGIC) return NULL;
    if (get_u32(data + 4) != GV_IDBM_VERSION) return NULL;

    /* Verify trailing CRC over everything before it. */
    size_t body = len - 4;
    uint32_t stored = get_u32(data + body);
    if (gv_idbm_crc32(data, body) != stored) return NULL;

    uint64_t nchunks = get_u64(data + 8);
    if (nchunks > GV_IDBM_MAX_CHUNKS) return NULL;

    GV_IdBitmap *b = gv_id_bitmap_create();
    if (!b) return NULL;

    size_t pos = 16; /* just past nchunks */
    uint64_t prev_key = 0;
    int have_prev = 0;

    for (uint64_t ci = 0; ci < nchunks; ci++) {
        /* key(8) + type(1) + card(4) */
        if (pos + 13 > body) { gv_id_bitmap_free(b); return NULL; }
        uint64_t key = get_u64(data + pos); pos += 8;
        uint8_t type = data[pos]; pos += 1;
        uint32_t card = get_u32(data + pos); pos += 4;

        /* Keys must be strictly ascending and fit the 48-bit key space. */
        if (key >= ((uint64_t)1u << 48)) { gv_id_bitmap_free(b); return NULL; }
        if (have_prev && key <= prev_key) { gv_id_bitmap_free(b); return NULL; }
        prev_key = key; have_prev = 1;

        if (type == GV_IDBM_CONT_ARRAY) {
            if (card == 0 || card > GV_IDBM_ARRAY_MAX) { gv_id_bitmap_free(b); return NULL; }
            size_t need = (size_t)card * 2u;
            if (pos + need > body) { gv_id_bitmap_free(b); return NULL; }

            if (bitmap_reserve_chunk(b) != 0) { gv_id_bitmap_free(b); return NULL; }
            GV_IdChunk *ch = &b->chunks[b->nchunks];
            ch->key = key;
            ch->type = GV_IDBM_CONT_ARRAY;
            ch->card = card;
            ch->cap = card;
            ch->c.arr = (uint16_t *)gv_alloc(need);
            if (!ch->c.arr) { gv_id_bitmap_free(b); return NULL; }

            uint16_t prev_v = 0; int have_v = 0;
            for (uint32_t k = 0; k < card; k++) {
                uint16_t v = (uint16_t)((uint16_t)data[pos] |
                                        ((uint16_t)data[pos + 1] << 8));
                pos += 2;
                if (have_v && v <= prev_v) { /* must be strictly ascending */
                    gv_free(ch->c.arr);
                    ch->c.arr = NULL;
                    gv_id_bitmap_free(b);
                    return NULL;
                }
                ch->c.arr[k] = v;
                prev_v = v; have_v = 1;
            }
            b->nchunks++;
        } else if (type == GV_IDBM_CONT_BITMAP) {
            if (pos + GV_IDBM_BITMAP_BYTES > body) { gv_id_bitmap_free(b); return NULL; }

            if (bitmap_reserve_chunk(b) != 0) { gv_id_bitmap_free(b); return NULL; }
            GV_IdChunk *ch = &b->chunks[b->nchunks];
            ch->key = key;
            ch->type = GV_IDBM_CONT_BITMAP;
            ch->cap = 0;
            ch->c.bits = (uint64_t *)gv_alloc(GV_IDBM_BITMAP_BYTES);
            if (!ch->c.bits) { gv_id_bitmap_free(b); return NULL; }

            uint32_t popcount = 0;
            for (uint32_t w = 0; w < GV_IDBM_BITMAP_WORDS; w++) {
                uint64_t word = get_u64(data + pos); pos += 8;
                ch->c.bits[w] = word;
                popcount += (uint32_t)__builtin_popcountll(word);
            }
            /* Stored card must match actual popcount and be > threshold-ish;
             * accept any nonzero popcount to stay lenient but self-consistent. */
            if (popcount == 0 || popcount != card) {
                gv_free(ch->c.bits);
                ch->c.bits = NULL;
                gv_id_bitmap_free(b);
                return NULL;
            }
            ch->card = card;
            b->nchunks++;
        } else {
            gv_id_bitmap_free(b);
            return NULL;
        }
    }

    /* Any trailing bytes between the last chunk and the crc => malformed. */
    if (pos != body) { gv_id_bitmap_free(b); return NULL; }

    return b;
}
