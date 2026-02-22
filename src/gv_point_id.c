#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gigavector/gv_point_id.h"

/*  Constants                                                          */

#define GV_POINTID_DEFAULT_CAPACITY 64
#define GV_POINTID_LOAD_FACTOR      0.7
#define GV_POINTID_UUID_LEN         36   /* "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx" */
#define GV_POINTID_UUID_BUF_MIN     37   /* UUID + NUL */

/*  FNV-1a 64-bit hash                                                 */

#define FNV_OFFSET_BASIS UINT64_C(14695981039346656037)
#define FNV_PRIME        UINT64_C(1099511628211)

static uint64_t fnv1a_hash(const char *str)
{
    const uint8_t *p = (const uint8_t *)str;
    uint64_t hash = FNV_OFFSET_BASIS;
    while (*p) {
        hash ^= (uint64_t)*p++;
        hash *= FNV_PRIME;
    }
    return hash;
}

/*  Hash-table entry                                                   */

typedef struct {
    char    *string_id;      /* Owned copy of the user string.  NULL if empty. */
    size_t   internal_index; /* Associated internal vector index. */
    uint64_t hash;           /* Cached FNV-1a hash of string_id. */
    int      occupied;       /* 1 = live entry, 0 = empty / tombstone. */
} GV_PointIDEntry;

/*  Reverse-lookup array                                               */

typedef struct {
    char  **ids;       /* Array of pointers into forward-table string_id's. */
    size_t  capacity;  /* Allocated slots. */
} GV_ReverseMap;

/*  Map structure                                                      */

struct GV_PointIDMap {
    GV_PointIDEntry *buckets;       /* Open-addressing hash table. */
    size_t           capacity;      /* Number of buckets (always power of 2). */
    size_t           count;         /* Number of live entries. */
    GV_ReverseMap    reverse;       /* index -> string_id pointer. */
    pthread_rwlock_t rwlock;        /* Reader-writer lock. */
};

/*  Helper: next power of two >= n (minimum 1)                         */

static size_t next_pow2(size_t n)
{
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
#if SIZE_MAX > 0xFFFFFFFFU
    n |= n >> 32;
#endif
    return n + 1;
}

/*  Internal: find bucket for a key (open addressing, linear probing)  */

/**
 * @brief Locate the bucket for @p key, or the first empty slot.
 *
 * @param buckets   Bucket array.
 * @param capacity  Number of buckets.
 * @param key       NUL-terminated string to look up.
 * @param hash      Precomputed FNV-1a hash of @p key.
 * @param out_idx   Output: bucket index.
 * @return 1 if the key was found (exact match), 0 if the slot is empty.
 */
static int find_bucket(const GV_PointIDEntry *buckets, size_t capacity,
                       const char *key, uint64_t hash, size_t *out_idx)
{
    size_t mask = capacity - 1;
    size_t idx  = (size_t)(hash & mask);

    for (size_t i = 0; i < capacity; i++) {
        size_t probe = (idx + i) & mask;
        const GV_PointIDEntry *e = &buckets[probe];

        if (!e->occupied) {
            *out_idx = probe;
            return 0; /* Empty slot. */
        }
        if (e->hash == hash && strcmp(e->string_id, key) == 0) {
            *out_idx = probe;
            return 1; /* Found. */
        }
    }

    /* Should never happen when load factor < 1.0 */
    *out_idx = 0;
    return 0;
}

/*  Internal: grow the reverse-lookup array if needed                   */

static int reverse_ensure(GV_ReverseMap *rev, size_t needed_index)
{
    if (needed_index < rev->capacity) {
        return 0;
    }

    size_t new_cap = rev->capacity == 0 ? 64 : rev->capacity;
    while (new_cap <= needed_index) {
        new_cap *= 2;
    }

    char **new_ids = (char **)realloc(rev->ids, new_cap * sizeof(char *));
    if (!new_ids) {
        return -1;
    }

    /* Zero-fill the new portion. */
    memset(new_ids + rev->capacity, 0, (new_cap - rev->capacity) * sizeof(char *));

    rev->ids      = new_ids;
    rev->capacity = new_cap;
    return 0;
}

/*  Internal: resize the hash table                                    */

static int map_resize(GV_PointIDMap *map, size_t new_capacity)
{
    new_capacity = next_pow2(new_capacity);

    GV_PointIDEntry *new_buckets = (GV_PointIDEntry *)calloc(new_capacity,
                                                              sizeof(GV_PointIDEntry));
    if (!new_buckets) {
        return -1;
    }

    /* Re-insert every live entry. */
    for (size_t i = 0; i < map->capacity; i++) {
        GV_PointIDEntry *old = &map->buckets[i];
        if (!old->occupied) {
            continue;
        }

        size_t new_idx;
        find_bucket(new_buckets, new_capacity, old->string_id, old->hash, &new_idx);

        new_buckets[new_idx] = *old; /* Shallow copy; we keep the same string_id pointer. */
    }

    free(map->buckets);
    map->buckets  = new_buckets;
    map->capacity = new_capacity;
    return 0;
}

/*  Public API: Create / Destroy                                       */

GV_PointIDMap *gv_point_id_create(size_t initial_capacity)
{
    if (initial_capacity == 0) {
        initial_capacity = GV_POINTID_DEFAULT_CAPACITY;
    }
    initial_capacity = next_pow2(initial_capacity);

    GV_PointIDMap *map = (GV_PointIDMap *)calloc(1, sizeof(GV_PointIDMap));
    if (!map) {
        return NULL;
    }

    if (pthread_rwlock_init(&map->rwlock, NULL) != 0) {
        free(map);
        return NULL;
    }

    map->buckets = (GV_PointIDEntry *)calloc(initial_capacity, sizeof(GV_PointIDEntry));
    if (!map->buckets) {
        pthread_rwlock_destroy(&map->rwlock);
        free(map);
        return NULL;
    }

    map->capacity       = initial_capacity;
    map->count          = 0;
    map->reverse.ids    = NULL;
    map->reverse.capacity = 0;

    return map;
}

void gv_point_id_destroy(GV_PointIDMap *map)
{
    if (!map) {
        return;
    }

    /* Free all owned string copies. */
    for (size_t i = 0; i < map->capacity; i++) {
        if (map->buckets[i].occupied) {
            free(map->buckets[i].string_id);
        }
    }

    free(map->buckets);
    free(map->reverse.ids);
    pthread_rwlock_destroy(&map->rwlock);
    free(map);
}

/*  Public API: Map Operations                                         */

int gv_point_id_set(GV_PointIDMap *map, const char *string_id, size_t internal_index)
{
    if (!map || !string_id) {
        return -1;
    }

    pthread_rwlock_wrlock(&map->rwlock);

    /* Resize if load factor would exceed threshold. */
    if ((double)(map->count + 1) / (double)map->capacity > GV_POINTID_LOAD_FACTOR) {
        if (map_resize(map, map->capacity * 2) != 0) {
            pthread_rwlock_unlock(&map->rwlock);
            return -1;
        }
    }

    uint64_t hash = fnv1a_hash(string_id);
    size_t   idx;
    int      found = find_bucket(map->buckets, map->capacity, string_id, hash, &idx);

    if (found) {
        /* Update existing entry. */
        size_t old_index = map->buckets[idx].internal_index;

        /* Clear old reverse pointer if it pointed to this string. */
        if (old_index < map->reverse.capacity && map->reverse.ids[old_index] == map->buckets[idx].string_id) {
            map->reverse.ids[old_index] = NULL;
        }

        map->buckets[idx].internal_index = internal_index;

        /* Update reverse map. */
        if (reverse_ensure(&map->reverse, internal_index) == 0) {
            map->reverse.ids[internal_index] = map->buckets[idx].string_id;
        }

        pthread_rwlock_unlock(&map->rwlock);
        return 0;
    }

    /* New entry. */
    char *id_copy = strdup(string_id);
    if (!id_copy) {
        pthread_rwlock_unlock(&map->rwlock);
        return -1;
    }

    map->buckets[idx].string_id      = id_copy;
    map->buckets[idx].internal_index = internal_index;
    map->buckets[idx].hash           = hash;
    map->buckets[idx].occupied       = 1;
    map->count++;

    /* Update reverse map. */
    if (reverse_ensure(&map->reverse, internal_index) == 0) {
        map->reverse.ids[internal_index] = id_copy;
    }

    pthread_rwlock_unlock(&map->rwlock);
    return 0;
}

int gv_point_id_get(const GV_PointIDMap *map, const char *string_id, size_t *out_index)
{
    if (!map || !string_id || !out_index) {
        return -1;
    }

    /* Cast away const for the lock -- the rwlock is logically mutable. */
    pthread_rwlock_rdlock((pthread_rwlock_t *)&map->rwlock);

    uint64_t hash = fnv1a_hash(string_id);
    size_t   idx;
    int      found = find_bucket(map->buckets, map->capacity, string_id, hash, &idx);

    if (found) {
        *out_index = map->buckets[idx].internal_index;
        pthread_rwlock_unlock((pthread_rwlock_t *)&map->rwlock);
        return 0;
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&map->rwlock);
    return -1;
}

int gv_point_id_remove(GV_PointIDMap *map, const char *string_id)
{
    if (!map || !string_id) {
        return -1;
    }

    pthread_rwlock_wrlock(&map->rwlock);

    uint64_t hash = fnv1a_hash(string_id);
    size_t   idx;
    int      found = find_bucket(map->buckets, map->capacity, string_id, hash, &idx);

    if (!found) {
        pthread_rwlock_unlock(&map->rwlock);
        return -1;
    }

    /* Clear reverse mapping. */
    size_t rev_idx = map->buckets[idx].internal_index;
    if (rev_idx < map->reverse.capacity && map->reverse.ids[rev_idx] == map->buckets[idx].string_id) {
        map->reverse.ids[rev_idx] = NULL;
    }

    /* Free the owned string. */
    free(map->buckets[idx].string_id);

    /* Mark as empty.  To maintain linear-probing correctness we must
     * rehash subsequent entries in the same cluster. */
    map->buckets[idx].string_id = NULL;
    map->buckets[idx].occupied  = 0;
    map->count--;

    /* Rehash the cluster following the deleted slot. */
    size_t mask = map->capacity - 1;
    size_t probe = (idx + 1) & mask;
    while (map->buckets[probe].occupied) {
        GV_PointIDEntry tmp = map->buckets[probe];
        map->buckets[probe].string_id = NULL;
        map->buckets[probe].occupied  = 0;
        map->count--;

        /* Re-insert tmp. */
        size_t new_idx;
        find_bucket(map->buckets, map->capacity, tmp.string_id, tmp.hash, &new_idx);
        map->buckets[new_idx] = tmp;
        map->count++;

        /* Fix reverse pointer if needed. */
        if (tmp.internal_index < map->reverse.capacity) {
            map->reverse.ids[tmp.internal_index] = map->buckets[new_idx].string_id;
        }

        probe = (probe + 1) & mask;
    }

    pthread_rwlock_unlock(&map->rwlock);
    return 0;
}

int gv_point_id_has(const GV_PointIDMap *map, const char *string_id)
{
    if (!map || !string_id) {
        return -1;
    }

    pthread_rwlock_rdlock((pthread_rwlock_t *)&map->rwlock);

    uint64_t hash = fnv1a_hash(string_id);
    size_t   idx;
    int      found = find_bucket(map->buckets, map->capacity, string_id, hash, &idx);

    pthread_rwlock_unlock((pthread_rwlock_t *)&map->rwlock);
    return found ? 1 : 0;
}

/*  Public API: Reverse Lookup                                         */

const char *gv_point_id_reverse_lookup(const GV_PointIDMap *map, size_t internal_index)
{
    if (!map) {
        return NULL;
    }

    pthread_rwlock_rdlock((pthread_rwlock_t *)&map->rwlock);

    const char *result = NULL;
    if (internal_index < map->reverse.capacity) {
        result = map->reverse.ids[internal_index];
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&map->rwlock);
    return result;
}

/*  Public API: UUID v4 Generation                                     */

int gv_point_id_generate_uuid(char *buf, size_t buf_size)
{
    if (!buf || buf_size < GV_POINTID_UUID_BUF_MIN) {
        return -1;
    }

    uint8_t bytes[16];
    int have_random = 0;

    /* Attempt /dev/urandom first. */
    FILE *fp = fopen("/dev/urandom", "rb");
    if (fp) {
        if (fread(bytes, 1, 16, fp) == 16) {
            have_random = 1;
        }
        fclose(fp);
    }

    /* Fallback: time-based seeding with a simple LCG. */
    if (!have_random) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        uint64_t seed = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
        seed ^= (uint64_t)(uintptr_t)buf; /* Mix in an address for extra entropy. */

        for (int i = 0; i < 16; i++) {
            seed = seed * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
            bytes[i] = (uint8_t)(seed >> 33);
        }
    }

    /* RFC 4122 v4: set version (4) and variant (10xx). */
    bytes[6] = (bytes[6] & 0x0F) | 0x40;  /* Version 4. */
    bytes[8] = (bytes[8] & 0x3F) | 0x80;  /* Variant 10xx. */

    snprintf(buf, buf_size,
             "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             bytes[0],  bytes[1],  bytes[2],  bytes[3],
             bytes[4],  bytes[5],
             bytes[6],  bytes[7],
             bytes[8],  bytes[9],
             bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]);

    return 0;
}

/*  Public API: Iteration                                              */

size_t gv_point_id_count(const GV_PointIDMap *map)
{
    if (!map) {
        return 0;
    }

    pthread_rwlock_rdlock((pthread_rwlock_t *)&map->rwlock);
    size_t n = map->count;
    pthread_rwlock_unlock((pthread_rwlock_t *)&map->rwlock);

    return n;
}

int gv_point_id_iterate(const GV_PointIDMap *map,
                         int (*callback)(const char *id, size_t index, void *ctx),
                         void *ctx)
{
    if (!map || !callback) {
        return -1;
    }

    pthread_rwlock_rdlock((pthread_rwlock_t *)&map->rwlock);

    for (size_t i = 0; i < map->capacity; i++) {
        const GV_PointIDEntry *e = &map->buckets[i];
        if (!e->occupied) {
            continue;
        }

        int rc = callback(e->string_id, e->internal_index, ctx);
        if (rc != 0) {
            pthread_rwlock_unlock((pthread_rwlock_t *)&map->rwlock);
            return rc;
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&map->rwlock);
    return 0;
}

/*  Public API: Save / Load                                            */

int gv_point_id_save(const GV_PointIDMap *map, const char *filepath)
{
    if (!map || !filepath) {
        return -1;
    }

    pthread_rwlock_rdlock((pthread_rwlock_t *)&map->rwlock);

    FILE *fp = fopen(filepath, "wb");
    if (!fp) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&map->rwlock);
        return -1;
    }

    /* Write entry count. */
    size_t count = map->count;
    if (fwrite(&count, sizeof(size_t), 1, fp) != 1) {
        goto fail;
    }

    /* Write each live entry: string_len, string_id bytes, internal_index. */
    for (size_t i = 0; i < map->capacity; i++) {
        const GV_PointIDEntry *e = &map->buckets[i];
        if (!e->occupied) {
            continue;
        }

        size_t slen = strlen(e->string_id);
        if (fwrite(&slen, sizeof(size_t), 1, fp) != 1) {
            goto fail;
        }
        if (slen > 0 && fwrite(e->string_id, 1, slen, fp) != slen) {
            goto fail;
        }
        if (fwrite(&e->internal_index, sizeof(size_t), 1, fp) != 1) {
            goto fail;
        }
    }

    fclose(fp);
    pthread_rwlock_unlock((pthread_rwlock_t *)&map->rwlock);
    return 0;

fail:
    fclose(fp);
    pthread_rwlock_unlock((pthread_rwlock_t *)&map->rwlock);
    return -1;
}

GV_PointIDMap *gv_point_id_load(const char *filepath)
{
    if (!filepath) {
        return NULL;
    }

    FILE *fp = fopen(filepath, "rb");
    if (!fp) {
        return NULL;
    }

    size_t count = 0;
    if (fread(&count, sizeof(size_t), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }

    /* Pick a capacity that keeps the load factor healthy. */
    size_t needed = (size_t)((double)count / GV_POINTID_LOAD_FACTOR) + 1;
    GV_PointIDMap *map = gv_point_id_create(needed);
    if (!map) {
        fclose(fp);
        return NULL;
    }

    for (size_t i = 0; i < count; i++) {
        size_t slen = 0;
        if (fread(&slen, sizeof(size_t), 1, fp) != 1) {
            goto fail;
        }

        /* Sanity check: reject absurdly large strings. */
        if (slen > (size_t)100 * 1024 * 1024) {
            goto fail;
        }

        char *id_buf = (char *)malloc(slen + 1);
        if (!id_buf) {
            goto fail;
        }

        if (slen > 0 && fread(id_buf, 1, slen, fp) != slen) {
            free(id_buf);
            goto fail;
        }
        id_buf[slen] = '\0';

        size_t internal_index = 0;
        if (fread(&internal_index, sizeof(size_t), 1, fp) != 1) {
            free(id_buf);
            goto fail;
        }

        /* gv_point_id_set will strdup the key internally, so we can free ours. */
        if (gv_point_id_set(map, id_buf, internal_index) != 0) {
            free(id_buf);
            goto fail;
        }

        free(id_buf);
    }

    fclose(fp);
    return map;

fail:
    fclose(fp);
    gv_point_id_destroy(map);
    return NULL;
}
