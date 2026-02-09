#define _POSIX_C_SOURCE 200809L

/**
 * @file gv_named_vectors.c
 * @brief Named vector fields implementation.
 *
 * Allows multiple named vector fields per document (e.g., "title", "content",
 * "image"), each with potentially different dimensions and distance metrics.
 */

#include "gigavector/gv_named_vectors.h"
#include "gigavector/gv_distance.h"
#include "gigavector/gv_types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <pthread.h>

/* ================================================================== */
/*  Internal constants                                                 */
/* ================================================================== */

#define GV_NV_MAX_FIELDS         32
#define GV_NV_FIELD_HASH_BUCKETS 64
#define GV_NV_INITIAL_POINT_CAP  64
#define GV_NV_MAGIC              0x47564E56U  /* "GVNV" */
#define GV_NV_VERSION            1U

/* ================================================================== */
/*  Internal structures                                                */
/* ================================================================== */

/**
 * @brief Per-field storage for a single named vector field.
 *
 * Vectors for all points are stored in a flat contiguous array
 * (point_count * dimension floats).  A parallel bitmap tracks which
 * point slots are occupied.
 */
typedef struct GV_NVField {
    char            *name;          /**< Heap-allocated field name. */
    size_t           dimension;     /**< Dimensionality of vectors in this field. */
    int              distance_type; /**< Default distance metric (GV_DistanceType). */

    float           *vectors;       /**< Contiguous float array [capacity * dimension]. */
    uint8_t         *occupied;      /**< 1 if slot has data, 0 otherwise. */
    size_t           capacity;      /**< Number of point slots allocated. */

    struct GV_NVField *hash_next;   /**< Next entry in hash bucket chain. */
} GV_NVField;

/**
 * @brief The named vector store.
 */
struct GV_NamedVectorStore {
    GV_NVField     *buckets[GV_NV_FIELD_HASH_BUCKETS]; /**< Hash table of fields. */
    size_t          field_count;                         /**< Number of registered fields. */

    size_t          point_count;   /**< High-water mark of point IDs inserted. */
    size_t          point_capacity;/**< Current capacity across all fields. */

    uint8_t        *point_alive;   /**< Global alive bitmap (1 = not deleted). */

    pthread_rwlock_t rwlock;
};

/* ================================================================== */
/*  Max-heap helpers for top-k selection                               */
/* ================================================================== */

typedef struct {
    float   dist;
    size_t  idx;
} GV_NVHeapItem;

static void gv_nv_heap_sift_down(GV_NVHeapItem *heap, size_t size, size_t i) {
    while (1) {
        size_t l = 2 * i + 1;
        size_t r = l + 1;
        size_t largest = i;
        if (l < size && heap[l].dist > heap[largest].dist) largest = l;
        if (r < size && heap[r].dist > heap[largest].dist) largest = r;
        if (largest == i) break;
        GV_NVHeapItem tmp = heap[i];
        heap[i] = heap[largest];
        heap[largest] = tmp;
        i = largest;
    }
}

static void gv_nv_heap_push(GV_NVHeapItem *heap, size_t *size, size_t capacity,
                             float dist, size_t idx) {
    if (*size < capacity) {
        heap[*size].dist = dist;
        heap[*size].idx  = idx;
        (*size)++;
        /* Sift up */
        size_t i = *size - 1;
        while (i > 0) {
            size_t parent = (i - 1) / 2;
            if (heap[i].dist > heap[parent].dist) {
                GV_NVHeapItem tmp = heap[i];
                heap[i] = heap[parent];
                heap[parent] = tmp;
                i = parent;
            } else {
                break;
            }
        }
    } else if (dist < heap[0].dist) {
        heap[0].dist = dist;
        heap[0].idx  = idx;
        gv_nv_heap_sift_down(heap, *size, 0);
    }
}

/* ================================================================== */
/*  Hash helpers                                                       */
/* ================================================================== */

static size_t gv_nv_hash_name(const char *name) {
    size_t h = 5381;
    int c;
    while ((c = (unsigned char)*name++)) {
        h = ((h << 5) + h) + (size_t)c;
    }
    return h % GV_NV_FIELD_HASH_BUCKETS;
}

static GV_NVField *gv_nv_find_field(const GV_NamedVectorStore *store, const char *name) {
    size_t bucket = gv_nv_hash_name(name);
    GV_NVField *f = store->buckets[bucket];
    while (f) {
        if (strcmp(f->name, name) == 0) return f;
        f = f->hash_next;
    }
    return NULL;
}

/* ================================================================== */
/*  I/O helpers                                                        */
/* ================================================================== */

static int gv_nv_write_u32(FILE *fp, uint32_t v) {
    return fwrite(&v, sizeof(uint32_t), 1, fp) == 1 ? 0 : -1;
}

static int gv_nv_read_u32(FILE *fp, uint32_t *v) {
    return (v && fread(v, sizeof(uint32_t), 1, fp) == 1) ? 0 : -1;
}

static int gv_nv_write_u64(FILE *fp, uint64_t v) {
    return fwrite(&v, sizeof(uint64_t), 1, fp) == 1 ? 0 : -1;
}

static int gv_nv_read_u64(FILE *fp, uint64_t *v) {
    return (v && fread(v, sizeof(uint64_t), 1, fp) == 1) ? 0 : -1;
}

static int gv_nv_write_str(FILE *fp, const char *s) {
    uint32_t len = (uint32_t)strlen(s);
    if (gv_nv_write_u32(fp, len) != 0) return -1;
    if (len > 0 && fwrite(s, 1, len, fp) != len) return -1;
    return 0;
}

static int gv_nv_read_str(FILE *fp, char **out) {
    uint32_t len = 0;
    if (gv_nv_read_u32(fp, &len) != 0) return -1;
    char *buf = (char *)malloc((size_t)len + 1);
    if (!buf) return -1;
    if (len > 0 && fread(buf, 1, len, fp) != len) {
        free(buf);
        return -1;
    }
    buf[len] = '\0';
    *out = buf;
    return 0;
}

/* ================================================================== */
/*  Capacity management                                                */
/* ================================================================== */

/**
 * @brief Ensure all per-field arrays and the global alive bitmap can hold
 *        at least @p required point slots.
 */
static int gv_nv_ensure_capacity(GV_NamedVectorStore *store, size_t required) {
    if (required <= store->point_capacity) return 0;

    size_t new_cap = store->point_capacity;
    if (new_cap == 0) new_cap = GV_NV_INITIAL_POINT_CAP;
    while (new_cap < required) {
        new_cap *= 2;
    }

    /* Grow the global alive bitmap */
    uint8_t *new_alive = (uint8_t *)realloc(store->point_alive, new_cap * sizeof(uint8_t));
    if (!new_alive) return -1;
    memset(new_alive + store->point_capacity, 0, (new_cap - store->point_capacity) * sizeof(uint8_t));
    store->point_alive = new_alive;

    /* Grow each field's per-point arrays */
    for (size_t b = 0; b < GV_NV_FIELD_HASH_BUCKETS; b++) {
        GV_NVField *f = store->buckets[b];
        while (f) {
            float *new_vecs = (float *)realloc(f->vectors,
                                               new_cap * f->dimension * sizeof(float));
            if (!new_vecs) return -1;
            /* Zero newly-allocated region */
            memset(new_vecs + f->capacity * f->dimension, 0,
                   (new_cap - f->capacity) * f->dimension * sizeof(float));
            f->vectors = new_vecs;

            uint8_t *new_occ = (uint8_t *)realloc(f->occupied, new_cap * sizeof(uint8_t));
            if (!new_occ) return -1;
            memset(new_occ + f->capacity, 0, (new_cap - f->capacity) * sizeof(uint8_t));
            f->occupied = new_occ;

            f->capacity = new_cap;
            f = f->hash_next;
        }
    }

    store->point_capacity = new_cap;
    return 0;
}

/* ================================================================== */
/*  Distance computation (raw float arrays)                            */
/* ================================================================== */

/**
 * @brief Compute distance between two raw float vectors using the
 *        GV_DistanceType enum.  Wraps gv_distance() by constructing
 *        temporary GV_Vector structs.
 */
static float gv_nv_compute_distance(const float *a, const float *b,
                                     size_t dimension, int distance_type) {
    GV_Vector va, vb;
    va.dimension = dimension;
    va.data      = (float *)a;  /* const-cast safe: gv_distance only reads */
    va.metadata  = NULL;
    vb.dimension = dimension;
    vb.data      = (float *)b;
    vb.metadata  = NULL;
    return gv_distance(&va, &vb, (GV_DistanceType)distance_type);
}

/* ================================================================== */
/*  Lifecycle                                                          */
/* ================================================================== */

GV_NamedVectorStore *gv_named_vectors_create(void) {
    GV_NamedVectorStore *store = (GV_NamedVectorStore *)calloc(1, sizeof(GV_NamedVectorStore));
    if (!store) return NULL;

    if (pthread_rwlock_init(&store->rwlock, NULL) != 0) {
        free(store);
        return NULL;
    }

    return store;
}

void gv_named_vectors_destroy(GV_NamedVectorStore *store) {
    if (!store) return;

    /* Free all fields */
    for (size_t b = 0; b < GV_NV_FIELD_HASH_BUCKETS; b++) {
        GV_NVField *f = store->buckets[b];
        while (f) {
            GV_NVField *next = f->hash_next;
            free(f->name);
            free(f->vectors);
            free(f->occupied);
            free(f);
            f = next;
        }
    }

    free(store->point_alive);
    pthread_rwlock_destroy(&store->rwlock);
    free(store);
}

/* ================================================================== */
/*  Field management                                                   */
/* ================================================================== */

int gv_named_vectors_add_field(GV_NamedVectorStore *store, const GV_VectorFieldConfig *config) {
    if (!store || !config || !config->name || config->dimension == 0) return -1;

    pthread_rwlock_wrlock(&store->rwlock);

    /* Check max fields */
    if (store->field_count >= GV_NV_MAX_FIELDS) {
        pthread_rwlock_unlock(&store->rwlock);
        return -1;
    }

    /* Check for duplicate name */
    if (gv_nv_find_field(store, config->name) != NULL) {
        pthread_rwlock_unlock(&store->rwlock);
        return -1;
    }

    /* Allocate field */
    GV_NVField *f = (GV_NVField *)calloc(1, sizeof(GV_NVField));
    if (!f) {
        pthread_rwlock_unlock(&store->rwlock);
        return -1;
    }

    f->name = strdup(config->name);
    if (!f->name) {
        free(f);
        pthread_rwlock_unlock(&store->rwlock);
        return -1;
    }

    f->dimension     = config->dimension;
    f->distance_type = config->distance_type;

    /* Pre-allocate to match current store capacity */
    if (store->point_capacity > 0) {
        f->vectors = (float *)calloc(store->point_capacity * f->dimension, sizeof(float));
        f->occupied = (uint8_t *)calloc(store->point_capacity, sizeof(uint8_t));
        if (!f->vectors || !f->occupied) {
            free(f->vectors);
            free(f->occupied);
            free(f->name);
            free(f);
            pthread_rwlock_unlock(&store->rwlock);
            return -1;
        }
        f->capacity = store->point_capacity;
    }

    /* Insert into hash table */
    size_t bucket = gv_nv_hash_name(config->name);
    f->hash_next = store->buckets[bucket];
    store->buckets[bucket] = f;
    store->field_count++;

    pthread_rwlock_unlock(&store->rwlock);
    return 0;
}

int gv_named_vectors_remove_field(GV_NamedVectorStore *store, const char *name) {
    if (!store || !name) return -1;

    pthread_rwlock_wrlock(&store->rwlock);

    size_t bucket = gv_nv_hash_name(name);
    GV_NVField *prev = NULL;
    GV_NVField *f = store->buckets[bucket];

    while (f) {
        if (strcmp(f->name, name) == 0) {
            if (prev) {
                prev->hash_next = f->hash_next;
            } else {
                store->buckets[bucket] = f->hash_next;
            }
            free(f->name);
            free(f->vectors);
            free(f->occupied);
            free(f);
            store->field_count--;
            pthread_rwlock_unlock(&store->rwlock);
            return 0;
        }
        prev = f;
        f = f->hash_next;
    }

    pthread_rwlock_unlock(&store->rwlock);
    return -1; /* not found */
}

size_t gv_named_vectors_field_count(const GV_NamedVectorStore *store) {
    if (!store) return 0;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&store->rwlock);
    size_t count = store->field_count;
    pthread_rwlock_unlock((pthread_rwlock_t *)&store->rwlock);

    return count;
}

int gv_named_vectors_get_field(const GV_NamedVectorStore *store, const char *name,
                                GV_VectorFieldConfig *out) {
    if (!store || !name || !out) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&store->rwlock);

    const GV_NVField *f = gv_nv_find_field(store, name);
    if (!f) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&store->rwlock);
        return -1;
    }

    out->name          = f->name;
    out->dimension     = f->dimension;
    out->distance_type = f->distance_type;

    pthread_rwlock_unlock((pthread_rwlock_t *)&store->rwlock);
    return 0;
}

/* ================================================================== */
/*  Point operations                                                   */
/* ================================================================== */

int gv_named_vectors_insert(GV_NamedVectorStore *store, size_t point_id,
                             const GV_NamedVector *vectors, size_t vector_count) {
    if (!store || !vectors || vector_count == 0) return -1;

    pthread_rwlock_wrlock(&store->rwlock);

    /* Ensure capacity for this point_id */
    if (gv_nv_ensure_capacity(store, point_id + 1) != 0) {
        pthread_rwlock_unlock(&store->rwlock);
        return -1;
    }

    /* Reject if point already alive */
    if (point_id < store->point_count && store->point_alive[point_id]) {
        pthread_rwlock_unlock(&store->rwlock);
        return -1;
    }

    /* Validate all fields exist and dimensions match before writing */
    for (size_t i = 0; i < vector_count; i++) {
        if (!vectors[i].field_name || !vectors[i].data) {
            pthread_rwlock_unlock(&store->rwlock);
            return -1;
        }
        GV_NVField *f = gv_nv_find_field(store, vectors[i].field_name);
        if (!f) {
            pthread_rwlock_unlock(&store->rwlock);
            return -1;
        }
        if (vectors[i].dimension != f->dimension) {
            pthread_rwlock_unlock(&store->rwlock);
            return -1;
        }
    }

    /* Copy vector data into per-field storage */
    for (size_t i = 0; i < vector_count; i++) {
        GV_NVField *f = gv_nv_find_field(store, vectors[i].field_name);
        memcpy(f->vectors + point_id * f->dimension,
               vectors[i].data,
               f->dimension * sizeof(float));
        f->occupied[point_id] = 1;
    }

    store->point_alive[point_id] = 1;
    if (point_id >= store->point_count) {
        store->point_count = point_id + 1;
    }

    pthread_rwlock_unlock(&store->rwlock);
    return 0;
}

int gv_named_vectors_update(GV_NamedVectorStore *store, size_t point_id,
                             const GV_NamedVector *vectors, size_t vector_count) {
    if (!store || !vectors || vector_count == 0) return -1;

    pthread_rwlock_wrlock(&store->rwlock);

    /* Point must exist and be alive */
    if (point_id >= store->point_count || !store->point_alive[point_id]) {
        pthread_rwlock_unlock(&store->rwlock);
        return -1;
    }

    /* Validate all fields and dimensions */
    for (size_t i = 0; i < vector_count; i++) {
        if (!vectors[i].field_name || !vectors[i].data) {
            pthread_rwlock_unlock(&store->rwlock);
            return -1;
        }
        GV_NVField *f = gv_nv_find_field(store, vectors[i].field_name);
        if (!f) {
            pthread_rwlock_unlock(&store->rwlock);
            return -1;
        }
        if (vectors[i].dimension != f->dimension) {
            pthread_rwlock_unlock(&store->rwlock);
            return -1;
        }
    }

    /* Overwrite vector data */
    for (size_t i = 0; i < vector_count; i++) {
        GV_NVField *f = gv_nv_find_field(store, vectors[i].field_name);
        memcpy(f->vectors + point_id * f->dimension,
               vectors[i].data,
               f->dimension * sizeof(float));
        f->occupied[point_id] = 1;
    }

    pthread_rwlock_unlock(&store->rwlock);
    return 0;
}

int gv_named_vectors_delete(GV_NamedVectorStore *store, size_t point_id) {
    if (!store) return -1;

    pthread_rwlock_wrlock(&store->rwlock);

    if (point_id >= store->point_count || !store->point_alive[point_id]) {
        pthread_rwlock_unlock(&store->rwlock);
        return -1;
    }

    store->point_alive[point_id] = 0;

    /* Clear occupied flag in each field for this point */
    for (size_t b = 0; b < GV_NV_FIELD_HASH_BUCKETS; b++) {
        GV_NVField *f = store->buckets[b];
        while (f) {
            if (point_id < f->capacity) {
                f->occupied[point_id] = 0;
            }
            f = f->hash_next;
        }
    }

    pthread_rwlock_unlock(&store->rwlock);
    return 0;
}

/* ================================================================== */
/*  Search                                                             */
/* ================================================================== */

int gv_named_vectors_search(const GV_NamedVectorStore *store, const char *field_name,
                             const float *query, size_t k, GV_NamedSearchResult *results) {
    if (!store || !field_name || !query || !results || k == 0) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&store->rwlock);

    const GV_NVField *field = gv_nv_find_field(store, field_name);
    if (!field) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&store->rwlock);
        return -1;
    }

    GV_NVHeapItem *heap = (GV_NVHeapItem *)malloc(k * sizeof(GV_NVHeapItem));
    if (!heap) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&store->rwlock);
        return -1;
    }
    size_t heap_size = 0;

    /* Brute-force scan over all alive points that have data in this field */
    for (size_t i = 0; i < store->point_count; i++) {
        if (!store->point_alive[i]) continue;
        if (i >= field->capacity || !field->occupied[i]) continue;

        const float *vec = field->vectors + i * field->dimension;
        float dist = gv_nv_compute_distance(query, vec, field->dimension,
                                             field->distance_type);

        gv_nv_heap_push(heap, &heap_size, k, dist, i);
    }

    /* Extract results from max-heap in ascending distance order */
    int n = (int)heap_size;
    for (int i = n - 1; i >= 0; i--) {
        results[i].point_index = heap[0].idx;
        results[i].distance    = heap[0].dist;
        results[i].field_name  = field->name;

        /* Pop top of heap */
        heap[0] = heap[heap_size - 1];
        heap_size--;
        if (heap_size > 0) {
            gv_nv_heap_sift_down(heap, heap_size, 0);
        }
    }

    free(heap);
    pthread_rwlock_unlock((pthread_rwlock_t *)&store->rwlock);
    return n;
}

/* ================================================================== */
/*  Accessors                                                          */
/* ================================================================== */

const float *gv_named_vectors_get(const GV_NamedVectorStore *store, size_t point_id,
                                   const char *field_name) {
    if (!store || !field_name) return NULL;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&store->rwlock);

    if (point_id >= store->point_count || !store->point_alive[point_id]) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&store->rwlock);
        return NULL;
    }

    const GV_NVField *f = gv_nv_find_field(store, field_name);
    if (!f || point_id >= f->capacity || !f->occupied[point_id]) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&store->rwlock);
        return NULL;
    }

    const float *ptr = f->vectors + point_id * f->dimension;

    pthread_rwlock_unlock((pthread_rwlock_t *)&store->rwlock);
    return ptr;
}

size_t gv_named_vectors_count(const GV_NamedVectorStore *store) {
    if (!store) return 0;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&store->rwlock);

    size_t count = 0;
    for (size_t i = 0; i < store->point_count; i++) {
        if (store->point_alive[i]) count++;
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&store->rwlock);
    return count;
}

/* ================================================================== */
/*  Save / Load                                                        */
/* ================================================================== */

int gv_named_vectors_save(const GV_NamedVectorStore *store, const char *filepath) {
    if (!store || !filepath) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&store->rwlock);

    FILE *fp = fopen(filepath, "wb");
    if (!fp) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&store->rwlock);
        return -1;
    }

    /* Header: magic, version */
    if (gv_nv_write_u32(fp, GV_NV_MAGIC) != 0) goto fail;
    if (gv_nv_write_u32(fp, GV_NV_VERSION) != 0) goto fail;

    /* Field count */
    if (gv_nv_write_u32(fp, (uint32_t)store->field_count) != 0) goto fail;

    /* Point count (high-water mark) */
    if (gv_nv_write_u64(fp, (uint64_t)store->point_count) != 0) goto fail;

    /* Write alive bitmap */
    if (store->point_count > 0) {
        if (fwrite(store->point_alive, sizeof(uint8_t), store->point_count, fp)
            != store->point_count) goto fail;
    }

    /* Write each field: config, then per-point vector data */
    for (size_t b = 0; b < GV_NV_FIELD_HASH_BUCKETS; b++) {
        const GV_NVField *f = store->buckets[b];
        while (f) {
            /* Field config */
            if (gv_nv_write_str(fp, f->name) != 0) goto fail;
            if (gv_nv_write_u64(fp, (uint64_t)f->dimension) != 0) goto fail;
            if (gv_nv_write_u32(fp, (uint32_t)f->distance_type) != 0) goto fail;

            /* Occupied bitmap for this field (point_count entries) */
            if (store->point_count > 0) {
                if (fwrite(f->occupied, sizeof(uint8_t), store->point_count, fp)
                    != store->point_count) goto fail;
            }

            /* Write only occupied vectors */
            for (size_t p = 0; p < store->point_count; p++) {
                if (!f->occupied[p]) continue;
                if (fwrite(f->vectors + p * f->dimension, sizeof(float),
                           f->dimension, fp) != f->dimension) goto fail;
            }

            f = f->hash_next;
        }
    }

    fclose(fp);
    pthread_rwlock_unlock((pthread_rwlock_t *)&store->rwlock);
    return 0;

fail:
    fclose(fp);
    pthread_rwlock_unlock((pthread_rwlock_t *)&store->rwlock);
    return -1;
}

GV_NamedVectorStore *gv_named_vectors_load(const char *filepath) {
    if (!filepath) return NULL;

    FILE *fp = fopen(filepath, "rb");
    if (!fp) return NULL;

    /* Read and validate header */
    uint32_t magic = 0, version = 0;
    if (gv_nv_read_u32(fp, &magic) != 0 || magic != GV_NV_MAGIC) goto fail;
    if (gv_nv_read_u32(fp, &version) != 0 || version != GV_NV_VERSION) goto fail;

    uint32_t field_count = 0;
    if (gv_nv_read_u32(fp, &field_count) != 0) goto fail;
    if (field_count > GV_NV_MAX_FIELDS) goto fail;

    uint64_t point_count_u64 = 0;
    if (gv_nv_read_u64(fp, &point_count_u64) != 0) goto fail;
    size_t point_count = (size_t)point_count_u64;

    GV_NamedVectorStore *store = gv_named_vectors_create();
    if (!store) goto fail;

    /* Allocate capacity for points */
    if (point_count > 0) {
        if (gv_nv_ensure_capacity(store, point_count) != 0) {
            gv_named_vectors_destroy(store);
            goto fail;
        }
        store->point_count = point_count;

        /* Read alive bitmap */
        if (fread(store->point_alive, sizeof(uint8_t), point_count, fp) != point_count) {
            gv_named_vectors_destroy(store);
            goto fail;
        }
    }

    /* Read each field */
    for (uint32_t fi = 0; fi < field_count; fi++) {
        char *name = NULL;
        uint64_t dimension_u64 = 0;
        uint32_t dist_type = 0;

        if (gv_nv_read_str(fp, &name) != 0) {
            gv_named_vectors_destroy(store);
            goto fail;
        }
        if (gv_nv_read_u64(fp, &dimension_u64) != 0) {
            free(name);
            gv_named_vectors_destroy(store);
            goto fail;
        }
        if (gv_nv_read_u32(fp, &dist_type) != 0) {
            free(name);
            gv_named_vectors_destroy(store);
            goto fail;
        }

        /* Register the field */
        GV_VectorFieldConfig cfg;
        cfg.name          = name;
        cfg.dimension     = (size_t)dimension_u64;
        cfg.distance_type = (int)dist_type;

        if (gv_named_vectors_add_field(store, &cfg) != 0) {
            free(name);
            gv_named_vectors_destroy(store);
            goto fail;
        }

        GV_NVField *f = gv_nv_find_field(store, name);
        free(name);

        if (!f) {
            gv_named_vectors_destroy(store);
            goto fail;
        }

        /* Read occupied bitmap */
        if (point_count > 0) {
            if (fread(f->occupied, sizeof(uint8_t), point_count, fp) != point_count) {
                gv_named_vectors_destroy(store);
                goto fail;
            }

            /* Read occupied vectors */
            for (size_t p = 0; p < point_count; p++) {
                if (!f->occupied[p]) continue;
                if (fread(f->vectors + p * f->dimension, sizeof(float),
                          f->dimension, fp) != f->dimension) {
                    gv_named_vectors_destroy(store);
                    goto fail;
                }
            }
        }
    }

    fclose(fp);
    return store;

fail:
    fclose(fp);
    return NULL;
}
