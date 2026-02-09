/**
 * @file gv_group_search.c
 * @brief Search result grouping by metadata field.
 *
 * Oversamples candidates via gv_db_search(), buckets them into groups
 * keyed by a caller-specified metadata field, and returns the top groups
 * with the top hits per group, both sorted by ascending distance.
 */

#include "gigavector/gv_group_search.h"
#include "gigavector/gv_database.h"
#include "gigavector/gv_types.h"

#include <stdlib.h>
#include <string.h>
#include <float.h>

/* ============================================================================
 * Internal helpers
 * ============================================================================ */

/**
 * Walk a GV_Metadata linked list and return the value for @p key, or NULL.
 */
static const char *metadata_lookup(const GV_Metadata *meta, const char *key) {
    const GV_Metadata *cur = meta;
    while (cur) {
        if (cur->key && strcmp(cur->key, key) == 0) {
            return cur->value;
        }
        cur = cur->next;
    }
    return NULL;
}

static char *safe_strdup(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char *copy = (char *)malloc(len);
    if (copy) memcpy(copy, s, len);
    return copy;
}

/* ---- Lightweight open-addressed hash map (group_value -> bucket index) ---- */

typedef struct {
    size_t index;      /* candidate position in the oversampled result array */
    float  distance;
} GroupHitEntry;

typedef struct {
    char          *key;         /* owned copy of the group_value string */
    GroupHitEntry *hits;        /* dynamic array of hits in this bucket */
    size_t         hit_count;
    size_t         hit_capacity;
    float          best_distance; /* minimum distance in this bucket (for sorting) */
} GroupBucket;

typedef struct {
    GroupBucket *buckets;
    size_t       bucket_count;
    size_t       capacity;
} GroupMap;

static void group_map_init(GroupMap *map, size_t initial_capacity) {
    map->buckets = NULL;
    map->bucket_count = 0;
    map->capacity = 0;
    if (initial_capacity == 0) initial_capacity = 64;
    map->buckets = (GroupBucket *)calloc(initial_capacity, sizeof(GroupBucket));
    if (map->buckets) {
        map->capacity = initial_capacity;
    }
}

static void group_map_destroy(GroupMap *map) {
    if (!map) return;
    for (size_t i = 0; i < map->bucket_count; i++) {
        free(map->buckets[i].key);
        free(map->buckets[i].hits);
    }
    free(map->buckets);
    map->buckets = NULL;
    map->bucket_count = 0;
    map->capacity = 0;
}

/**
 * Find or create a bucket for @p key.  Returns pointer to the bucket, or
 * NULL on allocation failure.  Uses simple linear scan -- adequate for the
 * expected number of distinct groups (tens, not millions).
 */
static GroupBucket *group_map_get_or_create(GroupMap *map, const char *key) {
    /* Linear search for existing key */
    for (size_t i = 0; i < map->bucket_count; i++) {
        if (strcmp(map->buckets[i].key, key) == 0) {
            return &map->buckets[i];
        }
    }

    /* Need a new bucket -- grow array if necessary */
    if (map->bucket_count >= map->capacity) {
        size_t new_cap = map->capacity == 0 ? 64 : map->capacity * 2;
        GroupBucket *tmp = (GroupBucket *)realloc(map->buckets, new_cap * sizeof(GroupBucket));
        if (!tmp) return NULL;
        memset(tmp + map->capacity, 0, (new_cap - map->capacity) * sizeof(GroupBucket));
        map->buckets = tmp;
        map->capacity = new_cap;
    }

    GroupBucket *b = &map->buckets[map->bucket_count];
    memset(b, 0, sizeof(*b));
    b->key = safe_strdup(key);
    if (!b->key) return NULL;
    b->best_distance = FLT_MAX;
    map->bucket_count++;
    return b;
}

static int bucket_add_hit(GroupBucket *b, size_t index, float distance) {
    if (b->hit_count >= b->hit_capacity) {
        size_t new_cap = b->hit_capacity == 0 ? 16 : b->hit_capacity * 2;
        GroupHitEntry *tmp = (GroupHitEntry *)realloc(b->hits, new_cap * sizeof(GroupHitEntry));
        if (!tmp) return -1;
        b->hits = tmp;
        b->hit_capacity = new_cap;
    }
    b->hits[b->hit_count].index = index;
    b->hits[b->hit_count].distance = distance;
    b->hit_count++;

    if (distance < b->best_distance) {
        b->best_distance = distance;
    }
    return 0;
}

/* ---- qsort comparators -------------------------------------------------- */

static int compare_hits_by_distance(const void *a, const void *b) {
    const GroupHitEntry *ha = (const GroupHitEntry *)a;
    const GroupHitEntry *hb = (const GroupHitEntry *)b;
    if (ha->distance < hb->distance) return -1;
    if (ha->distance > hb->distance) return  1;
    return 0;
}

static int compare_buckets_by_best_distance(const void *a, const void *b) {
    const GroupBucket *ba = (const GroupBucket *)a;
    const GroupBucket *bb = (const GroupBucket *)b;
    if (ba->best_distance < bb->best_distance) return -1;
    if (ba->best_distance > bb->best_distance) return  1;
    return 0;
}

/* ---- Free a GV_SearchResult's owned vector allocation -------------------- */

static void free_search_result_vector(GV_SearchResult *r) {
    if (!r || !r->vector) return;
    GV_Vector *v = (GV_Vector *)r->vector;
    /* Free the copied metadata chain */
    GV_Metadata *m = v->metadata;
    while (m) {
        GV_Metadata *next = m->next;
        free(m->key);
        free(m->value);
        free(m);
        m = next;
    }
    free(v->data);
    free(v);
    r->vector = NULL;
}

/* ============================================================================
 * Public API
 * ============================================================================ */

void gv_group_search_config_init(GV_GroupSearchConfig *config) {
    if (!config) return;
    memset(config, 0, sizeof(*config));
    config->group_by      = NULL;
    config->group_limit   = 10;
    config->hits_per_group = 3;
    config->distance_type = 0;   /* GV_DISTANCE_EUCLIDEAN */
    config->oversample    = 0;   /* 0 means auto: group_limit * hits_per_group * 4 */
}

int gv_group_search(const GV_Database *db, const float *query, size_t dimension,
                     const GV_GroupSearchConfig *config, GV_GroupedResult *result) {
    if (!db || !query || !config || !result || !config->group_by) {
        return -1;
    }
    if (dimension == 0) {
        return -1;
    }

    /* Zero the output so callers can safely call free_result on error paths. */
    memset(result, 0, sizeof(*result));

    /* Resolve effective parameters */
    size_t group_limit    = config->group_limit   > 0 ? config->group_limit   : 10;
    size_t hits_per_group = config->hits_per_group > 0 ? config->hits_per_group : 3;
    size_t oversample     = config->oversample;
    if (oversample == 0) {
        oversample = group_limit * hits_per_group * 4;
    }
    if (oversample == 0) {
        oversample = 40; /* absolute fallback */
    }

    /* Step 1: Oversample -- fetch many candidates via gv_db_search() */
    GV_SearchResult *candidates = (GV_SearchResult *)calloc(oversample, sizeof(GV_SearchResult));
    if (!candidates) {
        return -1;
    }

    int found = gv_db_search(db, query, oversample, candidates,
                              (GV_DistanceType)config->distance_type);
    if (found < 0) {
        free(candidates);
        return -1;
    }
    size_t n_candidates = (size_t)found;
    result->total_hits = n_candidates;

    if (n_candidates == 0) {
        free(candidates);
        return 0;
    }

    /* Step 2: Bucket candidates by the metadata field specified in group_by */
    GroupMap map;
    group_map_init(&map, group_limit * 2 > 64 ? group_limit * 2 : 64);

    for (size_t i = 0; i < n_candidates; i++) {
        const GV_Vector *vec = candidates[i].vector;
        if (!vec) continue;

        const char *val = metadata_lookup(vec->metadata, config->group_by);
        if (!val) continue;  /* skip candidates without the grouping field */

        GroupBucket *b = group_map_get_or_create(&map, val);
        if (!b) continue;

        bucket_add_hit(b, i, candidates[i].distance);
    }

    if (map.bucket_count == 0) {
        /* No candidates had the requested metadata field */
        for (size_t i = 0; i < n_candidates; i++) {
            free_search_result_vector(&candidates[i]);
        }
        free(candidates);
        group_map_destroy(&map);
        return 0;
    }

    /* Step 3: Sort each bucket's hits by distance, then sort buckets by best distance */
    for (size_t i = 0; i < map.bucket_count; i++) {
        if (map.buckets[i].hit_count > 1) {
            qsort(map.buckets[i].hits, map.buckets[i].hit_count,
                  sizeof(GroupHitEntry), compare_hits_by_distance);
        }
    }

    qsort(map.buckets, map.bucket_count, sizeof(GroupBucket),
          compare_buckets_by_best_distance);

    /* Step 4: Build the output -- top group_limit groups, top hits_per_group each */
    size_t out_groups = map.bucket_count < group_limit ? map.bucket_count : group_limit;

    result->groups = (GV_SearchGroup *)calloc(out_groups, sizeof(GV_SearchGroup));
    if (!result->groups) {
        for (size_t i = 0; i < n_candidates; i++) {
            free_search_result_vector(&candidates[i]);
        }
        free(candidates);
        group_map_destroy(&map);
        return -1;
    }
    result->group_count = out_groups;

    for (size_t g = 0; g < out_groups; g++) {
        GroupBucket *b = &map.buckets[g];

        result->groups[g].group_value = safe_strdup(b->key);
        if (!result->groups[g].group_value) {
            /* Partial failure -- clean up what we built so far and bail */
            gv_group_search_free_result(result);
            for (size_t i = 0; i < n_candidates; i++) {
                free_search_result_vector(&candidates[i]);
            }
            free(candidates);
            group_map_destroy(&map);
            return -1;
        }

        size_t n_hits = b->hit_count < hits_per_group ? b->hit_count : hits_per_group;
        result->groups[g].hits = (GV_GroupHit *)malloc(n_hits * sizeof(GV_GroupHit));
        if (!result->groups[g].hits) {
            gv_group_search_free_result(result);
            for (size_t i = 0; i < n_candidates; i++) {
                free_search_result_vector(&candidates[i]);
            }
            free(candidates);
            group_map_destroy(&map);
            return -1;
        }
        result->groups[g].hit_count = n_hits;

        for (size_t h = 0; h < n_hits; h++) {
            result->groups[g].hits[h].index    = b->hits[h].index;
            result->groups[g].hits[h].distance = b->hits[h].distance;
        }
    }

    /* Step 5: Cleanup temporary data */
    for (size_t i = 0; i < n_candidates; i++) {
        free_search_result_vector(&candidates[i]);
    }
    free(candidates);
    group_map_destroy(&map);

    return 0;
}

void gv_group_search_free_result(GV_GroupedResult *result) {
    if (!result) return;
    if (result->groups) {
        for (size_t i = 0; i < result->group_count; i++) {
            free(result->groups[i].group_value);
            free(result->groups[i].hits);
        }
        free(result->groups);
    }
    memset(result, 0, sizeof(*result));
}
