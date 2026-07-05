/**
 * @file group_search.c
 * @brief Search result grouping by metadata field.
 *
 * Oversamples candidates via db_search(), buckets them into groups
 * keyed by a caller-specified metadata field, and returns the top groups
 * with the top hits per group, both sorted by ascending distance.
 */

#include "search/group_search.h"
#include "core/memory.h"
#include "core/arena.h"
#include "core/scope.h"
#include "storage/database.h"
#include "core/types.h"
#include "core/utils.h"

#include <stdlib.h>
#include <string.h>
#include <float.h>

#define GV_GROUP_SEARCH_ARENA_BYTES (256u * 1024u)

typedef struct {
    size_t index;
    float  distance;
} GroupHitEntry;

typedef struct {
    char          *key;
    GroupHitEntry *hits;
    size_t         hit_count;
    size_t         hit_capacity;
    float          best_distance;
} GroupBucket;

typedef struct {
    GV_Arena     *arena;
    GroupBucket  *buckets;
    size_t        bucket_count;
    size_t        capacity;
} GroupMap;

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

static void group_map_init(GroupMap *map, GV_Arena *arena, size_t initial_capacity) {
    map->arena = arena;
    map->buckets = NULL;
    map->bucket_count = 0;
    map->capacity = 0;
    if (initial_capacity == 0) {
        initial_capacity = 64;
    }
    map->buckets = (GroupBucket *)gv_arena_calloc(arena, initial_capacity, sizeof(GroupBucket));
    if (map->buckets) {
        map->capacity = initial_capacity;
    }
}

static GroupBucket *group_map_grow(GroupMap *map) {
    size_t new_cap = map->capacity == 0 ? 64 : map->capacity * 2;
    GroupBucket *tmp =
        (GroupBucket *)gv_arena_calloc(map->arena, new_cap, sizeof(GroupBucket));
    if (!tmp) {
        return NULL;
    }
    if (map->bucket_count > 0) {
        memcpy(tmp, map->buckets, map->bucket_count * sizeof(GroupBucket));
    }
    map->buckets = tmp;
    map->capacity = new_cap;
    return &map->buckets[map->bucket_count];
}

static GroupBucket *group_map_get_or_create(GroupMap *map, const char *key) {
    for (size_t i = 0; i < map->bucket_count; i++) {
        if (strcmp(map->buckets[i].key, key) == 0) {
            return &map->buckets[i];
        }
    }

    if (map->bucket_count >= map->capacity) {
        if (group_map_grow(map) == NULL) {
            return NULL;
        }
    }

    GroupBucket *b = &map->buckets[map->bucket_count];
    memset(b, 0, sizeof(*b));
    b->key = gv_arena_strdup(map->arena, key);
    if (!b->key) {
        return NULL;
    }
    b->best_distance = FLT_MAX;
    map->bucket_count++;
    return b;
}

static int bucket_add_hit(GroupMap *map, GroupBucket *b, size_t index, float distance) {
    if (b->hit_count >= b->hit_capacity) {
        size_t new_cap = b->hit_capacity == 0 ? 16 : b->hit_capacity * 2;
        GroupHitEntry *tmp =
            (GroupHitEntry *)gv_arena_calloc(map->arena, new_cap, sizeof(GroupHitEntry));
        if (!tmp) {
            return -1;
        }
        if (b->hit_count > 0) {
            memcpy(tmp, b->hits, b->hit_count * sizeof(GroupHitEntry));
        }
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

static void free_search_result_vector(GV_SearchResult *r) {
    if (!r || !r->vector) return;
    GV_Vector *v = (GV_Vector *)r->vector;
    GV_Metadata *m = v->metadata;
    while (m) {
        GV_Metadata *next = m->next;
        gv_free(m->key);
        gv_free(m->value);
        gv_free(m);
        m = next;
    }
    gv_free(v->data);
    gv_free(v);
    r->vector = NULL;
}

static void group_search_release_candidates(GV_SearchResult *candidates,
                                            size_t n_candidates,
                                            int candidates_on_heap) {
    for (size_t i = 0; i < n_candidates; i++) {
        free_search_result_vector(&candidates[i]);
    }
    gv_tls_free_or_heap(candidates, candidates_on_heap);
}

void group_search_config_init(GV_GroupSearchConfig *config) {
    if (!config) return;
    memset(config, 0, sizeof(*config));
    config->group_by      = NULL;
    config->group_limit   = 10;
    config->hits_per_group = 3;
    config->distance_type = 0;
    config->oversample    = 0;
}

int group_search(const GV_Database *db, const float *query, size_t dimension,
                     const GV_GroupSearchConfig *config, GV_GroupedResult *result) {
    if (!db || !query || !config || !result || !config->group_by) {
        return -1;
    }
    if (dimension == 0) {
        return -1;
    }

    memset(result, 0, sizeof(*result));

    size_t group_limit    = config->group_limit   > 0 ? config->group_limit   : 10;
    size_t hits_per_group = config->hits_per_group > 0 ? config->hits_per_group : 3;
    size_t oversample     = config->oversample;
    if (oversample == 0) {
        oversample = group_limit * hits_per_group * 4;
    }
    if (oversample == 0) {
        oversample = 40;
    }

    int candidates_on_heap = 0;
    GV_SearchResult *candidates = (GV_SearchResult *)gv_tls_alloc_or_heap(
        oversample * sizeof(GV_SearchResult), sizeof(GV_SearchResult), &candidates_on_heap);
    if (!candidates) {
        return -1;
    }
    memset(candidates, 0, oversample * sizeof(GV_SearchResult));

    int found = db_search(db, query, oversample, candidates,
                          (GV_DistanceType)config->distance_type);
    if (found < 0) {
        gv_tls_free_or_heap(candidates, candidates_on_heap);
        return -1;
    }

    size_t n_candidates = (size_t)found;
    result->total_hits = n_candidates;

    if (n_candidates == 0) {
        gv_tls_free_or_heap(candidates, candidates_on_heap);
        return 0;
    }

    int rc = 0;
    int empty_groups = 0;
    GV_WITH_ARENA(scratch, GV_GROUP_SEARCH_ARENA_BYTES) {
        GroupMap map;
        group_map_init(&map, &scratch, group_limit * 2 > 64 ? group_limit * 2 : 64);

        for (size_t i = 0; i < n_candidates; i++) {
            const GV_Vector *vec = candidates[i].vector;
            if (!vec) continue;

            const char *val = metadata_lookup(vec->metadata, config->group_by);
            if (!val) continue;

            GroupBucket *b = group_map_get_or_create(&map, val);
            if (!b || bucket_add_hit(&map, b, i, candidates[i].distance) != 0) {
                rc = -1;
                break;
            }
        }

        if (rc == 0 && map.bucket_count == 0) {
            empty_groups = 1;
        } else if (rc == 0) {
            for (size_t i = 0; i < map.bucket_count; i++) {
                if (map.buckets[i].hit_count > 1) {
                    qsort(map.buckets[i].hits, map.buckets[i].hit_count,
                          sizeof(GroupHitEntry), compare_hits_by_distance);
                }
            }

            qsort(map.buckets, map.bucket_count, sizeof(GroupBucket),
                  compare_buckets_by_best_distance);

            size_t out_groups = map.bucket_count < group_limit ? map.bucket_count : group_limit;

            result->groups = (GV_SearchGroup *)gv_calloc(out_groups, sizeof(GV_SearchGroup));
            if (!result->groups) {
                rc = -1;
            } else {
                result->group_count = out_groups;

                for (size_t g = 0; g < out_groups && rc == 0; g++) {
                    GroupBucket *b = &map.buckets[g];

                    result->groups[g].group_value = gv_dup_cstr(b->key);
                    if (!result->groups[g].group_value) {
                        group_search_free_result(result);
                        rc = -1;
                        break;
                    }

                    size_t n_hits = b->hit_count < hits_per_group ? b->hit_count : hits_per_group;
                    result->groups[g].hits = (GV_GroupHit *)gv_alloc(n_hits * sizeof(GV_GroupHit));
                    if (!result->groups[g].hits) {
                        group_search_free_result(result);
                        rc = -1;
                        break;
                    }
                    result->groups[g].hit_count = n_hits;

                    for (size_t h = 0; h < n_hits; h++) {
                        result->groups[g].hits[h].index    = b->hits[h].index;
                        result->groups[g].hits[h].distance = b->hits[h].distance;
                    }
                }
            }
        }
    }

    group_search_release_candidates(candidates, n_candidates, candidates_on_heap);
    if (rc != 0) {
        return -1;
    }
    if (empty_groups) {
        return 0;
    }
    return 0;
}

void group_search_free_result(GV_GroupedResult *result) {
    if (!result) return;
    if (result->groups) {
        for (size_t i = 0; i < result->group_count; i++) {
            gv_free(result->groups[i].group_value);
            gv_free(result->groups[i].hits);
        }
        gv_free(result->groups);
    }
    memset(result, 0, sizeof(*result));
}
