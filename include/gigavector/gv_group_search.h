#ifndef GIGAVECTOR_GV_GROUP_SEARCH_H
#define GIGAVECTOR_GV_GROUP_SEARCH_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct GV_Database GV_Database;

typedef struct {
    size_t index;
    float distance;
} GV_GroupHit;

typedef struct {
    char *group_value;         /* The metadata value this group represents */
    GV_GroupHit *hits;         /* Top hits within this group */
    size_t hit_count;
} GV_SearchGroup;

typedef struct {
    GV_SearchGroup *groups;
    size_t group_count;
    size_t total_hits;         /* Total hits before grouping */
} GV_GroupedResult;

typedef struct {
    const char *group_by;      /* Metadata field to group by */
    size_t group_limit;        /* Max number of groups to return (default: 10) */
    size_t hits_per_group;     /* Max hits per group (default: 3) */
    int distance_type;         /* Distance metric */
    size_t oversample;         /* How many total candidates to fetch (default: group_limit * hits_per_group * 4) */
} GV_GroupSearchConfig;

void gv_group_search_config_init(GV_GroupSearchConfig *config);

int gv_group_search(const GV_Database *db, const float *query, size_t dimension,
                     const GV_GroupSearchConfig *config, GV_GroupedResult *result);

void gv_group_search_free_result(GV_GroupedResult *result);

#ifdef __cplusplus
}
#endif
#endif
