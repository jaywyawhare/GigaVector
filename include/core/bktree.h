#ifndef GV_BKTREE_H
#define GV_BKTREE_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct GV_BKNode {
    char   *word;
    int    *child_dists;
    struct GV_BKNode **children;
    size_t  nchildren;
    size_t  cap;
} GV_BKNode;

typedef struct {
    GV_BKNode *root;
    size_t     count;
} GV_BKTree;

GV_BKTree *bktree_create(void);
void       bktree_insert(GV_BKTree *t, const char *word);
/* results/nresults: caller-allocated array of char* pointers (not copies — point into tree) */
void       bktree_search(const GV_BKTree *t, const char *query, int max_dist,
                         const char **results, size_t *nresults, size_t max_results);
void       bktree_destroy(GV_BKTree *t);

#ifdef __cplusplus
}
#endif
#endif
