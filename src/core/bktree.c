#include <string.h>
#include <stdlib.h>

#include "core/bktree.h"
#include "core/memory.h"

/* ------------------------------------------------------------------ */
/* Levenshtein distance                                                 */
/* ------------------------------------------------------------------ */

#define LEV_STACK_MAX 257   /* strings <= 256 chars use stack storage */

static int levenshtein(const char *a, const char *b) {
    size_t la = strlen(a);
    size_t lb = strlen(b);

    /* Allocate two rows.  Use the stack for short strings. */
    int  stack_prev[LEV_STACK_MAX];
    int  stack_curr[LEV_STACK_MAX];
    int *prev, *curr;

    size_t cols = lb + 1;

    if (cols <= LEV_STACK_MAX) {
        prev = stack_prev;
        curr = stack_curr;
    } else {
        prev = (int *)gv_alloc(cols * sizeof(int));
        curr = (int *)gv_alloc(cols * sizeof(int));
        if (!prev || !curr) {
            gv_free(prev);
            gv_free(curr);
            return (int)(la > lb ? la : lb); /* degrade gracefully */
        }
    }

    /* Initialise the first row (edit distance from "" to b[0..j]) */
    for (size_t j = 0; j < cols; j++) prev[j] = (int)j;

    for (size_t i = 1; i <= la; i++) {
        curr[0] = (int)i;
        for (size_t j = 1; j <= lb; j++) {
            int cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            int del  = prev[j]     + 1;
            int ins  = curr[j - 1] + 1;
            int sub  = prev[j - 1] + cost;
            int best = del < ins ? del : ins;
            if (sub < best) best = sub;
            curr[j] = best;
        }
        /* swap rows */
        int *tmp = prev; prev = curr; curr = tmp;
    }

    int result = prev[lb];

    if (cols > LEV_STACK_MAX) {
        gv_free(prev);
        gv_free(curr);
    }

    return result;
}

/* ------------------------------------------------------------------ */
/* BK-tree node helpers                                                 */
/* ------------------------------------------------------------------ */

static GV_BKNode *bknode_create(const char *word) {
    GV_BKNode *n = (GV_BKNode *)gv_alloc(sizeof(GV_BKNode));
    if (!n) return NULL;
    n->word        = gv_strdup(word);
    n->child_dists = NULL;
    n->children    = NULL;
    n->nchildren   = 0;
    n->cap         = 0;
    if (!n->word) { gv_free(n); return NULL; }
    return n;
}

static void bknode_destroy(GV_BKNode *n) {
    if (!n) return;
    gv_free(n->word);
    for (size_t i = 0; i < n->nchildren; i++)
        bknode_destroy(n->children[i]);
    gv_free(n->children);
    gv_free(n->child_dists);
    gv_free(n);
}

/* Add a child at the given distance edge from this node. */
static int bknode_add_child(GV_BKNode *parent, int dist, GV_BKNode *child) {
    if (parent->nchildren == parent->cap) {
        size_t newcap = parent->cap ? parent->cap * 2 : 4;
        int *new_dists      = (int *)gv_realloc(parent->child_dists, newcap * sizeof(int));
        GV_BKNode **new_ch  = (GV_BKNode **)gv_realloc(parent->children, newcap * sizeof(GV_BKNode *));
        if (!new_dists || !new_ch) return -1;
        parent->child_dists = new_dists;
        parent->children    = new_ch;
        parent->cap         = newcap;
    }
    parent->child_dists[parent->nchildren] = dist;
    parent->children[parent->nchildren]    = child;
    parent->nchildren++;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

GV_BKTree *bktree_create(void) {
    GV_BKTree *t = (GV_BKTree *)gv_alloc(sizeof(GV_BKTree));
    if (!t) return NULL;
    t->root  = NULL;
    t->count = 0;
    return t;
}

void bktree_insert(GV_BKTree *t, const char *word) {
    if (!t || !word) return;

    GV_BKNode *node = bknode_create(word);
    if (!node) return;

    if (!t->root) {
        t->root = node;
        t->count++;
        return;
    }

    GV_BKNode *cur = t->root;
    for (;;) {
        int d = levenshtein(word, cur->word);
        if (d == 0) {
            /* duplicate — discard the new node */
            bknode_destroy(node);
            return;
        }
        /* find existing child with this edge distance */
        GV_BKNode *next = NULL;
        for (size_t i = 0; i < cur->nchildren; i++) {
            if (cur->child_dists[i] == d) {
                next = cur->children[i];
                break;
            }
        }
        if (next) {
            cur = next;
        } else {
            bknode_add_child(cur, d, node);
            t->count++;
            return;
        }
    }
}

/* Recursive search helper */
static void bktree_search_node(const GV_BKNode *n, const char *query,
                                int max_dist,
                                const char **results,
                                size_t *nresults, size_t max_results) {
    if (!n || *nresults >= max_results) return;

    int d = levenshtein(query, n->word);
    if (d <= max_dist) {
        results[(*nresults)++] = n->word;
        if (*nresults >= max_results) return;
    }

    /* Visit children whose edge distance is within [d - max_dist, d + max_dist] */
    int lo = d - max_dist;
    int hi = d + max_dist;
    for (size_t i = 0; i < n->nchildren; i++) {
        int ed = n->child_dists[i];
        if (ed >= lo && ed <= hi) {
            bktree_search_node(n->children[i], query, max_dist,
                               results, nresults, max_results);
            if (*nresults >= max_results) return;
        }
    }
}

void bktree_search(const GV_BKTree *t, const char *query, int max_dist,
                   const char **results, size_t *nresults, size_t max_results) {
    if (!t || !query || !results || !nresults) return;
    *nresults = 0;
    if (!t->root || max_results == 0) return;
    bktree_search_node(t->root, query, max_dist, results, nresults, max_results);
}

void bktree_destroy(GV_BKTree *t) {
    if (!t) return;
    bknode_destroy(t->root);
    gv_free(t);
}
