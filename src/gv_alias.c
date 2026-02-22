/**
 * @file gv_alias.c
 * @brief Collection alias implementation for atomic blue-green deployments.
 */

#include "gigavector/gv_alias.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>

/*  Internal Structures  */

#define MAX_ALIASES       256
#define ALIAS_NAME_MAX    128
#define COLLECTION_MAX    256

/**
 * @brief FNV-1a hash offset basis and prime for 32-bit.
 */
#define FNV_OFFSET_BASIS  0x811C9DC5u
#define FNV_PRIME         0x01000193u

/**
 * @brief Single alias entry within the open-addressing hash table.
 */
typedef struct {
    int      occupied;                       /**< 1 if this slot is in use. */
    char     alias_name[ALIAS_NAME_MAX];     /**< Alias name (key). */
    char     collection_name[COLLECTION_MAX];/**< Target collection (value). */
    uint64_t created_at;                     /**< Unix timestamp of creation. */
    uint64_t updated_at;                     /**< Unix timestamp of last update. */
} GV_AliasEntry;

/**
 * @brief Alias manager internal structure.
 */
struct GV_AliasManager {
    GV_AliasEntry    entries[MAX_ALIASES];
    size_t           count;
    pthread_rwlock_t rwlock;
};

/*  Hash Function (FNV-1a)  */

static uint32_t fnv1a(const char *str) {
    uint32_t hash = FNV_OFFSET_BASIS;
    for (const unsigned char *p = (const unsigned char *)str; *p; p++) {
        hash ^= (uint32_t)*p;
        hash *= FNV_PRIME;
    }
    return hash;
}

/*  Internal Helpers  */

/**
 * @brief Locate the slot index for a given alias name using open addressing
 *        with linear probing.
 *
 * @return Slot index if found, or (size_t)-1 if not present.
 */
static size_t find_slot(const GV_AliasManager *mgr, const char *alias_name) {
    uint32_t h = fnv1a(alias_name);
    for (size_t i = 0; i < MAX_ALIASES; i++) {
        size_t idx = (h + i) % MAX_ALIASES;
        const GV_AliasEntry *e = &mgr->entries[idx];
        if (!e->occupied) {
            return (size_t)-1;  /* Empty slot means the name does not exist. */
        }
        if (strcmp(e->alias_name, alias_name) == 0) {
            return idx;
        }
    }
    return (size_t)-1;  /* Table full, not found. */
}

/**
 * @brief Find an empty slot for insertion using open addressing.
 *
 * @return Slot index, or (size_t)-1 if the table is full.
 */
static size_t find_empty_slot(const GV_AliasManager *mgr, const char *alias_name) {
    uint32_t h = fnv1a(alias_name);
    for (size_t i = 0; i < MAX_ALIASES; i++) {
        size_t idx = (h + i) % MAX_ALIASES;
        if (!mgr->entries[idx].occupied) {
            return idx;
        }
    }
    return (size_t)-1;
}

static uint64_t current_time_unix(void) {
    return (uint64_t)time(NULL);
}

/*  Lifecycle  */

GV_AliasManager *gv_alias_manager_create(void) {
    GV_AliasManager *mgr = calloc(1, sizeof(GV_AliasManager));
    if (!mgr) return NULL;

    if (pthread_rwlock_init(&mgr->rwlock, NULL) != 0) {
        free(mgr);
        return NULL;
    }

    return mgr;
}

void gv_alias_manager_destroy(GV_AliasManager *mgr) {
    if (!mgr) return;

    pthread_rwlock_destroy(&mgr->rwlock);
    free(mgr);
}

/*  Alias Operations  */

int gv_alias_create(GV_AliasManager *mgr, const char *alias_name,
                    const char *collection_name) {
    if (!mgr || !alias_name || !collection_name) return -1;
    if (strlen(alias_name) >= ALIAS_NAME_MAX) return -1;
    if (strlen(collection_name) >= COLLECTION_MAX) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    /* Reject if alias already exists. */
    if (find_slot(mgr, alias_name) != (size_t)-1) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    /* Reject if table is full. */
    if (mgr->count >= MAX_ALIASES) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    size_t idx = find_empty_slot(mgr, alias_name);
    if (idx == (size_t)-1) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    GV_AliasEntry *e = &mgr->entries[idx];
    e->occupied = 1;
    strncpy(e->alias_name, alias_name, ALIAS_NAME_MAX - 1);
    e->alias_name[ALIAS_NAME_MAX - 1] = '\0';
    strncpy(e->collection_name, collection_name, COLLECTION_MAX - 1);
    e->collection_name[COLLECTION_MAX - 1] = '\0';

    uint64_t now = current_time_unix();
    e->created_at = now;
    e->updated_at = now;

    mgr->count++;

    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

int gv_alias_update(GV_AliasManager *mgr, const char *alias_name,
                    const char *new_collection_name) {
    if (!mgr || !alias_name || !new_collection_name) return -1;
    if (strlen(new_collection_name) >= COLLECTION_MAX) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    size_t idx = find_slot(mgr, alias_name);
    if (idx == (size_t)-1) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    GV_AliasEntry *e = &mgr->entries[idx];
    strncpy(e->collection_name, new_collection_name, COLLECTION_MAX - 1);
    e->collection_name[COLLECTION_MAX - 1] = '\0';
    e->updated_at = current_time_unix();

    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

int gv_alias_delete(GV_AliasManager *mgr, const char *alias_name) {
    if (!mgr || !alias_name) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    size_t idx = find_slot(mgr, alias_name);
    if (idx == (size_t)-1) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    /*
     * Open-addressing deletion: mark slot empty then rehash any entries
     * in the same cluster that may have been displaced past this slot.
     */
    mgr->entries[idx].occupied = 0;
    mgr->count--;

    /* Rehash subsequent entries that belong to the same cluster. */
    size_t cur = (idx + 1) % MAX_ALIASES;
    while (mgr->entries[cur].occupied) {
        GV_AliasEntry tmp = mgr->entries[cur];
        mgr->entries[cur].occupied = 0;
        mgr->count--;

        /* Re-insert the displaced entry. */
        size_t new_idx = find_empty_slot(mgr, tmp.alias_name);
        mgr->entries[new_idx] = tmp;
        mgr->count++;

        cur = (cur + 1) % MAX_ALIASES;
    }

    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

int gv_alias_exists(const GV_AliasManager *mgr, const char *alias_name) {
    if (!mgr || !alias_name) return 0;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&mgr->rwlock);
    int exists = find_slot(mgr, alias_name) != (size_t)-1 ? 1 : 0;
    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);

    return exists;
}

/*  Atomic Swap  */

int gv_alias_swap(GV_AliasManager *mgr, const char *alias_a, const char *alias_b) {
    if (!mgr || !alias_a || !alias_b) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    size_t idx_a = find_slot(mgr, alias_a);
    size_t idx_b = find_slot(mgr, alias_b);

    if (idx_a == (size_t)-1 || idx_b == (size_t)-1) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    /* Swap collection names. */
    char tmp[COLLECTION_MAX];
    memcpy(tmp, mgr->entries[idx_a].collection_name, COLLECTION_MAX);
    memcpy(mgr->entries[idx_a].collection_name,
           mgr->entries[idx_b].collection_name, COLLECTION_MAX);
    memcpy(mgr->entries[idx_b].collection_name, tmp, COLLECTION_MAX);

    uint64_t now = current_time_unix();
    mgr->entries[idx_a].updated_at = now;
    mgr->entries[idx_b].updated_at = now;

    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

/*  Resolve  */

const char *gv_alias_resolve(const GV_AliasManager *mgr, const char *alias_name) {
    if (!mgr || !alias_name) return NULL;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&mgr->rwlock);

    size_t idx = find_slot(mgr, alias_name);
    const char *result = NULL;
    if (idx != (size_t)-1) {
        result = mgr->entries[idx].collection_name;
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
    return result;
}

/*  List / Info / Count  */

int gv_alias_list(const GV_AliasManager *mgr, GV_AliasInfo **out_list,
                  size_t *out_count) {
    if (!mgr || !out_list || !out_count) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&mgr->rwlock);

    *out_count = mgr->count;
    if (mgr->count == 0) {
        *out_list = NULL;
        pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
        return 0;
    }

    *out_list = calloc(mgr->count, sizeof(GV_AliasInfo));
    if (!*out_list) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
        return -1;
    }

    size_t pos = 0;
    for (size_t i = 0; i < MAX_ALIASES && pos < mgr->count; i++) {
        const GV_AliasEntry *e = &mgr->entries[i];
        if (!e->occupied) continue;

        (*out_list)[pos].alias_name      = strdup(e->alias_name);
        (*out_list)[pos].collection_name = strdup(e->collection_name);
        (*out_list)[pos].created_at      = e->created_at;
        (*out_list)[pos].updated_at      = e->updated_at;
        pos++;
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
    return 0;
}

void gv_alias_free_list(GV_AliasInfo *list, size_t count) {
    if (!list) return;
    for (size_t i = 0; i < count; i++) {
        free(list[i].alias_name);
        free(list[i].collection_name);
    }
    free(list);
}

int gv_alias_get_info(const GV_AliasManager *mgr, const char *alias_name,
                      GV_AliasInfo *info) {
    if (!mgr || !alias_name || !info) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&mgr->rwlock);

    size_t idx = find_slot(mgr, alias_name);
    if (idx == (size_t)-1) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
        return -1;
    }

    const GV_AliasEntry *e = &mgr->entries[idx];
    info->alias_name      = strdup(e->alias_name);
    info->collection_name = strdup(e->collection_name);
    info->created_at      = e->created_at;
    info->updated_at      = e->updated_at;

    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
    return 0;
}

size_t gv_alias_count(const GV_AliasManager *mgr) {
    if (!mgr) return 0;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&mgr->rwlock);
    size_t c = mgr->count;
    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);

    return c;
}

/*  Persistence  */

/**
 * Binary format:
 *   uint32_t  count
 *   For each alias:
 *     uint32_t  alias_name_len   (excluding NUL)
 *     char[]    alias_name
 *     uint32_t  collection_name_len (excluding NUL)
 *     char[]    collection_name
 *     uint64_t  created_at
 *     uint64_t  updated_at
 */

int gv_alias_save(const GV_AliasManager *mgr, const char *filepath) {
    if (!mgr || !filepath) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&mgr->rwlock);

    FILE *fp = fopen(filepath, "wb");
    if (!fp) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
        return -1;
    }

    uint32_t count = (uint32_t)mgr->count;
    if (fwrite(&count, sizeof(uint32_t), 1, fp) != 1) {
        fclose(fp);
        pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
        return -1;
    }

    for (size_t i = 0; i < MAX_ALIASES; i++) {
        const GV_AliasEntry *e = &mgr->entries[i];
        if (!e->occupied) continue;

        uint32_t aname_len = (uint32_t)strlen(e->alias_name);
        uint32_t cname_len = (uint32_t)strlen(e->collection_name);

        if (fwrite(&aname_len, sizeof(uint32_t), 1, fp) != 1) goto fail;
        if (fwrite(e->alias_name, 1, aname_len, fp) != aname_len) goto fail;
        if (fwrite(&cname_len, sizeof(uint32_t), 1, fp) != 1) goto fail;
        if (fwrite(e->collection_name, 1, cname_len, fp) != cname_len) goto fail;
        if (fwrite(&e->created_at, sizeof(uint64_t), 1, fp) != 1) goto fail;
        if (fwrite(&e->updated_at, sizeof(uint64_t), 1, fp) != 1) goto fail;
    }

    fclose(fp);
    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
    return 0;

fail:
    fclose(fp);
    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
    return -1;
}

GV_AliasManager *gv_alias_load(const char *filepath) {
    if (!filepath) return NULL;

    FILE *fp = fopen(filepath, "rb");
    if (!fp) return NULL;

    uint32_t count = 0;
    if (fread(&count, sizeof(uint32_t), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }

    if (count > MAX_ALIASES) {
        fclose(fp);
        return NULL;
    }

    GV_AliasManager *mgr = gv_alias_manager_create();
    if (!mgr) {
        fclose(fp);
        return NULL;
    }

    for (uint32_t i = 0; i < count; i++) {
        uint32_t aname_len = 0;
        uint32_t cname_len = 0;
        char alias_buf[ALIAS_NAME_MAX];
        char coll_buf[COLLECTION_MAX];
        uint64_t created_at = 0;
        uint64_t updated_at = 0;

        if (fread(&aname_len, sizeof(uint32_t), 1, fp) != 1) goto fail;
        if (aname_len >= ALIAS_NAME_MAX) goto fail;
        if (fread(alias_buf, 1, aname_len, fp) != aname_len) goto fail;
        alias_buf[aname_len] = '\0';

        if (fread(&cname_len, sizeof(uint32_t), 1, fp) != 1) goto fail;
        if (cname_len >= COLLECTION_MAX) goto fail;
        if (fread(coll_buf, 1, cname_len, fp) != cname_len) goto fail;
        coll_buf[cname_len] = '\0';

        if (fread(&created_at, sizeof(uint64_t), 1, fp) != 1) goto fail;
        if (fread(&updated_at, sizeof(uint64_t), 1, fp) != 1) goto fail;

        /* Insert directly into the hash table (bypasses timestamp generation). */
        size_t idx = find_empty_slot(mgr, alias_buf);
        if (idx == (size_t)-1) goto fail;

        GV_AliasEntry *e = &mgr->entries[idx];
        e->occupied = 1;
        strncpy(e->alias_name, alias_buf, ALIAS_NAME_MAX - 1);
        e->alias_name[ALIAS_NAME_MAX - 1] = '\0';
        strncpy(e->collection_name, coll_buf, COLLECTION_MAX - 1);
        e->collection_name[COLLECTION_MAX - 1] = '\0';
        e->created_at = created_at;
        e->updated_at = updated_at;
        mgr->count++;
    }

    fclose(fp);
    return mgr;

fail:
    fclose(fp);
    gv_alias_manager_destroy(mgr);
    return NULL;
}
