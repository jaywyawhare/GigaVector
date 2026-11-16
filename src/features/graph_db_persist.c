/**
  * @file graph_db_persist.c
  * @brief Graph WAL attach/checkpoint/replay and binary snapshot save/load
  *        (split from graph_db.c).
  */
#define _POSIX_C_SOURCE 200112L

#include "features/graph_db_internal.h"
#include "core/memory.h"
#include "core/utils.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#ifndef _WIN32
#include <unistd.h>
#else
#include <io.h>
#define fsync(fd) _commit(fd)
#endif

int graph_wal_attach(GV_GraphDB *g, const char *snapshot_path)
{
    if (!g || !snapshot_path) return -1;

    pthread_rwlock_wrlock(&g->rwlock);
    size_t need = strlen(snapshot_path) + 5;
    char *path = (char *)gv_alloc(need);
    if (!path) {
        pthread_rwlock_unlock(&g->rwlock);
        return -1;
    }
    snprintf(path, need, "%s.wal", snapshot_path);

    FILE *f = fopen(path, "ab");
    if (!f) {
        gv_free(path);
        pthread_rwlock_unlock(&g->rwlock);
        return -1;
    }
    graph_wal_close(g);
    g->wal_file = f;
    g->wal_path = path;
    pthread_rwlock_unlock(&g->rwlock);
    return 0;
}

int graph_wal_checkpoint(GV_GraphDB *g)
{
    if (!g) return -1;

    pthread_rwlock_wrlock(&g->rwlock);
    if (!g->wal_file) {
        pthread_rwlock_unlock(&g->rwlock);
        return -1;
    }
    /* cppcheck-suppress leakReturnValNotUsed */
    if (freopen(g->wal_path, "wb", g->wal_file) == NULL) {
        g->wal_file = NULL;
        pthread_rwlock_unlock(&g->rwlock);
        return -1;
    }
    if (fflush(g->wal_file) != 0 || fsync(fileno(g->wal_file)) != 0) {
        pthread_rwlock_unlock(&g->rwlock);
        return -1;
    }
    pthread_rwlock_unlock(&g->rwlock);
    return 0;
}

/* Decode one record from buf[0..len): [1B op][4B plen][payload].
 * Applies it to @p g and returns bytes consumed, or 0 if more data is needed
 * (incomplete tail after a crash mid-append). */
size_t gwal_parse_record(GV_GraphDB *g, const uint8_t *buf, size_t len)
{
    if (len < 5) return 0;
    uint32_t plen = 0;
    for (int i = 0; i < 4; i++) plen |= (uint32_t)buf[1 + i] << (8 * i);
    if (len < 5u + (size_t)plen) return 0;
    gwal_apply_record(g, buf[0], buf + 5, buf + 5 + plen);
    return 5u + (size_t)plen;
}

/* Replay "<path>.wal" over a freshly loaded graph and keep the log attached
 * for future appends. Returns 0 on success (including no log present). */
static int graph_wal_replay_and_attach(GV_GraphDB *g, const char *snapshot_path)
{
    size_t need = strlen(snapshot_path) + 5;
    char *path = (char *)gv_alloc(need);
    if (!path) return -1;
    snprintf(path, need, "%s.wal", snapshot_path);

    FILE *f = fopen(path, "rb");
    if (!f) {
        gv_free(path);
        return 0;   /* no log: nothing to recover */
    }

    uint8_t rec[8192];
    size_t cap = sizeof(rec), len = 0, nread;

    pthread_rwlock_wrlock(&g->rwlock);
    g->wal_replaying = 1;

    /* Parse greedily; stop at end-of-file or the first incomplete record
     * (a crash mid-append loses only that record). */
    int ok = 1;
    while (ok && (nread = fread(rec + len, 1, cap - len, f)) > 0) {
        len += nread;
        size_t off = 0;
        while (ok) {
            size_t used = gwal_parse_record(g, rec + off, len - off);
            if (used == 0) break;
            off += used;
            if (off >= len) break;
        }
        if (off > 0) {
            memmove(rec, rec + off, len - off);
            len -= off;
        }
    }

    g->wal_replaying = 0;
    pthread_rwlock_unlock(&g->rwlock);
    fclose(f);

    /* Reopen for appending so recovered mutations stay durable. */
    FILE *af = fopen(path, "ab");
    if (!af) {
        gv_free(path);
        return -1;
    }
    g->wal_file = af;
    g->wal_path = path;
    return 0;
}

int graph_save(const GV_GraphDB *g, const char *path)
{
    if (!g || !path) return -1;

    graph_read_lock(g);

    FILE *f = fopen(path, "wb");
    if (!f) {
        graph_read_unlock(g);
        return -1;
    }

    if (fwrite(GV_GRAPH_MAGIC, 1, GV_GRAPH_MAGIC_LEN, f) != GV_GRAPH_MAGIC_LEN)
        goto save_fail;

    uint32_t version = GV_GRAPH_VERSION;
    if (write_u32(f, version) != 0) goto save_fail;

    if (write_u64(f, (uint64_t)g->node_count) != 0) goto save_fail;
    if (write_u64(f, (uint64_t)g->edge_count) != 0) goto save_fail;
    if (write_u64(f, g->next_node_id) != 0) goto save_fail;
    if (write_u64(f, g->next_edge_id) != 0) goto save_fail;

    for (size_t b = 0; b < g->node_bucket_count; b++) {
        NodeEntry *e = g->node_buckets[b];
        while (e) {
            if (write_u64(f, e->node.node_id) != 0) goto save_fail;
            if (write_string(f, e->node.label) != 0) goto save_fail;

            if (write_u32(f, (uint32_t)e->node.prop_count) != 0) goto save_fail;
            GV_GraphProp *prop = e->node.properties;
            while (prop) {
                if (write_string(f, prop->key) != 0) goto save_fail;
                char *vstr = gv_prop_to_string(prop->value);
                if (vstr == NULL || write_string(f, vstr) != 0) { gv_free(vstr); goto save_fail; }
                gv_free(vstr);
                prop = prop->next;
            }

            e = e->next;
        }
    }

    for (size_t b = 0; b < g->edge_bucket_count; b++) {
        EdgeEntry *e = g->edge_buckets[b];
        while (e) {
            if (write_u64(f, e->edge.edge_id) != 0) goto save_fail;
            if (write_u64(f, e->edge.source_id) != 0) goto save_fail;
            if (write_u64(f, e->edge.target_id) != 0) goto save_fail;
            if (write_string(f, e->edge.label) != 0) goto save_fail;
            if (write_f32(f, e->edge.weight) != 0) goto save_fail;

            if (write_u32(f, (uint32_t)e->edge.prop_count) != 0) goto save_fail;
            GV_GraphProp *prop = e->edge.properties;
            while (prop) {
                if (write_string(f, prop->key) != 0) goto save_fail;
                char *vstr = gv_prop_to_string(prop->value);
                if (vstr == NULL || write_string(f, vstr) != 0) { gv_free(vstr); goto save_fail; }
                gv_free(vstr);
                prop = prop->next;
            }

            e = e->next;
        }
    }

    int is_wal_base = 0;
    if (g->wal_path) {
        size_t wl = strlen(g->wal_path);
        size_t pl = strlen(path);
        is_wal_base = wl == pl + 4 &&
                      strncmp(g->wal_path, path, pl) == 0 &&
                      strcmp(g->wal_path + pl, ".wal") == 0;
    }

    if (fflush(f) != 0 || fsync(fileno(f)) != 0)
        goto save_fail;

    fclose(f);
    graph_read_unlock(g);

    /* Successful save of the attached snapshot makes the log redundant. */
    if (is_wal_base) {
        pthread_rwlock_wrlock((pthread_rwlock_t *)&g->rwlock);
        ((GV_GraphDB *)g)->wal_file = freopen(g->wal_path, "wb", g->wal_file);  /* truncate */
        if (g->wal_file) {
            fflush(g->wal_file);
            fsync(fileno(g->wal_file));
        }
        graph_read_unlock(g);
    }
    return 0;

save_fail:
    fclose(f);
    graph_read_unlock(g);
    return -1;
}

GV_GraphDB *graph_load(const char *path)
{
    if (!path) return NULL;

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    char magic[GV_GRAPH_MAGIC_LEN];
    if (fread(magic, 1, GV_GRAPH_MAGIC_LEN, f) != GV_GRAPH_MAGIC_LEN) {
        fclose(f);
        return NULL;
    }
    if (memcmp(magic, GV_GRAPH_MAGIC, GV_GRAPH_MAGIC_LEN) != 0) {
        fclose(f);
        return NULL;
    }

    uint32_t version;
    if (read_u32(f, &version) != 0 || version != GV_GRAPH_VERSION) {
        fclose(f);
        return NULL;
    }

    uint64_t node_count, edge_count, next_node_id, next_edge_id;
    if (read_u64(f, &node_count) != 0) { fclose(f); return NULL; }
    if (read_u64(f, &edge_count) != 0) { fclose(f); return NULL; }
    if (read_u64(f, &next_node_id) != 0) { fclose(f); return NULL; }
    if (read_u64(f, &next_edge_id) != 0) { fclose(f); return NULL; }

    GV_GraphDBConfig cfg;
    graph_config_init(&cfg);
    /* Temporarily disable referential integrity during load to be safe. */
    cfg.enforce_referential_integrity = 0;
    /* Presize for the on-disk counts so no rehash churn happens mid-load. */
    while (cfg.node_bucket_count < node_count * 2) cfg.node_bucket_count *= 2;
    while (cfg.edge_bucket_count < edge_count * 2) cfg.edge_bucket_count *= 2;

    GV_GraphDB *g = graph_create(&cfg);
    if (!g) {
        fclose(f);
        return NULL;
    }

    for (uint64_t i = 0; i < node_count; i++) {
        uint64_t nid;
        if (read_u64(f, &nid) != 0) goto load_fail;

        char *label = read_string(f);
        if (!label) goto load_fail;

        uint32_t prop_count;
        if (read_u32(f, &prop_count) != 0) {
            gv_free(label);
            goto load_fail;
        }

        NodeEntry *entry = node_pool_alloc(g);
        if (!entry) {
            gv_free(label);
            goto load_fail;
        }
        entry->node.node_id = nid;
        entry->node.label = label;
        entry->node.properties = NULL;
        entry->node.prop_count = 0;
        entry->node.out_edges = NULL;
        entry->node.out_count = 0;
        entry->node.out_cap = 0;
        entry->node.in_edges = NULL;
        entry->node.in_count = 0;
        entry->node.in_cap = 0;

        for (uint32_t p = 0; p < prop_count; p++) {
            char *key = read_string(f);
            char *val = read_string(f);
            if (!key || !val) {
                gv_free(key); gv_free(val);
                node_pool_free(g, entry);
                goto load_fail;
            }
            if (set_prop(&entry->node.properties, &entry->node.prop_count,
                         key, val) != 0) {
                gv_free(key); gv_free(val);
                node_pool_free(g, entry);
                goto load_fail;
            }
            gv_free(key);
            gv_free(val);
        }

        if (label_index_add(g, label, nid) != 0) {
            node_pool_free(g, entry);
            goto load_fail;
        }

        size_t idx = hash_u64(nid, g->node_bucket_count);
        entry->next = g->node_buckets[idx];
        g->node_buckets[idx] = entry;
        g->node_count++;
    }
    g->next_node_id = next_node_id;

    for (uint64_t i = 0; i < edge_count; i++) {
        uint64_t eid, src_id, tgt_id;
        if (read_u64(f, &eid) != 0) goto load_fail;
        if (read_u64(f, &src_id) != 0) goto load_fail;
        if (read_u64(f, &tgt_id) != 0) goto load_fail;

        char *label = read_string(f);
        if (!label) goto load_fail;

        float weight;
        if (read_f32(f, &weight) != 0) {
            gv_free(label);
            goto load_fail;
        }

        uint32_t prop_count;
        if (read_u32(f, &prop_count) != 0) {
            gv_free(label);
            goto load_fail;
        }

        EdgeEntry *entry = edge_pool_alloc(g);
        if (!entry) {
            gv_free(label);
            goto load_fail;
        }
        entry->edge.edge_id = eid;
        entry->edge.source_id = src_id;
        entry->edge.target_id = tgt_id;
        entry->edge.label = label;
        entry->edge.weight = weight;
        entry->edge.properties = NULL;
        entry->edge.prop_count = 0;

        for (uint32_t p = 0; p < prop_count; p++) {
            char *key = read_string(f);
            char *val = read_string(f);
            if (!key || !val) {
                gv_free(key); gv_free(val);
                edge_pool_free(g, entry);
                goto load_fail;
            }
            if (set_prop(&entry->edge.properties, &entry->edge.prop_count,
                         key, val) != 0) {
                gv_free(key); gv_free(val);
                edge_pool_free(g, entry);
                goto load_fail;
            }
            gv_free(key);
            gv_free(val);
        }

        size_t idx = hash_u64(eid, g->edge_bucket_count);
        entry->next = g->edge_buckets[idx];
        g->edge_buckets[idx] = entry;
        g->edge_count++;

        NodeEntry *src_node = find_node_entry(g, src_id);
        NodeEntry *tgt_node = find_node_entry(g, tgt_id);
        /* Abort the load on adjacency OOM rather than build a graph whose edge
         * table and adjacency lists disagree. `entry` is already linked into the
         * bucket, so graph_destroy() (via load_fail) frees it. */
        if (src_node &&
            adj_add(&src_node->node.out_edges, &src_node->node.out_count,
                    &src_node->node.out_cap, eid, tgt_id) != 0) {
            goto load_fail;
        }
        if (tgt_node &&
            adj_add(&tgt_node->node.in_edges, &tgt_node->node.in_count,
                    &tgt_node->node.in_cap, eid, src_id) != 0) {
            goto load_fail;
        }
    }
    g->next_edge_id = next_edge_id;

    g->enforce_referential_integrity = 1;

    fclose(f);

    /* Crash recovery: replay any WAL left over from mutations after the last
     * save, and keep it attached for future appends. */
    int wrc = graph_wal_replay_and_attach(g, path);
    if (wrc != 0) {
        graph_destroy(g);
        return NULL;
    }
    journal_break_locked(g);   /* replayed mutations aren't journaled */
    return g;

load_fail:
    fclose(f);
    graph_destroy(g);
    return NULL;
}

int graph_get_all_node_ids(const GV_GraphDB *g, uint64_t *out_ids, size_t max_count) {
    if (!g || !out_ids) return -1;
    graph_read_lock(g);
    if (g->node_count > max_count) {
        graph_read_unlock(g);
        return -1;
    }
    size_t n = 0;
    uint64_t *ids = collect_all_node_ids(g, &n);
    if (!ids && n > 0) {
        graph_read_unlock(g);
        return -1;
    }
    for (size_t i = 0; i < n; i++) out_ids[i] = ids[i];
    gv_free(ids);
    graph_read_unlock(g);
    return (int)n;
}

int graph_get_all_node_ids_unlocked(const GV_GraphDB *g, uint64_t *out_ids,
                                    size_t max_count)
{
    if (!g || !out_ids) return -1;
    if (g->node_count > max_count) return -1;
    size_t n = 0;
    uint64_t *ids = collect_all_node_ids(g, &n);
    if (!ids && n > 0) return -1;
    for (size_t i = 0; i < n; i++) out_ids[i] = ids[i];
    gv_free(ids);
    return (int)n;
}
