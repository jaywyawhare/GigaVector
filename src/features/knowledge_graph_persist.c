/**
  * @file knowledge_graph_persist.c
  * @brief KG WAL attach/checkpoint/replay and binary snapshot save/load
  *        (split from knowledge_graph.c).
  */
#define _POSIX_C_SOURCE 200112L

#include "features/knowledge_graph.h"
#include "features/knowledge_graph_internal.h"
#include "gv_wal_codec.h"
#include "core/memory.h"
#include "core/id_bitmap.h"
#include "core/utils.h"
#include "core/types.h"
#include "storage/database.h"
#include "search/distance.h"
#include "schema/vector.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>
#ifndef _WIN32
#include <unistd.h>
#else
#include <io.h>
#define fsync(fd) _commit(fd)
#endif

static int kg_write_props(FILE *fp, const GV_KGProp *props, size_t count) {
    if (write_u32(fp, (uint32_t)count) != 0) return -1;
    for (const GV_KGProp *p = props; p; p = p->next) {
        if (write_string(fp, p->key) != 0) return -1;
        /* write_prop_value: u8 type byte + varint length + bytes */
        char *s = gv_prop_to_string(p->value);
        if (s == NULL) {
            /* fallback: write as string */
            if (write_string(fp, "<null>") != 0) return -1;
        } else {
            if (write_string(fp, s) != 0) { gv_free(s); return -1; }
            gv_free(s);
        }
    }
    return 0;
}

int kg_wal_attach(GV_KnowledgeGraph *kg, const char *snapshot_path) {
    if (!kg || !snapshot_path) return -1;

    pthread_rwlock_wrlock(&kg->rwlock);
    size_t need = strlen(snapshot_path) + 5;
    char *path = (char *)gv_alloc(need);
    if (!path) {
        pthread_rwlock_unlock(&kg->rwlock);
        return -1;
    }
    snprintf(path, need, "%s.wal", snapshot_path);

    FILE *f = fopen(path, "ab");
    if (!f) {
        gv_free(path);
        pthread_rwlock_unlock(&kg->rwlock);
        return -1;
    }
    kg_wal_close(kg);
    kg->wal_file = f;
    kg->wal_path = path;
    pthread_rwlock_unlock(&kg->rwlock);
    return 0;
}

int kg_wal_checkpoint(GV_KnowledgeGraph *kg) {
    if (!kg) return -1;

    pthread_rwlock_wrlock(&kg->rwlock);
    if (!kg->wal_file) {
        pthread_rwlock_unlock(&kg->rwlock);
        return -1;
    }
    /* cppcheck-suppress leakReturnValNotUsed */
    if (freopen(kg->wal_path, "wb", kg->wal_file) == NULL) {
        kg->wal_file = NULL;
        pthread_rwlock_unlock(&kg->rwlock);
        return -1;
    }
#ifndef _WIN32
    int bad = fflush(kg->wal_file) != 0 || fsync(fileno(kg->wal_file)) != 0;
#else
    int bad = fflush(kg->wal_file) != 0 || _commit(_fileno(kg->wal_file)) != 0;
#endif
    pthread_rwlock_unlock(&kg->rwlock);
    return bad ? -1 : 0;
}

/* Apply one decoded record. Called with the write lock held and wal_replaying
 * set; uses the lock-free internal insert/remove paths. */
static void kg_wal_apply_record(GV_KnowledgeGraph *kg, uint8_t op,
                                const uint8_t *p, const uint8_t *end) {
    char *s1 = NULL, *s2 = NULL;
    const float *emb = NULL;
    uint64_t id, a, b2;
    float w;
    uint8_t has_emb = 0;
    uint32_t dim = 0;

    switch (op) {
    case KG_WAL_OP_ADD_ENTITY:
        if ((p = gv_wal_get_u64(p, end, &id)) != NULL &&
            (p = gv_wal_get_str(p, end, &s1)) != NULL &&
            (p = gv_wal_get_str(p, end, &s2)) != NULL &&
            (p = gv_wal_get_u8(p, end, &has_emb)) != NULL) {
            if (has_emb &&
                (p = gv_wal_get_u32(p, end, &dim)) != NULL) {
                /* floats are stored inline right after dim (no extra length
                 * prefix - dim already bounds them) */
                if ((size_t)(end - p) < (size_t)dim * sizeof(float)) {
                    gv_free(s1);
                    gv_free(s2);
                    return;
                }
                /* cppcheck-suppress invalidPointerCast */
                emb = (const float *)p;
                p += (size_t)dim * sizeof(float);
            }
            if (p && (!has_emb || p))
                kg_insert_entity_with_id(kg, id, s1, s2,
                                         has_emb ? emb : NULL,
                                         has_emb ? dim : 0);
        }
        break;
    case KG_WAL_OP_REMOVE_ENTITY:
        if ((p = gv_wal_get_u64(p, end, &id)) != NULL)
            kg_remove_entity_internal(kg, id);   /* lock already held */
        break;
    case KG_WAL_OP_ADD_RELATION:
        if ((p = gv_wal_get_u64(p, end, &id)) != NULL &&
            (p = gv_wal_get_u64(p, end, &a)) != NULL &&
            (p = gv_wal_get_str(p, end, &s1)) != NULL &&
            (p = gv_wal_get_u64(p, end, &b2)) != NULL &&
            (p = gv_wal_get_f32(p, end, &w)) != NULL)
            kg_insert_relation_with_id(kg, id, a, s1, b2, w);
        break;
    case KG_WAL_OP_REMOVE_RELATION:
        if ((p = gv_wal_get_u64(p, end, &id)) != NULL)
            kg_remove_relation_internal(kg, id);
        break;
    case KG_WAL_OP_SET_ENTITY_PROP: {
        if ((p = gv_wal_get_u64(p, end, &id)) != NULL &&
            (p = gv_wal_get_str(p, end, &s1)) != NULL &&
            (p = gv_wal_get_str(p, end, &s2)) != NULL) {
            KG_EntityNode *n = kg_find_entity_node(kg, id);
            if (n)
                kg_prop_set(&n->entity.properties, &n->entity.prop_count,
                            s1, s2);
        }
        break;
    }
    case KG_WAL_OP_SET_RELATION_PROP: {
        /* Route through kg_set_relation_prop semantics minus logging by
         * updating the property directly; chunk index kept in sync here. */
        if ((p = gv_wal_get_u64(p, end, &id)) != NULL &&
            (p = gv_wal_get_str(p, end, &s1)) != NULL &&
            (p = gv_wal_get_str(p, end, &s2)) != NULL) {
            KG_RelationNode *n = kg_find_relation_node(kg, id);
            if (n) {
                size_t pc = 0;
                for (GV_KGProp *q = n->relation.properties; q; q = q->next) pc++;
                kg_prop_set(&n->relation.properties, &pc, s1, s2);
                if (strcmp(s1, "chunk_id") == 0) {
                    KG_IndexEntry *ce = kg_index_get_or_create(
                        kg->chunk_index, kg->spo_bucket_count,
                        kg_hash_string(s2));
                    if (ce) kg_idlist_push(&ce->list, id);
                }
            }
        }
        break;
    }
    case KG_WAL_OP_REMOVE_ENTITY_PROP:
        if ((p = gv_wal_get_u64(p, end, &id)) != NULL &&
            (p = gv_wal_get_str(p, end, &s1)) != NULL) {
            KG_EntityNode *n = kg_find_entity_node(kg, id);
            if (n)
                kg_prop_remove(&n->entity.properties, &n->entity.prop_count, s1);
        }
        break;
    case KG_WAL_OP_REMOVE_RELATION_PROP:
        if ((p = gv_wal_get_u64(p, end, &id)) != NULL &&
            (p = gv_wal_get_str(p, end, &s1)) != NULL) {
            KG_RelationNode *n = kg_find_relation_node(kg, id);
            if (n) {
                int is_chunk_key = strcmp(s1, "chunk_id") == 0;
GV_KGProp *oldp = is_chunk_key
                    ? kg_prop_find(n->relation.properties, "chunk_id") : NULL;
if (oldp && oldp->value.type != GV_PROP_NULL) {
    kg_index_remove_id(kg->chunk_index, kg->spo_bucket_count,
                       kg_prop_hash(oldp->value), id);
}
                size_t pc = 0;
                for (GV_KGProp *q = n->relation.properties; q; q = q->next) pc++;
                kg_prop_remove(&n->relation.properties, &pc, s1);
            }
        }
        break;
    default:
        break;  /* unknown op: ignore (forward compatibility) */
    }
    gv_free(s1);
    gv_free(s2);
}

/* Decode one framed record from buf[0..len): [1B op][4B plen][payload].
 * Returns bytes consumed, or 0 if more data is needed. */
static size_t kg_wal_parse_record(GV_KnowledgeGraph *kg, const uint8_t *buf,
                                  size_t len) {
    if (len < 5) return 0;
    uint32_t plen = 0;
    for (int i = 0; i < 4; i++) plen |= (uint32_t)buf[1 + i] << (8 * i);
    if (len < 5u + (size_t)plen) return 0;
    kg_wal_apply_record(kg, buf[0], buf + 5, buf + 5 + plen);
    return 5u + (size_t)plen;
}

/* Replay "<path>.wal" over a freshly loaded graph and keep it attached.
 * Returns 0 on success (including no log present). */
static int kg_wal_replay_and_attach(GV_KnowledgeGraph *kg,
                                    const char *snapshot_path) {
    size_t need = strlen(snapshot_path) + 5;
    char *path = (char *)gv_alloc(need);
    if (!path) return -1;
    snprintf(path, need, "%s.wal", snapshot_path);

    FILE *f = fopen(path, "rb");
    if (!f) {
        gv_free(path);
        return 0;
    }

    uint8_t rec[65536];
    size_t cap = sizeof(rec), len = 0, nread;

    pthread_rwlock_wrlock(&kg->rwlock);
    kg->wal_replaying = 1;

    int ok = 1;
    while (ok && (nread = fread(rec + len, 1, cap - len, f)) > 0) {
        len += nread;
        size_t off = 0;
        while (ok) {
            size_t used = kg_wal_parse_record(kg, rec + off, len - off);
            if (used == 0) break;
            off += used;
            if (off >= len) break;
        }
        if (off > 0) {
            memmove(rec, rec + off, len - off);
            len -= off;
        }
    }

    kg->wal_replaying = 0;
    pthread_rwlock_unlock(&kg->rwlock);
    fclose(f);

    FILE *af = fopen(path, "ab");
    if (!af) {
        gv_free(path);
        return -1;
    }
    kg->wal_file = af;
    kg->wal_path = path;
    return 0;
}

int kg_save(const GV_KnowledgeGraph *kg, const char *path) {
    if (!kg || !path) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&kg->rwlock);

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
        return -1;
    }

    if (write_bytes(fp, KG_MAGIC, KG_MAGIC_LEN) != 0) goto fail;
    if (write_u32(fp, KG_VERSION) != 0) goto fail;

    if (write_u32(fp, (uint32_t)kg->config.embedding_dimension) != 0)
        goto fail;
    if (write_f32(fp, kg->config.similarity_threshold) != 0) goto fail;
    if (write_f32(fp, kg->config.link_prediction_threshold) != 0)
        goto fail;

    if (write_u64(fp, (uint64_t)kg->entity_count) != 0) goto fail;
    if (write_u64(fp, (uint64_t)kg->relation_count) != 0) goto fail;
    if (write_u64(fp, kg->next_entity_id) != 0) goto fail;
    if (write_u64(fp, kg->next_relation_id) != 0) goto fail;

    for (size_t b = 0; b < kg->entity_bucket_count; b++) {
        for (KG_EntityNode *n = kg->entity_buckets[b]; n; n = n->next) {
            const GV_KGEntity *e = &n->entity;
            if (write_u64(fp, e->entity_id) != 0) goto fail;
            if (write_string(fp, e->name) != 0) goto fail;
            if (write_string(fp, e->type) != 0) goto fail;

            uint8_t has_emb = (e->embedding && e->dimension > 0) ? 1 : 0;
            if (write_u8(fp, has_emb) != 0) goto fail;
            if (has_emb) {
                if (write_bytes(fp, e->embedding,
                    e->dimension * sizeof(float)) != 0) goto fail;
            }

            if (write_f32(fp, e->confidence) != 0) goto fail;
            if (write_u64(fp, e->created_at) != 0) goto fail;
            if (kg_write_props(fp, e->properties, e->prop_count) != 0)
                goto fail;
        }
    }

    for (size_t b = 0; b < kg->relation_bucket_count; b++) {
        for (KG_RelationNode *n = kg->relation_buckets[b]; n; n = n->next) {
            const GV_KGRelation *r = &n->relation;
            if (write_u64(fp, r->relation_id) != 0) goto fail;
            if (write_u64(fp, r->subject_id) != 0) goto fail;
            if (write_u64(fp, r->object_id) != 0) goto fail;
            if (write_string(fp, r->predicate) != 0) goto fail;
            if (write_f32(fp, r->weight) != 0) goto fail;
            if (write_u64(fp, r->created_at) != 0) goto fail;

            size_t pc = 0;
            for (GV_KGProp *p = r->properties; p; p = p->next) pc++;
            if (kg_write_props(fp, r->properties, pc) != 0) goto fail;
        }
    }

    int is_wal_base = 0;
    if (kg->wal_path) {
        size_t wl = strlen(kg->wal_path);
        size_t pl = strlen(path);
        is_wal_base = wl == pl + 4 &&
                      strncmp(kg->wal_path, path, pl) == 0 &&
                      strcmp(kg->wal_path + pl, ".wal") == 0;
    }

    if (fflush(fp) != 0 || fsync(fileno(fp)) != 0) goto fail;

    fclose(fp);
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);

    /* Successful save of the attached snapshot makes the log redundant. */
    if (is_wal_base) {
        pthread_rwlock_wrlock((pthread_rwlock_t *)&kg->rwlock);
        if (kg->wal_file) {
            /* Truncate the WAL by reopening it; if reopen fails the stream is
             * closed by freopen, so drop our dangling handle rather than use it. */
            FILE *reopened = freopen(kg->wal_path, "wb", kg->wal_file);
            ((GV_KnowledgeGraph *)kg)->wal_file = reopened;
            if (kg->wal_file) {
                fflush(kg->wal_file);
                fsync(fileno(kg->wal_file));
            }
        }
        pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    }
    return 0;

fail:
    fclose(fp);
    pthread_rwlock_unlock((pthread_rwlock_t *)&kg->rwlock);
    return -1;
}


static GV_KGProp *kg_read_props(FILE *fp, uint32_t count) {
    GV_KGProp *head = NULL;
    GV_KGProp *tail = NULL;
    for (uint32_t i = 0; i < count; i++) {
        GV_KGProp *p = (GV_KGProp *)gv_calloc(1, sizeof(GV_KGProp));
        if (!p) return head;
        p->key = read_string(fp);
        char *val_str = read_string(fp);
        p->value = gv_prop_parse(val_str, GV_PROP_STRING);
        gv_free(val_str);
        p->next = NULL;
        if (!p->key || p->value.type == GV_PROP_NULL) {
            gv_free(p->key);
            gv_prop_free(&p->value);
            gv_free(p);
            return head;
        }
        if (tail) tail->next = p;
        else head = p;
        tail = p;
    }
    return head;
}

GV_KnowledgeGraph *kg_load(const char *path) {
    if (!path) return NULL;

    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;

    char magic[KG_MAGIC_LEN];
    if (read_bytes(fp, magic, KG_MAGIC_LEN) != 0 ||
        memcmp(magic, KG_MAGIC, KG_MAGIC_LEN) != 0) {
        fclose(fp);
        return NULL;
    }

    uint32_t version;
    if (read_u32(fp, &version) != 0 || version != KG_VERSION) {
        fclose(fp);
        return NULL;
    }

    uint32_t emb_dim;
    float sim_thresh, lp_thresh;
    if (read_u32(fp, &emb_dim) != 0) { fclose(fp); return NULL; }
    if (read_f32(fp, &sim_thresh) != 0) { fclose(fp); return NULL; }
    if (read_f32(fp, &lp_thresh) != 0) { fclose(fp); return NULL; }

    GV_KGConfig cfg;
    kg_config_init(&cfg);
    cfg.embedding_dimension = (size_t)emb_dim;
    cfg.similarity_threshold = sim_thresh;
    cfg.link_prediction_threshold = lp_thresh;

    uint64_t entity_count, relation_count, next_eid, next_rid;
    if (read_u64(fp, &entity_count) != 0) { fclose(fp); return NULL; }
    if (read_u64(fp, &relation_count) != 0) { fclose(fp); return NULL; }
    if (read_u64(fp, &next_eid) != 0) { fclose(fp); return NULL; }
    if (read_u64(fp, &next_rid) != 0) { fclose(fp); return NULL; }

    GV_KnowledgeGraph *kg = kg_create(&cfg);
    if (!kg) { fclose(fp); return NULL; }
    kg->next_entity_id = next_eid;
    kg->next_relation_id = next_rid;

    for (uint64_t i = 0; i < entity_count; i++) {
        uint64_t eid;
        if (read_u64(fp, &eid) != 0) goto load_fail;

        char *name = read_string(fp);
        char *type = read_string(fp);
        if (!name || !type) {
            gv_free(name);
            gv_free(type);
            goto load_fail;
        }

        uint8_t has_emb;
        if (read_u8(fp, &has_emb) != 0) {
            gv_free(name); gv_free(type);
            goto load_fail;
        }

        float *emb_data = NULL;
        if (has_emb && emb_dim > 0) {
            emb_data = (float *)gv_alloc(emb_dim * sizeof(float));
            if (!emb_data || read_bytes(fp, emb_data,
                emb_dim * sizeof(float)) != 0) {
                gv_free(emb_data); gv_free(name); gv_free(type);
                goto load_fail;
            }
        }

        float confidence;
        uint64_t created_at;
        if (read_f32(fp, &confidence) != 0 ||
            read_u64(fp, &created_at) != 0) {
            gv_free(emb_data); gv_free(name); gv_free(type);
            goto load_fail;
        }

        uint32_t prop_count;
        if (read_u32(fp, &prop_count) != 0) {
            gv_free(emb_data); gv_free(name); gv_free(type);
            goto load_fail;
        }
        GV_KGProp *props = kg_read_props(fp, prop_count);

        KG_EntityNode *node = (KG_EntityNode *)gv_calloc(1,
                                  sizeof(KG_EntityNode));
        if (!node) {
            gv_free(emb_data); gv_free(name); gv_free(type);
            kg_prop_free_list(props);
            goto load_fail;
        }

        GV_KGEntity *e = &node->entity;
        e->entity_id  = eid;
        e->name       = name;
        e->type       = type;
        e->embedding  = emb_data;
        e->dimension  = has_emb ? (size_t)emb_dim : 0;
        e->confidence = confidence;
        e->created_at = created_at;
        e->properties = props;
        e->prop_count = (size_t)prop_count;

        size_t bucket = (size_t)kg_hash_uint64(eid, kg->entity_bucket_count);
        node->next = kg->entity_buckets[bucket];
        kg->entity_buckets[bucket] = node;
        kg->entity_count++;

        kg_str_index_add(kg, kg->name_index, kg->entity_bucket_count, e->name, eid);
        kg_str_index_add(kg, kg->type_index, kg->entity_bucket_count, e->type, eid);

        if (has_emb && emb_data && emb_dim > 0) {
            kg_embedding_add(kg, eid, emb_data, (size_t)emb_dim);
        }
    }

    for (uint64_t i = 0; i < relation_count; i++) {
        uint64_t rid, sid, oid;
        if (read_u64(fp, &rid) != 0 ||
            read_u64(fp, &sid) != 0 ||
            read_u64(fp, &oid) != 0) goto load_fail;

        char *predicate = read_string(fp);
        if (!predicate) goto load_fail;

        float weight;
        uint64_t created_at;
        if (read_f32(fp, &weight) != 0 ||
            read_u64(fp, &created_at) != 0) {
            gv_free(predicate);
            goto load_fail;
        }

        uint32_t prop_count;
        if (read_u32(fp, &prop_count) != 0) {
            gv_free(predicate);
            goto load_fail;
        }
        GV_KGProp *props = kg_read_props(fp, prop_count);

        KG_RelationNode *node = (KG_RelationNode *)gv_calloc(1,
                                    sizeof(KG_RelationNode));
        if (!node) {
            gv_free(predicate);
            kg_prop_free_list(props);
            goto load_fail;
        }

        GV_KGRelation *r = &node->relation;
        r->relation_id = rid;
        r->subject_id  = sid;
        r->object_id   = oid;
        r->predicate   = predicate;
        r->weight      = weight;
        r->created_at  = created_at;
        r->properties  = props;

        size_t bucket = (size_t)kg_hash_uint64(rid, kg->relation_bucket_count);
        node->next = kg->relation_buckets[bucket];
        kg->relation_buckets[bucket] = node;
        kg->relation_count++;

        KG_IndexEntry *se = kg_index_get_or_create(kg->subject_index,
                                                    kg->spo_bucket_count, sid);
        if (se) kg_idlist_push(&se->list, rid);

        KG_IndexEntry *oe = kg_index_get_or_create(kg->object_index,
                                                    kg->spo_bucket_count, oid);
        if (oe) kg_idlist_push(&oe->list, rid);

        uint64_t pred_hash = kg_hash_string(predicate);
        KG_IndexEntry *pe = kg_index_get_or_create(kg->predicate_index,
                                                    kg->spo_bucket_count,
                                                    pred_hash);
        if (pe) kg_idlist_push(&pe->list, rid);

        GV_KGProp *cp = kg_prop_find(r->properties, "chunk_id");
        if (cp && cp->value.type == GV_PROP_STRING && cp->value.as.s) {
            KG_IndexEntry *ce = kg_index_get_or_create(kg->chunk_index,
                                                       kg->spo_bucket_count,
                                                       kg_hash_string(cp->value.as.s));
            if (ce) kg_idlist_push(&ce->list, rid);
        }
    }

    /* Entities and relations are fully loaded; wire up direct adjacency. */
    kg_rebuild_adjacency(kg);

    fclose(fp);

    /* Crash recovery: replay any WAL left over from mutations after the last
     * save, then keep it attached for future appends. */
    if (kg_wal_replay_and_attach(kg, path) != 0) {
        kg_destroy(kg);
        return NULL;
    }
    return kg;

load_fail:
    fclose(fp);
    kg_destroy(kg);
    return NULL;
}
