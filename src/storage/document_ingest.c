/**
 * @file document_ingest.c
 * @brief Document ingest coordinator (Phase 2). Chunk -> embed -> db, and
 *        optionally extract triples -> kg and facts -> memory, all sharing
 *        each chunk's chunk_id.
 */
#include "storage/document_ingest.h"
#include "storage/chunker.h"
#include "core/memory.h"
#include "core/utils.h"

#include <string.h>
#include <stdlib.h>

/* Compute the embedding of one text into out[dim]. Returns 0 on success. */
static int embed_one(const GV_IngestConfig *cfg, const char *textv, size_t dim,
                     float *out) {
    if (cfg->embed != NULL) {
        const char *arr[1] = { textv };
        return cfg->embed(cfg->embed_ctx, arr, 1, dim, out);
    }
    if (cfg->default_embedder != NULL) {
        size_t d = 0;
        float *e = auto_embed_text(cfg->default_embedder, textv, &d);
        if (!e) return -1;
        if (d != dim) { gv_free(e); return -1; }
        memcpy(out, e, dim * sizeof(float));
        gv_free(e);
        return 0;
    }
    return -1; /* no embedder configured */
}

/* Resolve an entity by name (first match) or create it. Returns id, or 0. */
static uint64_t resolve_entity(GV_KnowledgeGraph *kg, const char *name,
                               const char *type) {
    uint64_t ids[1];
    int n = kg_find_entities_by_name(kg, name, ids, 1);
    if (n > 0) return ids[0];
    return kg_add_entity(kg, name, type ? type : "Entity", NULL, 0);
}

int gv_ingest_document(GV_Database *db, GV_KnowledgeGraph *kg, GV_MemoryLayer *mem,
                       const char *text, const char *doc_id,
                       const GV_IngestConfig *cfg, GV_IngestStats *out) {
    if (out) memset(out, 0, sizeof(*out));
    if (!db || !text || !cfg) return -1;

    size_t dim = db->dimension;
    if (dim == 0) return -1;

    size_t max_tr = cfg->max_triples_per_chunk ? cfg->max_triples_per_chunk : 8;
    size_t max_fa = cfg->max_facts_per_chunk ? cfg->max_facts_per_chunk : 4;

    GV_Chunk *chunks = NULL;
    size_t nchunks = 0;
    if (gv_chunk_document(text, cfg->chunk_tokens, cfg->overlap_tokens,
                          doc_id, &chunks, &nchunks) != 0) {
        return -1;
    }

    float *emb = (float *)gv_alloc(dim * sizeof(float));
    if (!emb) { gv_chunks_free(chunks, nchunks); return -1; }

    int rc = 0;
    for (size_t i = 0; i < nchunks; i++) {
        GV_Chunk *ck = &chunks[i];

        /* --- Embedding layer (store the chunk text as metadata "text") --- */
        if (embed_one(cfg, ck->text, dim, emb) != 0) { rc = -1; break; }
        {
            const char *mk[1] = { "text" };
            const char *mv[1] = { ck->text };
            if (db_add_vector_with_id_meta(db, ck->chunk_id, emb, dim, mk, mv, 1) != 0) {
                rc = -1; break;
            }
        }
        if (out) { out->chunks++; out->vectors++; }

        /* --- Graph layer (optional) --- */
        if (kg && cfg->extract_triples) {
            GV_ExtractedTriple tr[32];
            size_t cap = max_tr > 32 ? 32 : max_tr;
            int nt = cfg->extract_triples(cfg->triples_ctx, ck->text, tr, cap);
            for (int t = 0; t < nt; t++) {
                if (!tr[t].subject || !tr[t].predicate || !tr[t].object) continue;
                uint64_t s = resolve_entity(kg, tr[t].subject, tr[t].subject_type);
                uint64_t o = resolve_entity(kg, tr[t].object, tr[t].object_type);
                if (s && o &&
                    kg_add_relation_with_chunk(kg, s, tr[t].predicate, o, 1.0f,
                                               ck->chunk_id) != 0) {
                    if (out) out->triples++;
                }
            }
        }

        /* --- Memory layer (optional) --- */
        if (mem && cfg->extract_facts) {
            char *facts[16];
            size_t cap = max_fa > 16 ? 16 : max_fa;
            memset(facts, 0, sizeof(facts));
            int nf = cfg->extract_facts(cfg->facts_ctx, ck->text, facts, cap);
            for (int f = 0; f < nf; f++) {
                if (!facts[f]) continue;
                if (embed_one(cfg, facts[f], dim, emb) == 0) {
                    GV_MemoryMetadata *md =
                        (GV_MemoryMetadata *)gv_calloc(1, sizeof(GV_MemoryMetadata));
                    if (md) {
                        md->memory_type = GV_MEMORY_TYPE_FACT;
                        md->source = gv_dup_cstr(ck->chunk_id); /* chunk_id join key */
                        md->importance_score = 0.5;
                        char *mid = memory_add(mem, facts[f], emb, md, NULL);
                        if (mid) { gv_free(mid); if (out) out->facts++; }
                        /* memory_add copies the metadata by value; we still own md.
                         * memory_metadata_free frees the fields; free the struct too. */
                        memory_metadata_free(md);
                        gv_free(md);
                    }
                }
                gv_free(facts[f]);
            }
        }
    }

    gv_free(emb);
    gv_chunks_free(chunks, nchunks);
    return rc;
}
