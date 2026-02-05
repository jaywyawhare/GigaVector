/**
 * @file gv_namespace.c
 * @brief Namespace/Collection implementation for multi-tenancy.
 */

#include "gigavector/gv_namespace.h"
#include "gigavector/gv_database.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>
#include <sys/stat.h>
#include <dirent.h>

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

#define MAX_NAMESPACE_NAME 64
#define MAX_NAMESPACES 256

/**
 * @brief Namespace internal structure.
 */
struct GV_Namespace {
    char name[MAX_NAMESPACE_NAME];
    GV_Database *db;
    size_t max_vectors;
    size_t max_memory_bytes;
    uint64_t created_at;
    uint64_t last_modified;
    char *filepath;
    pthread_mutex_t mutex;
};

/**
 * @brief Namespace manager internal structure.
 */
struct GV_NamespaceManager {
    GV_Namespace *namespaces[MAX_NAMESPACES];
    size_t namespace_count;
    char *base_path;
    pthread_rwlock_t rwlock;
};

/* ============================================================================
 * Configuration
 * ============================================================================ */

void gv_namespace_config_init(GV_NamespaceConfig *config) {
    if (!config) return;
    memset(config, 0, sizeof(*config));
    config->index_type = GV_NS_INDEX_HNSW;
}

/* ============================================================================
 * Namespace Manager Lifecycle
 * ============================================================================ */

GV_NamespaceManager *gv_namespace_manager_create(const char *base_path) {
    GV_NamespaceManager *mgr = calloc(1, sizeof(GV_NamespaceManager));
    if (!mgr) return NULL;

    if (pthread_rwlock_init(&mgr->rwlock, NULL) != 0) {
        free(mgr);
        return NULL;
    }

    if (base_path) {
        mgr->base_path = strdup(base_path);
        /* Create directory if it doesn't exist */
        mkdir(base_path, 0755);
    }

    return mgr;
}

void gv_namespace_manager_destroy(GV_NamespaceManager *mgr) {
    if (!mgr) return;

    /* Close all namespaces */
    for (size_t i = 0; i < mgr->namespace_count; i++) {
        if (mgr->namespaces[i]) {
            pthread_mutex_destroy(&mgr->namespaces[i]->mutex);
            if (mgr->namespaces[i]->db) {
                gv_db_close(mgr->namespaces[i]->db);
            }
            free(mgr->namespaces[i]->filepath);
            free(mgr->namespaces[i]);
        }
    }

    pthread_rwlock_destroy(&mgr->rwlock);
    free(mgr->base_path);
    free(mgr);
}

/* ============================================================================
 * Internal Helpers
 * ============================================================================ */

static GV_Namespace *find_namespace(GV_NamespaceManager *mgr, const char *name) {
    for (size_t i = 0; i < mgr->namespace_count; i++) {
        if (mgr->namespaces[i] && strcmp(mgr->namespaces[i]->name, name) == 0) {
            return mgr->namespaces[i];
        }
    }
    return NULL;
}

static char *build_filepath(const char *base_path, const char *name) {
    if (!base_path) return NULL;
    size_t len = strlen(base_path) + strlen(name) + 10;
    char *path = malloc(len);
    if (!path) return NULL;
    snprintf(path, len, "%s/%s.gvdb", base_path, name);
    return path;
}

static GV_IndexType ns_index_to_db_index(GV_NSIndexType ns_type) {
    switch (ns_type) {
        case GV_NS_INDEX_KDTREE: return GV_INDEX_TYPE_KDTREE;
        case GV_NS_INDEX_HNSW:   return GV_INDEX_TYPE_HNSW;
        case GV_NS_INDEX_IVFPQ:  return GV_INDEX_TYPE_IVFPQ;
        case GV_NS_INDEX_SPARSE: return GV_INDEX_TYPE_SPARSE;
        default:                 return GV_INDEX_TYPE_HNSW;
    }
}

/* Convert database index type to namespace index type.
 * Reserved for future use when reading existing database configurations. */
static GV_NSIndexType db_index_to_ns_index(GV_IndexType db_type) __attribute__((unused));
static GV_NSIndexType db_index_to_ns_index(GV_IndexType db_type) {
    switch (db_type) {
        case GV_INDEX_TYPE_KDTREE: return GV_NS_INDEX_KDTREE;
        case GV_INDEX_TYPE_HNSW:   return GV_NS_INDEX_HNSW;
        case GV_INDEX_TYPE_IVFPQ:  return GV_NS_INDEX_IVFPQ;
        case GV_INDEX_TYPE_SPARSE: return GV_NS_INDEX_SPARSE;
        default:                   return GV_NS_INDEX_HNSW;
    }
}

/**
 * @brief Build manifest filepath from database filepath.
 */
static char *build_manifest_path(const char *db_filepath) {
    if (!db_filepath) return NULL;
    size_t len = strlen(db_filepath) + 20;  /* Extra space for .manifest.json */
    char *path = malloc(len);
    if (!path) return NULL;

    /* Replace .gvdb with .manifest.json */
    const char *ext = strrchr(db_filepath, '.');
    if (ext) {
        size_t base_len = ext - db_filepath;
        snprintf(path, len, "%.*s.manifest.json", (int)base_len, db_filepath);
    } else {
        snprintf(path, len, "%s.manifest.json", db_filepath);
    }
    return path;
}

/**
 * @brief Write namespace manifest file.
 * Format: {"dimension":N,"index_type":T,"max_vectors":M,"max_memory":B}
 */
static int write_manifest(const char *db_filepath, size_t dimension,
                          GV_NSIndexType index_type, size_t max_vectors,
                          size_t max_memory_bytes) {
    char *manifest_path = build_manifest_path(db_filepath);
    if (!manifest_path) return -1;

    FILE *fp = fopen(manifest_path, "w");
    if (!fp) {
        free(manifest_path);
        return -1;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"dimension\": %zu,\n", dimension);
    fprintf(fp, "  \"index_type\": %d,\n", (int)index_type);
    fprintf(fp, "  \"max_vectors\": %zu,\n", max_vectors);
    fprintf(fp, "  \"max_memory_bytes\": %zu\n", max_memory_bytes);
    fprintf(fp, "}\n");

    fclose(fp);
    free(manifest_path);
    return 0;
}

/**
 * @brief Read namespace manifest file.
 * Returns 0 on success, -1 on failure.
 */
static int read_manifest(const char *db_filepath, size_t *dimension,
                         GV_NSIndexType *index_type, size_t *max_vectors,
                         size_t *max_memory_bytes) {
    char *manifest_path = build_manifest_path(db_filepath);
    if (!manifest_path) return -1;

    FILE *fp = fopen(manifest_path, "r");
    if (!fp) {
        free(manifest_path);
        return -1;
    }

    /* Read entire file */
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char *content = malloc(fsize + 1);
    if (!content) {
        fclose(fp);
        free(manifest_path);
        return -1;
    }

    fread(content, 1, fsize, fp);
    content[fsize] = '\0';
    fclose(fp);
    free(manifest_path);

    /* Simple JSON parsing */
    *dimension = 128;  /* Default */
    *index_type = GV_NS_INDEX_HNSW;
    *max_vectors = 0;
    *max_memory_bytes = 0;

    char *p;
    if ((p = strstr(content, "\"dimension\"")) != NULL) {
        p = strchr(p, ':');
        if (p) *dimension = (size_t)strtoul(p + 1, NULL, 10);
    }
    if ((p = strstr(content, "\"index_type\"")) != NULL) {
        p = strchr(p, ':');
        if (p) *index_type = (GV_NSIndexType)atoi(p + 1);
    }
    if ((p = strstr(content, "\"max_vectors\"")) != NULL) {
        p = strchr(p, ':');
        if (p) *max_vectors = (size_t)strtoul(p + 1, NULL, 10);
    }
    if ((p = strstr(content, "\"max_memory_bytes\"")) != NULL) {
        p = strchr(p, ':');
        if (p) *max_memory_bytes = (size_t)strtoul(p + 1, NULL, 10);
    }

    free(content);
    return 0;
}

/* ============================================================================
 * Namespace Operations
 * ============================================================================ */

GV_Namespace *gv_namespace_create(GV_NamespaceManager *mgr, const GV_NamespaceConfig *config) {
    if (!mgr || !config || !config->name || config->dimension == 0) {
        return NULL;
    }

    if (strlen(config->name) >= MAX_NAMESPACE_NAME) {
        return NULL;
    }

    pthread_rwlock_wrlock(&mgr->rwlock);

    /* Check if namespace already exists */
    if (find_namespace(mgr, config->name)) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return NULL;
    }

    /* Check max namespaces */
    if (mgr->namespace_count >= MAX_NAMESPACES) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return NULL;
    }

    /* Create namespace */
    GV_Namespace *ns = calloc(1, sizeof(GV_Namespace));
    if (!ns) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return NULL;
    }

    strncpy(ns->name, config->name, MAX_NAMESPACE_NAME - 1);
    ns->max_vectors = config->max_vectors;
    ns->max_memory_bytes = config->max_memory_bytes;
    ns->created_at = (uint64_t)time(NULL);
    ns->last_modified = ns->created_at;

    if (pthread_mutex_init(&ns->mutex, NULL) != 0) {
        free(ns);
        pthread_rwlock_unlock(&mgr->rwlock);
        return NULL;
    }

    /* Build filepath */
    ns->filepath = build_filepath(mgr->base_path, config->name);

    /* Create underlying database */
    GV_IndexType index_type = ns_index_to_db_index(config->index_type);
    ns->db = gv_db_open(ns->filepath, config->dimension, index_type);
    if (!ns->db) {
        pthread_mutex_destroy(&ns->mutex);
        free(ns->filepath);
        free(ns);
        pthread_rwlock_unlock(&mgr->rwlock);
        return NULL;
    }

    /* Write manifest file for persistence */
    write_manifest(ns->filepath, config->dimension, config->index_type,
                   config->max_vectors, config->max_memory_bytes);

    /* Add to manager */
    mgr->namespaces[mgr->namespace_count++] = ns;

    pthread_rwlock_unlock(&mgr->rwlock);
    return ns;
}

GV_Namespace *gv_namespace_get(GV_NamespaceManager *mgr, const char *name) {
    if (!mgr || !name) return NULL;

    pthread_rwlock_rdlock(&mgr->rwlock);
    GV_Namespace *ns = find_namespace(mgr, name);
    pthread_rwlock_unlock(&mgr->rwlock);

    return ns;
}

int gv_namespace_delete(GV_NamespaceManager *mgr, const char *name) {
    if (!mgr || !name) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    /* Find namespace */
    size_t index = (size_t)-1;
    for (size_t i = 0; i < mgr->namespace_count; i++) {
        if (mgr->namespaces[i] && strcmp(mgr->namespaces[i]->name, name) == 0) {
            index = i;
            break;
        }
    }

    if (index == (size_t)-1) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    GV_Namespace *ns = mgr->namespaces[index];

    /* Remove from array */
    for (size_t i = index; i < mgr->namespace_count - 1; i++) {
        mgr->namespaces[i] = mgr->namespaces[i + 1];
    }
    mgr->namespaces[--mgr->namespace_count] = NULL;

    pthread_rwlock_unlock(&mgr->rwlock);

    /* Delete file if exists */
    if (ns->filepath) {
        remove(ns->filepath);
        /* Also remove WAL file */
        char wal_path[512];
        snprintf(wal_path, sizeof(wal_path), "%s.wal", ns->filepath);
        remove(wal_path);
    }

    /* Cleanup */
    pthread_mutex_destroy(&ns->mutex);
    if (ns->db) {
        gv_db_close(ns->db);
    }
    free(ns->filepath);
    free(ns);

    return 0;
}

int gv_namespace_list(GV_NamespaceManager *mgr, char ***names, size_t *count) {
    if (!mgr || !names || !count) return -1;

    pthread_rwlock_rdlock(&mgr->rwlock);

    *count = mgr->namespace_count;
    if (*count == 0) {
        *names = NULL;
        pthread_rwlock_unlock(&mgr->rwlock);
        return 0;
    }

    *names = malloc(*count * sizeof(char *));
    if (!*names) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    for (size_t i = 0; i < *count; i++) {
        (*names)[i] = strdup(mgr->namespaces[i]->name);
    }

    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

int gv_namespace_get_info(const GV_Namespace *ns, GV_NamespaceInfo *info) {
    if (!ns || !info) return -1;

    memset(info, 0, sizeof(*info));

    pthread_mutex_lock((pthread_mutex_t *)&ns->mutex);

    info->name = strdup(ns->name);
    info->dimension = ns->db ? ns->db->dimension : 0;
    info->vector_count = ns->db ? ns->db->count : 0;
    info->memory_bytes = ns->db ? gv_db_get_memory_usage(ns->db) : 0;
    info->created_at = ns->created_at;
    info->last_modified = ns->last_modified;

    /* Determine index type */
    if (ns->db) {
        switch (ns->db->index_type) {
            case GV_INDEX_TYPE_KDTREE: info->index_type = GV_NS_INDEX_KDTREE; break;
            case GV_INDEX_TYPE_HNSW:   info->index_type = GV_NS_INDEX_HNSW; break;
            case GV_INDEX_TYPE_IVFPQ:  info->index_type = GV_NS_INDEX_IVFPQ; break;
            case GV_INDEX_TYPE_SPARSE: info->index_type = GV_NS_INDEX_SPARSE; break;
            default:                   info->index_type = GV_NS_INDEX_HNSW; break;
        }
    }

    pthread_mutex_unlock((pthread_mutex_t *)&ns->mutex);
    return 0;
}

void gv_namespace_free_info(GV_NamespaceInfo *info) {
    if (!info) return;
    free(info->name);
    memset(info, 0, sizeof(*info));
}

int gv_namespace_exists(GV_NamespaceManager *mgr, const char *name) {
    if (!mgr || !name) return -1;

    pthread_rwlock_rdlock(&mgr->rwlock);
    int exists = find_namespace(mgr, name) != NULL ? 1 : 0;
    pthread_rwlock_unlock(&mgr->rwlock);

    return exists;
}

/* ============================================================================
 * Vector Operations within Namespace
 * ============================================================================ */

int gv_namespace_add_vector(GV_Namespace *ns, const float *data, size_t dimension) {
    if (!ns || !ns->db || !data) return -1;

    pthread_mutex_lock(&ns->mutex);

    /* Check limits */
    if (ns->max_vectors > 0 && ns->db->count >= ns->max_vectors) {
        pthread_mutex_unlock(&ns->mutex);
        return -1;
    }

    if (ns->max_memory_bytes > 0 && gv_db_get_memory_usage(ns->db) >= ns->max_memory_bytes) {
        pthread_mutex_unlock(&ns->mutex);
        return -1;
    }

    int result = gv_db_add_vector(ns->db, data, dimension);
    if (result == 0) {
        ns->last_modified = (uint64_t)time(NULL);
    }

    pthread_mutex_unlock(&ns->mutex);
    return result;
}

int gv_namespace_add_vector_with_metadata(GV_Namespace *ns, const float *data, size_t dimension,
                                           const char *const *keys, const char *const *values,
                                           size_t meta_count) {
    if (!ns || !ns->db || !data) return -1;

    pthread_mutex_lock(&ns->mutex);

    /* Check limits */
    if (ns->max_vectors > 0 && ns->db->count >= ns->max_vectors) {
        pthread_mutex_unlock(&ns->mutex);
        return -1;
    }

    int result = gv_db_add_vector_with_rich_metadata(ns->db, data, dimension, keys, values, meta_count);
    if (result == 0) {
        ns->last_modified = (uint64_t)time(NULL);
    }

    pthread_mutex_unlock(&ns->mutex);
    return result;
}

int gv_namespace_search(const GV_Namespace *ns, const float *query, size_t k,
                        GV_SearchResult *results, GV_DistanceType distance_type) {
    if (!ns || !ns->db || !query || !results) return -1;

    pthread_mutex_lock((pthread_mutex_t *)&ns->mutex);
    int found = gv_db_search(ns->db, query, k, results, distance_type);
    pthread_mutex_unlock((pthread_mutex_t *)&ns->mutex);

    return found;
}

int gv_namespace_search_filtered(const GV_Namespace *ns, const float *query, size_t k,
                                  GV_SearchResult *results, GV_DistanceType distance_type,
                                  const char *filter_key, const char *filter_value) {
    if (!ns || !ns->db || !query || !results) return -1;

    pthread_mutex_lock((pthread_mutex_t *)&ns->mutex);
    int found = gv_db_search_filtered(ns->db, query, k, results, distance_type, filter_key, filter_value);
    pthread_mutex_unlock((pthread_mutex_t *)&ns->mutex);

    return found;
}

int gv_namespace_delete_vector(GV_Namespace *ns, size_t vector_index) {
    if (!ns || !ns->db) return -1;

    pthread_mutex_lock(&ns->mutex);
    int result = gv_db_delete_vector_by_index(ns->db, vector_index);
    if (result == 0) {
        ns->last_modified = (uint64_t)time(NULL);
    }
    pthread_mutex_unlock(&ns->mutex);

    return result;
}

size_t gv_namespace_count(const GV_Namespace *ns) {
    if (!ns || !ns->db) return 0;

    pthread_mutex_lock((pthread_mutex_t *)&ns->mutex);
    size_t count = ns->db->count;
    pthread_mutex_unlock((pthread_mutex_t *)&ns->mutex);

    return count;
}

/* ============================================================================
 * Persistence
 * ============================================================================ */

int gv_namespace_save(GV_Namespace *ns) {
    if (!ns || !ns->db) return -1;

    pthread_mutex_lock(&ns->mutex);
    int result = gv_db_save(ns->db, ns->filepath);
    pthread_mutex_unlock(&ns->mutex);

    return result;
}

int gv_namespace_manager_save_all(GV_NamespaceManager *mgr) {
    if (!mgr) return -1;

    pthread_rwlock_rdlock(&mgr->rwlock);

    int errors = 0;
    for (size_t i = 0; i < mgr->namespace_count; i++) {
        if (gv_namespace_save(mgr->namespaces[i]) != 0) {
            errors++;
        }
    }

    pthread_rwlock_unlock(&mgr->rwlock);
    return errors > 0 ? -1 : 0;
}

int gv_namespace_manager_load_all(GV_NamespaceManager *mgr) {
    if (!mgr || !mgr->base_path) return -1;

    DIR *dir = opendir(mgr->base_path);
    if (!dir) return -1;

    int loaded = 0;
    struct dirent *entry;

    while ((entry = readdir(dir)) != NULL) {
        /* Check for .gvdb files */
        const char *ext = strrchr(entry->d_name, '.');
        if (!ext || strcmp(ext, ".gvdb") != 0) {
            continue;
        }

        /* Extract namespace name */
        char name[MAX_NAMESPACE_NAME];
        size_t name_len = ext - entry->d_name;
        if (name_len >= MAX_NAMESPACE_NAME) continue;

        strncpy(name, entry->d_name, name_len);
        name[name_len] = '\0';

        /* Build full path */
        char filepath[512];
        snprintf(filepath, sizeof(filepath), "%s/%s", mgr->base_path, entry->d_name);

        /* Read namespace config from manifest file */
        GV_NamespaceConfig config;
        gv_namespace_config_init(&config);
        config.name = name;

        size_t dimension = 128;
        GV_NSIndexType index_type = GV_NS_INDEX_HNSW;
        size_t max_vectors = 0;
        size_t max_memory_bytes = 0;

        /* Try to read from manifest file */
        if (read_manifest(filepath, &dimension, &index_type, &max_vectors, &max_memory_bytes) == 0) {
            config.dimension = dimension;
            config.index_type = index_type;
            config.max_vectors = max_vectors;
            config.max_memory_bytes = max_memory_bytes;
        } else {
            /* Fallback: try to read dimension from database file header */
            FILE *db_fp = fopen(filepath, "rb");
            if (db_fp) {
                /* Skip magic bytes and version, read dimension */
                fseek(db_fp, 8, SEEK_SET);  /* After magic(4) + version(4) */
                uint32_t dim_from_file = 0;
                if (fread(&dim_from_file, sizeof(uint32_t), 1, db_fp) == 1) {
                    if (dim_from_file > 0 && dim_from_file <= 65536) {
                        config.dimension = dim_from_file;
                    }
                }
                fclose(db_fp);
            }
        }

        if (gv_namespace_create(mgr, &config)) {
            loaded++;
        }
    }

    closedir(dir);
    return loaded;
}

/* ============================================================================
 * Utility
 * ============================================================================ */

GV_Database *gv_namespace_get_db(GV_Namespace *ns) {
    if (!ns) return NULL;
    return ns->db;
}
