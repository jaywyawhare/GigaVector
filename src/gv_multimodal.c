/**
 * @file gv_multimodal.c
 * @brief Multimodal media storage implementation.
 *
 * Content-addressable blob storage on disk with an in-memory hash table
 * mapping vector indices to media metadata entries.  Thread-safe via
 * pthread_rwlock_t.
 */

#include "gigavector/gv_multimodal.h"
#include "gigavector/gv_auth.h"     /* For gv_auth_sha256, gv_auth_to_hex */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>
#include <pthread.h>

/*  Constants  */

#define MEDIA_INDEX_MAGIC "GVMED"
#define MEDIA_INDEX_MAGIC_LEN 5
#define MEDIA_INDEX_VERSION 1

#define HASH_TABLE_INITIAL_CAPACITY 256
#define FILE_BUFFER_SIZE (64 * 1024)

/*  Internal Structures  */

/**
 * @brief Internal entry stored in the hash table.
 */
typedef struct GV_MediaNode {
    GV_MediaEntry entry;            /**< Metadata entry. */
    struct GV_MediaNode *next;      /**< Next node in the chain. */
} GV_MediaNode;

/**
 * @brief Full definition of the opaque GV_MediaStore.
 */
struct GV_MediaStore {
    GV_MediaConfig config;          /**< Copy of caller configuration. */
    char *storage_dir;              /**< Owned copy of storage directory path. */

    /* Hash table: vector_index -> GV_MediaNode */
    GV_MediaNode **buckets;         /**< Array of bucket head pointers. */
    size_t num_buckets;             /**< Number of buckets. */
    size_t count;                   /**< Number of entries. */

    pthread_rwlock_t lock;          /**< Read-write lock for thread safety. */
};

/*  Configuration  */

static const GV_MediaConfig DEFAULT_CONFIG = {
    .storage_dir = NULL,
    .max_blob_size_mb = 100,
    .deduplicate = 1,
    .compress_blobs = 0
};

void gv_media_config_init(GV_MediaConfig *config) {
    if (!config) return;
    *config = DEFAULT_CONFIG;
}

/*  Internal Helpers  */

/**
 * @brief Compute bucket index from vector_index.
 */
static size_t bucket_index(size_t vector_index, size_t num_buckets) {
    /* Simple hash mixing */
    size_t h = vector_index;
    h ^= h >> 16;
    h *= 0x45d9f3b;
    h ^= h >> 16;
    return h % num_buckets;
}

/**
 * @brief Look up an entry by vector_index (caller must hold at least read lock).
 */
static GV_MediaNode *find_node(const GV_MediaStore *store, size_t vector_index) {
    size_t idx = bucket_index(vector_index, store->num_buckets);
    GV_MediaNode *node = store->buckets[idx];
    while (node) {
        if (node->entry.vector_index == vector_index) {
            return node;
        }
        node = node->next;
    }
    return NULL;
}

/**
 * @brief Free a node and its owned strings.
 */
static void free_node(GV_MediaNode *node) {
    if (!node) return;
    free(node->entry.filename);
    free(node->entry.mime_type);
    free(node);
}

/**
 * @brief Count how many entries reference a given hash (caller must hold lock).
 */
static size_t hash_ref_count(const GV_MediaStore *store, const char *hash) {
    size_t refs = 0;
    for (size_t i = 0; i < store->num_buckets; i++) {
        GV_MediaNode *node = store->buckets[i];
        while (node) {
            if (strcmp(node->entry.hash, hash) == 0) {
                refs++;
            }
            node = node->next;
        }
    }
    return refs;
}

/**
 * @brief Build the on-disk blob path: storage_dir/{hash}.blob
 */
static int build_blob_path(const GV_MediaStore *store, const char *hash,
                            char *path, size_t path_size) {
    int n = snprintf(path, path_size, "%s/%s.blob", store->storage_dir, hash);
    if (n < 0 || (size_t)n >= path_size) return -1;
    return 0;
}

/**
 * @brief Compute SHA-256 of data and produce a 64-char hex string.
 */
static int compute_sha256_hex(const void *data, size_t len, char *hex_out) {
    unsigned char raw[32];
    if (gv_auth_sha256(data, len, raw) != 0) return -1;
    gv_auth_to_hex(raw, 32, hex_out);
    return 0;
}

/**
 * @brief Ensure directory exists, creating it if necessary.
 */
static int ensure_directory(const char *dir) {
    struct stat st;
    if (stat(dir, &st) == 0) {
        return S_ISDIR(st.st_mode) ? 0 : -1;
    }
    /* Attempt to create (mode 0755) */
    return mkdir(dir, 0755);
}

/**
 * @brief Extract the basename from a file path.
 */
static const char *path_basename(const char *path) {
    const char *slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

/**
 * @brief Insert a node into the hash table (caller must hold write lock).
 */
static int insert_node(GV_MediaStore *store, GV_MediaNode *node) {
    size_t idx = bucket_index(node->entry.vector_index, store->num_buckets);
    node->next = store->buckets[idx];
    store->buckets[idx] = node;
    store->count++;
    return 0;
}

/**
 * @brief Write raw data to a file on disk.
 */
static int write_blob_file(const char *path, const void *data, size_t size) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;

    size_t written = fwrite(data, 1, size, fp);
    fclose(fp);

    return (written == size) ? 0 : -1;
}

/**
 * @brief Read an entire file into a malloc'd buffer.
 *
 * @param path     File path.
 * @param out_size Set to file size on success.
 * @return Allocated buffer (caller frees), or NULL on error.
 */
static void *read_entire_file(const char *path, size_t *out_size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;

    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    if (sz < 0) {
        fclose(fp);
        return NULL;
    }
    fseek(fp, 0, SEEK_SET);

    void *buf = malloc((size_t)sz);
    if (!buf) {
        fclose(fp);
        return NULL;
    }

    size_t nread = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);

    if (nread != (size_t)sz) {
        free(buf);
        return NULL;
    }

    *out_size = (size_t)sz;
    return buf;
}

/*  Store Lifecycle  */

GV_MediaStore *gv_media_create(const GV_MediaConfig *config) {
    if (!config || !config->storage_dir) return NULL;

    /* Ensure storage directory exists */
    if (ensure_directory(config->storage_dir) != 0) return NULL;

    GV_MediaStore *store = calloc(1, sizeof(GV_MediaStore));
    if (!store) return NULL;

    store->config = *config;
    store->storage_dir = strdup(config->storage_dir);
    if (!store->storage_dir) {
        free(store);
        return NULL;
    }

    /* Allocate hash table */
    store->num_buckets = HASH_TABLE_INITIAL_CAPACITY;
    store->buckets = calloc(store->num_buckets, sizeof(GV_MediaNode *));
    if (!store->buckets) {
        free(store->storage_dir);
        free(store);
        return NULL;
    }

    store->count = 0;

    if (pthread_rwlock_init(&store->lock, NULL) != 0) {
        free(store->buckets);
        free(store->storage_dir);
        free(store);
        return NULL;
    }

    return store;
}

void gv_media_destroy(GV_MediaStore *store) {
    if (!store) return;

    pthread_rwlock_wrlock(&store->lock);

    /* Free all nodes */
    for (size_t i = 0; i < store->num_buckets; i++) {
        GV_MediaNode *node = store->buckets[i];
        while (node) {
            GV_MediaNode *next = node->next;
            free_node(node);
            node = next;
        }
    }

    free(store->buckets);
    free(store->storage_dir);

    pthread_rwlock_unlock(&store->lock);
    pthread_rwlock_destroy(&store->lock);

    free(store);
}

/*  Store Operations  */

int gv_media_store_blob(GV_MediaStore *store, size_t vector_index,
                         GV_MediaType type, const void *data, size_t data_size,
                         const char *filename, const char *mime_type) {
    if (!store || !data || data_size == 0) return -1;

    /* Enforce size limit */
    size_t max_bytes = store->config.max_blob_size_mb * 1024UL * 1024UL;
    if (data_size > max_bytes) return -1;

    /* Compute SHA-256 hash */
    char hash_hex[65];
    if (compute_sha256_hex(data, data_size, hash_hex) != 0) return -1;

    pthread_rwlock_wrlock(&store->lock);

    /* Check if this vector_index already has an entry */
    if (find_node(store, vector_index)) {
        pthread_rwlock_unlock(&store->lock);
        return -1;
    }

    /* Build blob path and write to disk (skip if dedup and file exists) */
    char blob_path[1024];
    if (build_blob_path(store, hash_hex, blob_path, sizeof(blob_path)) != 0) {
        pthread_rwlock_unlock(&store->lock);
        return -1;
    }

    struct stat st;
    int file_exists = (stat(blob_path, &st) == 0);

    if (!file_exists || !store->config.deduplicate) {
        if (write_blob_file(blob_path, data, data_size) != 0) {
            pthread_rwlock_unlock(&store->lock);
            return -1;
        }
    }

    /* Create metadata node */
    GV_MediaNode *node = calloc(1, sizeof(GV_MediaNode));
    if (!node) {
        pthread_rwlock_unlock(&store->lock);
        return -1;
    }

    node->entry.vector_index = vector_index;
    node->entry.type = type;
    node->entry.filename = filename ? strdup(filename) : NULL;
    node->entry.file_size = data_size;
    memcpy(node->entry.hash, hash_hex, 65);
    node->entry.created_at = (uint64_t)time(NULL);
    node->entry.mime_type = mime_type ? strdup(mime_type) : NULL;

    insert_node(store, node);

    pthread_rwlock_unlock(&store->lock);
    return 0;
}

int gv_media_store_file(GV_MediaStore *store, size_t vector_index,
                         GV_MediaType type, const char *file_path) {
    if (!store || !file_path) return -1;

    /* Read the source file */
    size_t file_size = 0;
    void *data = read_entire_file(file_path, &file_size);
    if (!data) return -1;

    /* Extract filename from path */
    const char *basename = path_basename(file_path);

    int rc = gv_media_store_blob(store, vector_index, type,
                                  data, file_size, basename, NULL);
    free(data);
    return rc;
}

int gv_media_retrieve(const GV_MediaStore *store, size_t vector_index,
                       void *buffer, size_t buffer_size, size_t *actual_size) {
    if (!store || !actual_size) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&store->lock);

    GV_MediaNode *node = find_node(store, vector_index);
    if (!node) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&store->lock);
        return -1;
    }

    /* Build path from hash */
    char blob_path[1024];
    if (build_blob_path(store, node->entry.hash, blob_path, sizeof(blob_path)) != 0) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&store->lock);
        return -1;
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&store->lock);

    /* Read file */
    size_t file_size = 0;
    void *data = read_entire_file(blob_path, &file_size);
    if (!data) return -1;

    *actual_size = file_size;

    if (buffer && buffer_size >= file_size) {
        memcpy(buffer, data, file_size);
    }

    free(data);
    return 0;
}

int gv_media_get_path(const GV_MediaStore *store, size_t vector_index,
                       char *path, size_t path_size) {
    if (!store || !path || path_size == 0) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&store->lock);

    GV_MediaNode *node = find_node(store, vector_index);
    if (!node) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&store->lock);
        return -1;
    }

    int rc = build_blob_path(store, node->entry.hash, path, path_size);

    pthread_rwlock_unlock((pthread_rwlock_t *)&store->lock);
    return rc;
}

int gv_media_get_info(const GV_MediaStore *store, size_t vector_index,
                       GV_MediaEntry *entry) {
    if (!store || !entry) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&store->lock);

    GV_MediaNode *node = find_node(store, vector_index);
    if (!node) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&store->lock);
        return -1;
    }

    /* Copy the entry; allocate fresh strings for the caller */
    entry->vector_index = node->entry.vector_index;
    entry->type = node->entry.type;
    entry->filename = node->entry.filename ? strdup(node->entry.filename) : NULL;
    entry->file_size = node->entry.file_size;
    memcpy(entry->hash, node->entry.hash, 65);
    entry->created_at = node->entry.created_at;
    entry->mime_type = node->entry.mime_type ? strdup(node->entry.mime_type) : NULL;

    pthread_rwlock_unlock((pthread_rwlock_t *)&store->lock);
    return 0;
}

int gv_media_delete(GV_MediaStore *store, size_t vector_index) {
    if (!store) return -1;

    pthread_rwlock_wrlock(&store->lock);

    size_t idx = bucket_index(vector_index, store->num_buckets);
    GV_MediaNode *prev = NULL;
    GV_MediaNode *node = store->buckets[idx];

    while (node) {
        if (node->entry.vector_index == vector_index) {
            /* Unlink from chain */
            if (prev) {
                prev->next = node->next;
            } else {
                store->buckets[idx] = node->next;
            }
            store->count--;

            /* Check if we should delete the file (no other refs to same hash) */
            char hash_copy[65];
            memcpy(hash_copy, node->entry.hash, 65);

            free_node(node);

            /* Remove blob file if no other entries reference this hash */
            if (hash_ref_count(store, hash_copy) == 0) {
                char blob_path[1024];
                if (build_blob_path(store, hash_copy, blob_path, sizeof(blob_path)) == 0) {
                    remove(blob_path);
                }
            }

            pthread_rwlock_unlock(&store->lock);
            return 0;
        }
        prev = node;
        node = node->next;
    }

    pthread_rwlock_unlock(&store->lock);
    return -1;
}

/*  Query Operations  */

int gv_media_exists(const GV_MediaStore *store, size_t vector_index) {
    if (!store) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&store->lock);

    GV_MediaNode *node = find_node(store, vector_index);
    int result = node ? 1 : 0;

    pthread_rwlock_unlock((pthread_rwlock_t *)&store->lock);
    return result;
}

size_t gv_media_count(const GV_MediaStore *store) {
    if (!store) return 0;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&store->lock);
    size_t count = store->count;
    pthread_rwlock_unlock((pthread_rwlock_t *)&store->lock);

    return count;
}

size_t gv_media_total_size(const GV_MediaStore *store) {
    if (!store) return 0;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&store->lock);

    size_t total = 0;
    for (size_t i = 0; i < store->num_buckets; i++) {
        GV_MediaNode *node = store->buckets[i];
        while (node) {
            total += node->entry.file_size;
            node = node->next;
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&store->lock);
    return total;
}

/*  Index Persistence  */

int gv_media_save_index(const GV_MediaStore *store, const char *path) {
    if (!store || !path) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&store->lock);

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&store->lock);
        return -1;
    }

    /* Write magic and version */
    fwrite(MEDIA_INDEX_MAGIC, 1, MEDIA_INDEX_MAGIC_LEN, fp);
    uint32_t version = MEDIA_INDEX_VERSION;
    fwrite(&version, sizeof(version), 1, fp);

    /* Write entry count */
    uint64_t count = (uint64_t)store->count;
    fwrite(&count, sizeof(count), 1, fp);

    /* Write each entry */
    for (size_t i = 0; i < store->num_buckets; i++) {
        GV_MediaNode *node = store->buckets[i];
        while (node) {
            const GV_MediaEntry *e = &node->entry;

            /* vector_index */
            uint64_t vi = (uint64_t)e->vector_index;
            fwrite(&vi, sizeof(vi), 1, fp);

            /* type */
            uint32_t t = (uint32_t)e->type;
            fwrite(&t, sizeof(t), 1, fp);

            /* hash (64 bytes, no null) */
            fwrite(e->hash, 1, 64, fp);

            /* filename_len + filename */
            uint32_t fn_len = e->filename ? (uint32_t)strlen(e->filename) : 0;
            fwrite(&fn_len, sizeof(fn_len), 1, fp);
            if (fn_len > 0) {
                fwrite(e->filename, 1, fn_len, fp);
            }

            /* file_size */
            uint64_t fs = (uint64_t)e->file_size;
            fwrite(&fs, sizeof(fs), 1, fp);

            /* mime_type_len + mime_type */
            uint32_t mt_len = e->mime_type ? (uint32_t)strlen(e->mime_type) : 0;
            fwrite(&mt_len, sizeof(mt_len), 1, fp);
            if (mt_len > 0) {
                fwrite(e->mime_type, 1, mt_len, fp);
            }

            /* created_at */
            fwrite(&e->created_at, sizeof(e->created_at), 1, fp);

            node = node->next;
        }
    }

    fclose(fp);
    pthread_rwlock_unlock((pthread_rwlock_t *)&store->lock);
    return 0;
}

GV_MediaStore *gv_media_load_index(const char *index_path,
                                    const char *storage_dir) {
    if (!index_path || !storage_dir) return NULL;

    FILE *fp = fopen(index_path, "rb");
    if (!fp) return NULL;

    /* Read and verify magic */
    char magic[MEDIA_INDEX_MAGIC_LEN];
    if (fread(magic, 1, MEDIA_INDEX_MAGIC_LEN, fp) != MEDIA_INDEX_MAGIC_LEN ||
        memcmp(magic, MEDIA_INDEX_MAGIC, MEDIA_INDEX_MAGIC_LEN) != 0) {
        fclose(fp);
        return NULL;
    }

    /* Read and verify version */
    uint32_t version;
    if (fread(&version, sizeof(version), 1, fp) != 1 ||
        version != MEDIA_INDEX_VERSION) {
        fclose(fp);
        return NULL;
    }

    /* Read count */
    uint64_t count;
    if (fread(&count, sizeof(count), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }

    /* Create store with the given storage_dir */
    GV_MediaConfig config;
    gv_media_config_init(&config);
    config.storage_dir = storage_dir;

    GV_MediaStore *store = gv_media_create(&config);
    if (!store) {
        fclose(fp);
        return NULL;
    }

    /* Read entries */
    for (uint64_t i = 0; i < count; i++) {
        GV_MediaNode *node = calloc(1, sizeof(GV_MediaNode));
        if (!node) {
            fclose(fp);
            gv_media_destroy(store);
            return NULL;
        }

        /* vector_index */
        uint64_t vi;
        if (fread(&vi, sizeof(vi), 1, fp) != 1) goto load_error;
        node->entry.vector_index = (size_t)vi;

        /* type */
        uint32_t t;
        if (fread(&t, sizeof(t), 1, fp) != 1) goto load_error;
        node->entry.type = (GV_MediaType)t;

        /* hash */
        if (fread(node->entry.hash, 1, 64, fp) != 64) goto load_error;
        node->entry.hash[64] = '\0';

        /* filename */
        uint32_t fn_len;
        if (fread(&fn_len, sizeof(fn_len), 1, fp) != 1) goto load_error;
        if (fn_len > 0) {
            node->entry.filename = malloc(fn_len + 1);
            if (!node->entry.filename) goto load_error;
            if (fread(node->entry.filename, 1, fn_len, fp) != fn_len) goto load_error;
            node->entry.filename[fn_len] = '\0';
        }

        /* file_size */
        uint64_t fs;
        if (fread(&fs, sizeof(fs), 1, fp) != 1) goto load_error;
        node->entry.file_size = (size_t)fs;

        /* mime_type */
        uint32_t mt_len;
        if (fread(&mt_len, sizeof(mt_len), 1, fp) != 1) goto load_error;
        if (mt_len > 0) {
            node->entry.mime_type = malloc(mt_len + 1);
            if (!node->entry.mime_type) goto load_error;
            if (fread(node->entry.mime_type, 1, mt_len, fp) != mt_len) goto load_error;
            node->entry.mime_type[mt_len] = '\0';
        }

        /* created_at */
        if (fread(&node->entry.created_at, sizeof(node->entry.created_at), 1, fp) != 1) {
            goto load_error;
        }

        insert_node(store, node);
        continue;

    load_error:
        free_node(node);
        fclose(fp);
        gv_media_destroy(store);
        return NULL;
    }

    fclose(fp);
    return store;
}
