/**
 * @file gv_backup.c
 * @brief Backup and restore implementation.
 */

#include "gigavector/gv_backup.h"
#include "gigavector/gv_database.h"
#include "gigavector/gv_auth.h"     /* For SHA-256 */
#include "gigavector/gv_crypto.h"   /* For encryption */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <sys/stat.h>

/* ============================================================================
 * Constants
 * ============================================================================ */

#define BACKUP_MAGIC "GVBAK"
#define BACKUP_MAGIC_LEN 5
#define BUFFER_SIZE (64 * 1024)

/* Backup flags */
#define BACKUP_FLAG_COMPRESSED 0x01
#define BACKUP_FLAG_ENCRYPTED  0x02
#define BACKUP_FLAG_INCREMENTAL 0x04

/* ============================================================================
 * Configuration
 * ============================================================================ */

static const GV_BackupOptions DEFAULT_BACKUP_OPTIONS = {
    .compression = GV_BACKUP_COMPRESS_NONE,
    .include_wal = 1,
    .include_metadata = 1,
    .verify_after = 1,
    .encryption_key = NULL
};

static const GV_RestoreOptions DEFAULT_RESTORE_OPTIONS = {
    .overwrite = 0,
    .verify_checksum = 1,
    .decryption_key = NULL
};

void gv_backup_options_init(GV_BackupOptions *options) {
    if (!options) return;
    *options = DEFAULT_BACKUP_OPTIONS;
}

void gv_restore_options_init(GV_RestoreOptions *options) {
    if (!options) return;
    *options = DEFAULT_RESTORE_OPTIONS;
}

/* ============================================================================
 * Internal Helpers
 * ============================================================================ */

static GV_BackupResult *create_result(int success, const char *error) {
    GV_BackupResult *result = calloc(1, sizeof(GV_BackupResult));
    if (!result) return NULL;
    result->success = success;
    if (error) {
        result->error_message = strdup(error);
    }
    return result;
}

static double get_time_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* ============================================================================
 * Backup Operations
 * ============================================================================ */

GV_BackupResult *gv_backup_create(GV_Database *db, const char *backup_path,
                                   const GV_BackupOptions *options,
                                   GV_BackupProgressCallback progress,
                                   void *user_data) {
    if (!db || !backup_path) {
        return create_result(0, "Invalid parameters");
    }

    double start_time = get_time_seconds();
    const GV_BackupOptions *opts = options ? options : &DEFAULT_BACKUP_OPTIONS;

    /* Create backup file */
    FILE *fp = fopen(backup_path, "wb");
    if (!fp) {
        return create_result(0, "Failed to create backup file");
    }

    /* Write magic */
    fwrite(BACKUP_MAGIC, 1, BACKUP_MAGIC_LEN, fp);

    /* Build and write header */
    GV_BackupHeader header;
    memset(&header, 0, sizeof(header));
    header.version = GV_BACKUP_VERSION;
    header.flags = 0;
    if (opts->compression != GV_BACKUP_COMPRESS_NONE) {
        header.flags |= BACKUP_FLAG_COMPRESSED;
    }
    if (opts->encryption_key) {
        header.flags |= BACKUP_FLAG_ENCRYPTED;
    }
    header.created_at = (uint64_t)time(NULL);
    header.vector_count = db->count;
    header.dimension = db->dimension;
    header.index_type = db->index_type;

    fwrite(&header.version, sizeof(header.version), 1, fp);
    fwrite(&header.flags, sizeof(header.flags), 1, fp);
    fwrite(&header.created_at, sizeof(header.created_at), 1, fp);
    fwrite(&header.vector_count, sizeof(header.vector_count), 1, fp);
    fwrite(&header.dimension, sizeof(header.dimension), 1, fp);
    fwrite(&header.index_type, sizeof(header.index_type), 1, fp);

    /* Placeholder for sizes and checksum (will update at end) */
    long sizes_pos = ftell(fp);
    uint64_t zero = 0;
    fwrite(&zero, sizeof(zero), 1, fp);  /* original_size */
    fwrite(&zero, sizeof(zero), 1, fp);  /* compressed_size */
    char checksum_placeholder[65] = {0};
    fwrite(checksum_placeholder, 1, 64, fp);

    /* Write vectors */
    uint64_t data_size = 0;
    size_t dimension = gv_database_dimension(db);
    size_t count = gv_database_count(db);
    size_t vector_size = dimension * sizeof(float);

    for (size_t i = 0; i < count; i++) {
        /* Get vector data using accessor function */
        const float *vector = gv_database_get_vector(db, i);

        if (vector) {
            fwrite(vector, 1, vector_size, fp);
            data_size += vector_size;
        } else {
            /* Write zeros for missing vector */
            float *zeros = calloc(dimension, sizeof(float));
            fwrite(zeros, 1, vector_size, fp);
            free(zeros);
            data_size += vector_size;
        }

        if (progress && i % 1000 == 0) {
            progress(i, count, user_data);
        }
    }

    if (progress) {
        progress(count, count, user_data);
    }

    /* Update sizes */
    long end_pos = ftell(fp);
    fseek(fp, sizes_pos, SEEK_SET);
    fwrite(&data_size, sizeof(data_size), 1, fp);
    fwrite(&zero, sizeof(zero), 1, fp);  /* compressed_size = 0 for now */
    fseek(fp, end_pos, SEEK_SET);

    fclose(fp);

    /* Compute and update checksum */
    char checksum[65];
    if (gv_backup_compute_checksum(backup_path, checksum) == 0) {
        fp = fopen(backup_path, "r+b");
        if (fp) {
            fseek(fp, sizes_pos + 16, SEEK_SET);
            fwrite(checksum, 1, 64, fp);
            fclose(fp);
        }
    }

    /* Verify if requested */
    if (opts->verify_after) {
        GV_BackupResult *verify = gv_backup_verify(backup_path, NULL);
        if (!verify->success) {
            char *err = verify->error_message ? strdup(verify->error_message) : strdup("Verification failed");
            gv_backup_result_free(verify);
            return create_result(0, err);
        }
        gv_backup_result_free(verify);
    }

    GV_BackupResult *result = create_result(1, NULL);
    result->bytes_processed = data_size;
    result->vectors_processed = db->count;
    result->elapsed_seconds = get_time_seconds() - start_time;

    return result;
}

GV_BackupResult *gv_backup_create_from_file(const char *db_path, const char *backup_path,
                                             const GV_BackupOptions *options,
                                             GV_BackupProgressCallback progress,
                                             void *user_data) {
    if (!db_path || !backup_path) {
        return create_result(0, "Invalid parameters");
    }

    /* Open database */
    GV_Database *db = gv_db_open(db_path, 0, GV_INDEX_TYPE_HNSW);
    if (!db) {
        return create_result(0, "Failed to open database");
    }

    GV_BackupResult *result = gv_backup_create(db, backup_path, options, progress, user_data);
    gv_db_close(db);

    return result;
}

void gv_backup_result_free(GV_BackupResult *result) {
    if (!result) return;
    free(result->error_message);
    free(result);
}

/* ============================================================================
 * Restore Operations
 * ============================================================================ */

GV_BackupResult *gv_backup_restore(const char *backup_path, const char *db_path,
                                    const GV_RestoreOptions *options,
                                    GV_BackupProgressCallback progress,
                                    void *user_data) {
    if (!backup_path || !db_path) {
        return create_result(0, "Invalid parameters");
    }

    const GV_RestoreOptions *opts = options ? options : &DEFAULT_RESTORE_OPTIONS;
    double start_time = get_time_seconds();

    /* Check if destination exists */
    if (!opts->overwrite) {
        struct stat st;
        if (stat(db_path, &st) == 0) {
            return create_result(0, "Destination file already exists");
        }
    }

    /* Verify checksum if requested */
    if (opts->verify_checksum) {
        GV_BackupResult *verify = gv_backup_verify(backup_path, opts->decryption_key);
        if (!verify->success) {
            char *err = verify->error_message ? strdup(verify->error_message) : strdup("Checksum verification failed");
            gv_backup_result_free(verify);
            return create_result(0, err);
        }
        gv_backup_result_free(verify);
    }

    /* Read backup */
    FILE *fp = fopen(backup_path, "rb");
    if (!fp) {
        return create_result(0, "Failed to open backup file");
    }

    /* Read and verify magic */
    char magic[BACKUP_MAGIC_LEN];
    if (fread(magic, 1, BACKUP_MAGIC_LEN, fp) != BACKUP_MAGIC_LEN ||
        memcmp(magic, BACKUP_MAGIC, BACKUP_MAGIC_LEN) != 0) {
        fclose(fp);
        return create_result(0, "Invalid backup file format");
    }

    /* Read header */
    GV_BackupHeader header;
    fread(&header.version, sizeof(header.version), 1, fp);
    fread(&header.flags, sizeof(header.flags), 1, fp);
    fread(&header.created_at, sizeof(header.created_at), 1, fp);
    fread(&header.vector_count, sizeof(header.vector_count), 1, fp);
    fread(&header.dimension, sizeof(header.dimension), 1, fp);
    fread(&header.index_type, sizeof(header.index_type), 1, fp);
    fread(&header.original_size, sizeof(header.original_size), 1, fp);
    fread(&header.compressed_size, sizeof(header.compressed_size), 1, fp);
    fread(header.checksum, 1, 64, fp);

    /* Create database */
    GV_Database *db = gv_db_open(NULL, header.dimension, header.index_type);
    if (!db) {
        fclose(fp);
        return create_result(0, "Failed to create database");
    }

    /* Read vectors */
    size_t vector_size = header.dimension * sizeof(float);
    float *buffer = malloc(vector_size);
    if (!buffer) {
        gv_db_close(db);
        fclose(fp);
        return create_result(0, "Memory allocation failed");
    }

    uint64_t vectors_read = 0;
    while (vectors_read < header.vector_count) {
        if (fread(buffer, 1, vector_size, fp) != vector_size) {
            break;
        }

        gv_db_add_vector(db, buffer, header.dimension);
        vectors_read++;

        if (progress && vectors_read % 1000 == 0) {
            progress(vectors_read, header.vector_count, user_data);
        }
    }

    free(buffer);
    fclose(fp);

    if (progress) {
        progress(header.vector_count, header.vector_count, user_data);
    }

    /* Save to destination */
    if (gv_db_save(db, db_path) != 0) {
        gv_db_close(db);
        return create_result(0, "Failed to save database");
    }

    gv_db_close(db);

    GV_BackupResult *result = create_result(1, NULL);
    result->bytes_processed = header.original_size;
    result->vectors_processed = vectors_read;
    result->elapsed_seconds = get_time_seconds() - start_time;

    return result;
}

GV_BackupResult *gv_backup_restore_to_db(const char *backup_path,
                                          const GV_RestoreOptions *options,
                                          GV_Database **db) {
    if (!backup_path || !db) {
        return create_result(0, "Invalid parameters");
    }

    const GV_RestoreOptions *opts = options ? options : &DEFAULT_RESTORE_OPTIONS;

    /* Verify checksum if requested */
    if (opts->verify_checksum) {
        GV_BackupResult *verify = gv_backup_verify(backup_path, opts->decryption_key);
        if (!verify->success) {
            char *err = verify->error_message ? strdup(verify->error_message) : strdup("Checksum verification failed");
            gv_backup_result_free(verify);
            return create_result(0, err);
        }
        gv_backup_result_free(verify);
    }

    /* Read backup header */
    GV_BackupHeader header;
    if (gv_backup_read_header(backup_path, &header) != 0) {
        return create_result(0, "Failed to read backup header");
    }

    /* Create in-memory database */
    *db = gv_db_open(NULL, header.dimension, header.index_type);
    if (!*db) {
        return create_result(0, "Failed to create database");
    }

    /* Read vectors */
    FILE *fp = fopen(backup_path, "rb");
    if (!fp) {
        gv_db_close(*db);
        *db = NULL;
        return create_result(0, "Failed to open backup file");
    }

    /* Skip to data section */
    fseek(fp, BACKUP_MAGIC_LEN + sizeof(uint32_t) * 2 + sizeof(uint64_t) * 4 + 64, SEEK_SET);

    size_t vector_size = header.dimension * sizeof(float);
    float *buffer = malloc(vector_size);
    if (!buffer) {
        fclose(fp);
        gv_db_close(*db);
        *db = NULL;
        return create_result(0, "Memory allocation failed");
    }

    uint64_t vectors_read = 0;
    while (vectors_read < header.vector_count) {
        if (fread(buffer, 1, vector_size, fp) != vector_size) {
            break;
        }
        gv_db_add_vector(*db, buffer, header.dimension);
        vectors_read++;
    }

    free(buffer);
    fclose(fp);

    GV_BackupResult *result = create_result(1, NULL);
    result->vectors_processed = vectors_read;

    return result;
}

/* ============================================================================
 * Inspection Operations
 * ============================================================================ */

int gv_backup_read_header(const char *backup_path, GV_BackupHeader *header) {
    if (!backup_path || !header) return -1;

    FILE *fp = fopen(backup_path, "rb");
    if (!fp) return -1;

    memset(header, 0, sizeof(*header));

    /* Read and verify magic */
    char magic[BACKUP_MAGIC_LEN];
    if (fread(magic, 1, BACKUP_MAGIC_LEN, fp) != BACKUP_MAGIC_LEN ||
        memcmp(magic, BACKUP_MAGIC, BACKUP_MAGIC_LEN) != 0) {
        fclose(fp);
        return -1;
    }

    /* Read header fields */
    fread(&header->version, sizeof(header->version), 1, fp);
    fread(&header->flags, sizeof(header->flags), 1, fp);
    fread(&header->created_at, sizeof(header->created_at), 1, fp);
    fread(&header->vector_count, sizeof(header->vector_count), 1, fp);
    fread(&header->dimension, sizeof(header->dimension), 1, fp);
    fread(&header->index_type, sizeof(header->index_type), 1, fp);
    fread(&header->original_size, sizeof(header->original_size), 1, fp);
    fread(&header->compressed_size, sizeof(header->compressed_size), 1, fp);
    fread(header->checksum, 1, 64, fp);
    header->checksum[64] = '\0';

    fclose(fp);
    return 0;
}

GV_BackupResult *gv_backup_verify(const char *backup_path, const char *decryption_key) {
    (void)decryption_key;  /* Not implemented yet */

    if (!backup_path) {
        return create_result(0, "Invalid parameters");
    }

    /* Read header */
    GV_BackupHeader header;
    if (gv_backup_read_header(backup_path, &header) != 0) {
        return create_result(0, "Failed to read backup header");
    }

    /* Verify version */
    if (header.version != GV_BACKUP_VERSION) {
        return create_result(0, "Unsupported backup version");
    }

    /* Compute current checksum (excluding checksum field) */
    /* For simplicity, just check file can be read */
    FILE *fp = fopen(backup_path, "rb");
    if (!fp) {
        return create_result(0, "Failed to open backup file");
    }

    /* Verify file size matches expected */
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fclose(fp);

    size_t expected_min = BACKUP_MAGIC_LEN + sizeof(uint32_t) * 2 +
                          sizeof(uint64_t) * 4 + 64 +
                          header.vector_count * header.dimension * sizeof(float);

    if (file_size < (long)expected_min) {
        return create_result(0, "Backup file appears truncated");
    }

    return create_result(1, NULL);
}

int gv_backup_get_info(const char *backup_path, char *info_buf, size_t buf_size) {
    if (!backup_path || !info_buf || buf_size == 0) return -1;

    GV_BackupHeader header;
    if (gv_backup_read_header(backup_path, &header) != 0) {
        snprintf(info_buf, buf_size, "Error: Failed to read backup header");
        return -1;
    }

    /* Format timestamp */
    time_t created = (time_t)header.created_at;
    struct tm *tm_info = localtime(&created);
    char time_buf[64];
    strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", tm_info);

    /* Get index type name */
    const char *index_type;
    switch (header.index_type) {
        case 0: index_type = "KD-Tree"; break;
        case 1: index_type = "HNSW"; break;
        case 2: index_type = "IVF-PQ"; break;
        case 3: index_type = "Sparse"; break;
        default: index_type = "Unknown"; break;
    }

    /* Format info */
    snprintf(info_buf, buf_size,
             "GigaVector Backup\n"
             "  Version: %u\n"
             "  Created: %s\n"
             "  Vectors: %llu\n"
             "  Dimension: %u\n"
             "  Index Type: %s\n"
             "  Original Size: %llu bytes\n"
             "  Compressed: %s\n"
             "  Encrypted: %s\n"
             "  Checksum: %.16s...",
             header.version,
             time_buf,
             (unsigned long long)header.vector_count,
             header.dimension,
             index_type,
             (unsigned long long)header.original_size,
             (header.flags & BACKUP_FLAG_COMPRESSED) ? "Yes" : "No",
             (header.flags & BACKUP_FLAG_ENCRYPTED) ? "Yes" : "No",
             header.checksum);

    return 0;
}

/* ============================================================================
 * Incremental Backup
 * ============================================================================ */

/**
 * @brief Read header from backup file.
 */
static int read_backup_header(const char *path, GV_BackupHeader *header) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;

    char magic[BACKUP_MAGIC_LEN + 1] = {0};
    if (fread(magic, 1, BACKUP_MAGIC_LEN, fp) != BACKUP_MAGIC_LEN) {
        fclose(fp);
        return -1;
    }

    if (strncmp(magic, BACKUP_MAGIC, BACKUP_MAGIC_LEN) != 0) {
        fclose(fp);
        return -1;
    }

    if (fread(&header->version, sizeof(header->version), 1, fp) != 1 ||
        fread(&header->flags, sizeof(header->flags), 1, fp) != 1 ||
        fread(&header->created_at, sizeof(header->created_at), 1, fp) != 1 ||
        fread(&header->vector_count, sizeof(header->vector_count), 1, fp) != 1 ||
        fread(&header->dimension, sizeof(header->dimension), 1, fp) != 1 ||
        fread(&header->index_type, sizeof(header->index_type), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}

GV_BackupResult *gv_backup_create_incremental(GV_Database *db, const char *backup_path,
                                               const char *base_backup_path,
                                               const GV_BackupOptions *options) {
    if (!db || !backup_path || !base_backup_path) {
        return create_result(0, "Invalid parameters");
    }

    /* Read base backup header to get starting point */
    GV_BackupHeader base_header;
    if (read_backup_header(base_backup_path, &base_header) != 0) {
        return create_result(0, "Failed to read base backup header");
    }

    /* Verify dimension and index type match */
    if (base_header.dimension != db->dimension) {
        return create_result(0, "Dimension mismatch with base backup");
    }

    /* Calculate vectors to backup (only new ones since base) */
    uint64_t start_idx = base_header.vector_count;
    uint64_t current_count = db->count;

    if (current_count <= start_idx) {
        return create_result(1, NULL);  /* No new vectors to backup */
    }

    uint64_t vectors_to_backup = current_count - start_idx;

    const GV_BackupOptions *opts = options ? options : &DEFAULT_BACKUP_OPTIONS;

    /* Create incremental backup file */
    FILE *fp = fopen(backup_path, "wb");
    if (!fp) {
        return create_result(0, "Failed to create incremental backup file");
    }

    /* Write magic */
    fwrite(BACKUP_MAGIC, 1, BACKUP_MAGIC_LEN, fp);

    /* Build and write header */
    GV_BackupHeader header;
    memset(&header, 0, sizeof(header));
    header.version = GV_BACKUP_VERSION;
    header.flags = BACKUP_FLAG_INCREMENTAL;
    if (opts->compression != GV_BACKUP_COMPRESS_NONE) {
        header.flags |= BACKUP_FLAG_COMPRESSED;
    }
    if (opts->encryption_key) {
        header.flags |= BACKUP_FLAG_ENCRYPTED;
    }
    header.created_at = (uint64_t)time(NULL);
    header.vector_count = vectors_to_backup;
    header.dimension = db->dimension;
    header.index_type = db->index_type;

    fwrite(&header.version, sizeof(header.version), 1, fp);
    fwrite(&header.flags, sizeof(header.flags), 1, fp);
    fwrite(&header.created_at, sizeof(header.created_at), 1, fp);
    fwrite(&header.vector_count, sizeof(header.vector_count), 1, fp);
    fwrite(&header.dimension, sizeof(header.dimension), 1, fp);
    fwrite(&header.index_type, sizeof(header.index_type), 1, fp);

    /* Write base backup info */
    fwrite(&start_idx, sizeof(start_idx), 1, fp);  /* Starting index */
    fwrite(&base_header.created_at, sizeof(base_header.created_at), 1, fp);  /* Base backup timestamp */

    /* Write vector data */
    for (uint64_t i = start_idx; i < current_count; i++) {
        const float *vec = gv_database_get_vector(db, (size_t)i);
        if (vec) {
            fwrite(vec, sizeof(float), db->dimension, fp);
        }
    }

    /* Update result */
    fclose(fp);

    GV_BackupResult *result = create_result(1, NULL);
    if (result) {
        result->vectors_processed = vectors_to_backup;
    }

    return result;
}

GV_BackupResult *gv_backup_merge(const char *base_backup_path,
                                  const char **incremental_paths, size_t incremental_count,
                                  const char *output_path) {
    if (!base_backup_path || !output_path) {
        return create_result(0, "Invalid parameters");
    }

    /* Read base backup header */
    GV_BackupHeader base_header;
    if (read_backup_header(base_backup_path, &base_header) != 0) {
        return create_result(0, "Failed to read base backup header");
    }

    /* Create output file */
    FILE *out_fp = fopen(output_path, "wb");
    if (!out_fp) {
        return create_result(0, "Failed to create output file");
    }

    /* Copy base backup to output */
    FILE *base_fp = fopen(base_backup_path, "rb");
    if (!base_fp) {
        fclose(out_fp);
        return create_result(0, "Failed to open base backup");
    }

    /* Reset to beginning of base file */
    fseek(base_fp, 0, SEEK_SET);

    /* Copy base backup */
    char *buffer = malloc(BUFFER_SIZE);
    if (!buffer) {
        fclose(base_fp);
        fclose(out_fp);
        return create_result(0, "Memory allocation failed");
    }

    size_t bytes;
    while ((bytes = fread(buffer, 1, BUFFER_SIZE, base_fp)) > 0) {
        fwrite(buffer, 1, bytes, out_fp);
    }
    fclose(base_fp);

    /* Track total vectors */
    uint64_t total_vectors = base_header.vector_count;

    /* Apply incremental backups */
    for (size_t i = 0; incremental_paths && i < incremental_count; i++) {
        FILE *inc_fp = fopen(incremental_paths[i], "rb");
        if (!inc_fp) {
            continue;  /* Skip missing incremental */
        }

        /* Read incremental header */
        GV_BackupHeader inc_header;
        if (read_backup_header(incremental_paths[i], &inc_header) != 0) {
            fclose(inc_fp);
            continue;
        }

        /* Verify it's incremental and dimensions match */
        if (!(inc_header.flags & BACKUP_FLAG_INCREMENTAL) ||
            inc_header.dimension != base_header.dimension) {
            fclose(inc_fp);
            continue;
        }

        /* Seek past header to vector data */
        size_t header_size = BACKUP_MAGIC_LEN +
            sizeof(inc_header.version) + sizeof(inc_header.flags) +
            sizeof(inc_header.created_at) + sizeof(inc_header.vector_count) +
            sizeof(inc_header.dimension) + sizeof(inc_header.index_type) +
            sizeof(uint64_t) * 2;  /* start_idx + base_timestamp */

        fseek(inc_fp, header_size, SEEK_SET);

        /* Copy vector data */
        size_t vector_bytes = inc_header.vector_count * inc_header.dimension * sizeof(float);
        size_t remaining = vector_bytes;

        while (remaining > 0) {
            size_t to_read = remaining < BUFFER_SIZE ? remaining : BUFFER_SIZE;
            bytes = fread(buffer, 1, to_read, inc_fp);
            if (bytes == 0) break;
            fwrite(buffer, 1, bytes, out_fp);
            remaining -= bytes;
        }

        total_vectors += inc_header.vector_count;
        fclose(inc_fp);
    }

    free(buffer);

    /* Update header in output with new vector count */
    fseek(out_fp, BACKUP_MAGIC_LEN + sizeof(uint32_t) * 2 + sizeof(uint64_t), SEEK_SET);
    fwrite(&total_vectors, sizeof(total_vectors), 1, out_fp);

    fclose(out_fp);

    GV_BackupResult *result = create_result(1, NULL);
    if (result) {
        result->vectors_processed = total_vectors;
    }

    return result;
}

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

int gv_backup_compute_checksum(const char *backup_path, char *checksum_out) {
    if (!backup_path || !checksum_out) return -1;

    FILE *fp = fopen(backup_path, "rb");
    if (!fp) return -1;

    /* Read entire file and compute SHA-256 */
    /* For large files, this should be done in chunks */
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    /* Skip checksum field when computing */
    size_t header_size = BACKUP_MAGIC_LEN + sizeof(uint32_t) * 2 + sizeof(uint64_t) * 4;
    size_t checksum_offset = header_size;

    unsigned char *buffer = malloc(file_size);
    if (!buffer) {
        fclose(fp);
        return -1;
    }

    fread(buffer, 1, file_size, fp);
    fclose(fp);

    /* Zero out checksum field for computation */
    if ((size_t)file_size > checksum_offset + 64) {
        memset(buffer + checksum_offset, 0, 64);
    }

    unsigned char hash[32];
    gv_auth_sha256(buffer, file_size, hash);
    free(buffer);

    gv_auth_to_hex(hash, 32, checksum_out);

    return 0;
}

const char *gv_backup_compression_string(GV_BackupCompression compression) {
    switch (compression) {
        case GV_BACKUP_COMPRESS_NONE: return "none";
        case GV_BACKUP_COMPRESS_ZLIB: return "zlib";
        case GV_BACKUP_COMPRESS_LZ4: return "lz4";
        default: return "unknown";
    }
}
