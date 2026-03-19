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

#define BACKUP_MAGIC "GVBAK"
#define BACKUP_MAGIC_LEN 5
#define BUFFER_SIZE (64 * 1024)

#define BACKUP_FLAG_COMPRESSED 0x01
#define BACKUP_FLAG_ENCRYPTED  0x02
#define BACKUP_FLAG_INCREMENTAL 0x04

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

GV_BackupResult *gv_backup_create(GV_Database *db, const char *backup_path,
                                   const GV_BackupOptions *options,
                                   GV_BackupProgressCallback progress,
                                   void *user_data) {
    if (!db || !backup_path) {
        return create_result(0, "Invalid parameters");
    }

    double start_time = get_time_seconds();
    const GV_BackupOptions *opts = options ? options : &DEFAULT_BACKUP_OPTIONS;

    FILE *fp = fopen(backup_path, "wb");
    if (!fp) {
        return create_result(0, "Failed to create backup file");
    }

    fwrite(BACKUP_MAGIC, 1, BACKUP_MAGIC_LEN, fp);

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
    fwrite(&zero, sizeof(zero), 1, fp);
    fwrite(&zero, sizeof(zero), 1, fp);
    char checksum_placeholder[65] = {0};
    fwrite(checksum_placeholder, 1, 64, fp);

    uint64_t data_size = 0;
    size_t dimension = gv_database_dimension(db);
    size_t count = gv_database_count(db);
    size_t vector_size = dimension * sizeof(float);

    for (size_t i = 0; i < count; i++) {
        const float *vector = gv_database_get_vector(db, i);

        if (vector) {
            fwrite(vector, 1, vector_size, fp);
            data_size += vector_size;
        } else {
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

    long end_pos = ftell(fp);
    fseek(fp, sizes_pos, SEEK_SET);
    fwrite(&data_size, sizeof(data_size), 1, fp);
    fwrite(&zero, sizeof(zero), 1, fp);
    fseek(fp, end_pos, SEEK_SET);

    fclose(fp);

    char checksum[65];
    if (gv_backup_compute_checksum(backup_path, checksum) == 0) {
        fp = fopen(backup_path, "r+b");
        if (fp) {
            fseek(fp, sizes_pos + 16, SEEK_SET);
            fwrite(checksum, 1, 64, fp);
            fclose(fp);
        }
    }

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

GV_BackupResult *gv_backup_restore(const char *backup_path, const char *db_path,
                                    const GV_RestoreOptions *options,
                                    GV_BackupProgressCallback progress,
                                    void *user_data) {
    if (!backup_path || !db_path) {
        return create_result(0, "Invalid parameters");
    }

    const GV_RestoreOptions *opts = options ? options : &DEFAULT_RESTORE_OPTIONS;
    double start_time = get_time_seconds();

    if (!opts->overwrite) {
        struct stat st;
        if (stat(db_path, &st) == 0) {
            return create_result(0, "Destination file already exists");
        }
    }

    if (opts->verify_checksum) {
        GV_BackupResult *verify = gv_backup_verify(backup_path, opts->decryption_key);
        if (!verify->success) {
            char *err = verify->error_message ? strdup(verify->error_message) : strdup("Checksum verification failed");
            gv_backup_result_free(verify);
            return create_result(0, err);
        }
        gv_backup_result_free(verify);
    }

    FILE *fp = fopen(backup_path, "rb");
    if (!fp) {
        return create_result(0, "Failed to open backup file");
    }

    char magic[BACKUP_MAGIC_LEN];
    if (fread(magic, 1, BACKUP_MAGIC_LEN, fp) != BACKUP_MAGIC_LEN ||
        memcmp(magic, BACKUP_MAGIC, BACKUP_MAGIC_LEN) != 0) {
        fclose(fp);
        return create_result(0, "Invalid backup file format");
    }

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

    GV_Database *db = gv_db_open(NULL, header.dimension, header.index_type);
    if (!db) {
        fclose(fp);
        return create_result(0, "Failed to create database");
    }

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

    if (opts->verify_checksum) {
        GV_BackupResult *verify = gv_backup_verify(backup_path, opts->decryption_key);
        if (!verify->success) {
            char *err = verify->error_message ? strdup(verify->error_message) : strdup("Checksum verification failed");
            gv_backup_result_free(verify);
            return create_result(0, err);
        }
        gv_backup_result_free(verify);
    }

    GV_BackupHeader header;
    if (gv_backup_read_header(backup_path, &header) != 0) {
        return create_result(0, "Failed to read backup header");
    }

    *db = gv_db_open(NULL, header.dimension, header.index_type);
    if (!*db) {
        return create_result(0, "Failed to create database");
    }

    FILE *fp = fopen(backup_path, "rb");
    if (!fp) {
        gv_db_close(*db);
        *db = NULL;
        return create_result(0, "Failed to open backup file");
    }

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

int gv_backup_read_header(const char *backup_path, GV_BackupHeader *header) {
    if (!backup_path || !header) return -1;

    FILE *fp = fopen(backup_path, "rb");
    if (!fp) return -1;

    memset(header, 0, sizeof(*header));

    char magic[BACKUP_MAGIC_LEN];
    if (fread(magic, 1, BACKUP_MAGIC_LEN, fp) != BACKUP_MAGIC_LEN ||
        memcmp(magic, BACKUP_MAGIC, BACKUP_MAGIC_LEN) != 0) {
        fclose(fp);
        return -1;
    }

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
    if (!backup_path) {
        return create_result(0, "Invalid parameters");
    }

    GV_BackupHeader header;
    if (gv_backup_read_header(backup_path, &header) != 0) {
        return create_result(0, "Failed to read backup header");
    }

    if (header.version != GV_BACKUP_VERSION) {
        return create_result(0, "Unsupported backup version");
    }

    /* Check if backup is encrypted */
    if (header.flags & BACKUP_FLAG_ENCRYPTED) {
        if (!decryption_key || decryption_key[0] == '\0') {
            return create_result(0, "Backup is encrypted but no decryption key provided");
        }

        /* Verify decryption key by attempting to decrypt a small probe block.
         * Read the first vector-sized chunk of data after the header and try
         * to decrypt it — if decryption succeeds and produces valid floats,
         * the key is correct. */
        GV_CryptoContext *ctx = gv_crypto_create(NULL);
        if (!ctx) {
            return create_result(0, "Failed to create crypto context for verification");
        }

        GV_CryptoKey key;
        unsigned char salt[16] = {0};  /* Backup uses zero salt for deterministic derivation */
        if (gv_crypto_derive_key(ctx, decryption_key, strlen(decryption_key),
                                 salt, sizeof(salt), &key) != 0) {
            gv_crypto_destroy(ctx);
            return create_result(0, "Failed to derive decryption key");
        }

        /* Read probe block: first vector's encrypted data */
        FILE *fp = fopen(backup_path, "rb");
        if (!fp) {
            gv_crypto_wipe_key(&key);
            gv_crypto_destroy(ctx);
            return create_result(0, "Failed to open backup file");
        }

        size_t header_total = BACKUP_MAGIC_LEN + sizeof(uint32_t) * 2 +
                              sizeof(uint64_t) * 4 + 64;
        fseek(fp, (long)header_total, SEEK_SET);

        size_t probe_size = header.dimension * sizeof(float);
        /* Encrypted data may have padding — read extra 16 bytes */
        size_t read_size = probe_size + 16;
        unsigned char *encrypted_probe = malloc(read_size);
        unsigned char *decrypted_probe = malloc(read_size);
        if (!encrypted_probe || !decrypted_probe) {
            free(encrypted_probe);
            free(decrypted_probe);
            fclose(fp);
            gv_crypto_wipe_key(&key);
            gv_crypto_destroy(ctx);
            return create_result(0, "Memory allocation failed during verification");
        }

        size_t bytes_read = fread(encrypted_probe, 1, read_size, fp);
        fclose(fp);

        if (bytes_read < probe_size) {
            free(encrypted_probe);
            free(decrypted_probe);
            gv_crypto_wipe_key(&key);
            gv_crypto_destroy(ctx);
            return create_result(0, "Backup file truncated — cannot read probe block");
        }

        size_t decrypted_len = 0;
        int dec_rc = gv_crypto_decrypt(ctx, &key, encrypted_probe, bytes_read,
                                       decrypted_probe, &decrypted_len);
        free(encrypted_probe);
        free(decrypted_probe);
        gv_crypto_wipe_key(&key);
        gv_crypto_destroy(ctx);

        if (dec_rc != 0) {
            return create_result(0, "Decryption failed — wrong key or corrupted backup");
        }
    }

    FILE *fp = fopen(backup_path, "rb");
    if (!fp) {
        return create_result(0, "Failed to open backup file");
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fclose(fp);

    size_t expected_min = BACKUP_MAGIC_LEN + sizeof(uint32_t) * 2 +
                          sizeof(uint64_t) * 4 + 64 +
                          header.vector_count * header.dimension * sizeof(float);

    if (!(header.flags & BACKUP_FLAG_ENCRYPTED) && file_size < (long)expected_min) {
        return create_result(0, "Backup file appears truncated");
    }

    /* Verify checksum if present */
    if (header.checksum[0] != '\0') {
        char computed[65];
        if (gv_backup_compute_checksum(backup_path, computed) == 0) {
            if (strcmp(computed, header.checksum) != 0) {
                return create_result(0, "Checksum mismatch — backup may be corrupted");
            }
        }
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

    time_t created = (time_t)header.created_at;
    struct tm *tm_info = localtime(&created);
    char time_buf[64];
    strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", tm_info);

    const char *index_type;
    switch (header.index_type) {
        case 0: index_type = "KD-Tree"; break;
        case 1: index_type = "HNSW"; break;
        case 2: index_type = "IVF-PQ"; break;
        case 3: index_type = "Sparse"; break;
        default: index_type = "Unknown"; break;
    }

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

    GV_BackupHeader base_header;
    if (read_backup_header(base_backup_path, &base_header) != 0) {
        return create_result(0, "Failed to read base backup header");
    }

    if (base_header.dimension != db->dimension) {
        return create_result(0, "Dimension mismatch with base backup");
    }

    uint64_t start_idx = base_header.vector_count;
    uint64_t current_count = db->count;

    if (current_count <= start_idx) {
        return create_result(1, NULL);  /* No new vectors to backup */
    }

    uint64_t vectors_to_backup = current_count - start_idx;

    const GV_BackupOptions *opts = options ? options : &DEFAULT_BACKUP_OPTIONS;

    FILE *fp = fopen(backup_path, "wb");
    if (!fp) {
        return create_result(0, "Failed to create incremental backup file");
    }

    fwrite(BACKUP_MAGIC, 1, BACKUP_MAGIC_LEN, fp);

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

    fwrite(&start_idx, sizeof(start_idx), 1, fp);
    fwrite(&base_header.created_at, sizeof(base_header.created_at), 1, fp);

    for (uint64_t i = start_idx; i < current_count; i++) {
        const float *vec = gv_database_get_vector(db, (size_t)i);
        if (vec) {
            fwrite(vec, sizeof(float), db->dimension, fp);
        }
    }

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

    GV_BackupHeader base_header;
    if (read_backup_header(base_backup_path, &base_header) != 0) {
        return create_result(0, "Failed to read base backup header");
    }

    FILE *out_fp = fopen(output_path, "wb");
    if (!out_fp) {
        return create_result(0, "Failed to create output file");
    }

    FILE *base_fp = fopen(base_backup_path, "rb");
    if (!base_fp) {
        fclose(out_fp);
        return create_result(0, "Failed to open base backup");
    }

    fseek(base_fp, 0, SEEK_SET);

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

    uint64_t total_vectors = base_header.vector_count;

    for (size_t i = 0; incremental_paths && i < incremental_count; i++) {
        FILE *inc_fp = fopen(incremental_paths[i], "rb");
        if (!inc_fp) {
            continue;  /* Skip missing incremental */
        }

        GV_BackupHeader inc_header;
        if (read_backup_header(incremental_paths[i], &inc_header) != 0) {
            fclose(inc_fp);
            continue;
        }

        if (!(inc_header.flags & BACKUP_FLAG_INCREMENTAL) ||
            inc_header.dimension != base_header.dimension) {
            fclose(inc_fp);
            continue;
        }

        size_t header_size = BACKUP_MAGIC_LEN +
            sizeof(inc_header.version) + sizeof(inc_header.flags) +
            sizeof(inc_header.created_at) + sizeof(inc_header.vector_count) +
            sizeof(inc_header.dimension) + sizeof(inc_header.index_type) +
            sizeof(uint64_t) * 2;

        fseek(inc_fp, header_size, SEEK_SET);

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

    fseek(out_fp, BACKUP_MAGIC_LEN + sizeof(uint32_t) * 2 + sizeof(uint64_t), SEEK_SET);
    fwrite(&total_vectors, sizeof(total_vectors), 1, out_fp);

    fclose(out_fp);

    GV_BackupResult *result = create_result(1, NULL);
    if (result) {
        result->vectors_processed = total_vectors;
    }

    return result;
}

int gv_backup_compute_checksum(const char *backup_path, char *checksum_out) {
    if (!backup_path || !checksum_out) return -1;

    FILE *fp = fopen(backup_path, "rb");
    if (!fp) return -1;

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    size_t header_size = BACKUP_MAGIC_LEN + sizeof(uint32_t) * 2 + sizeof(uint64_t) * 4;
    size_t checksum_offset = header_size;

    unsigned char *buffer = malloc(file_size);
    if (!buffer) {
        fclose(fp);
        return -1;
    }

    fread(buffer, 1, file_size, fp);
    fclose(fp);

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
