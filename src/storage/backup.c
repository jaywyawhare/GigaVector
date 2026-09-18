/**
 * @file gv_backup.c
 * @brief Backup and restore implementation.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE   /* for fileno()/fsync() prototypes on glibc */
#endif

#include "storage/backup.h"
#include "core/memory.h"
#include "storage/database.h"
#include "core/utils.h"
#include "security/auth.h"     /* For SHA-256 */
#include "security/crypto.h"   /* For encryption */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include "core/compat.h"
#include <sys/stat.h>
#include <stdint.h>
#ifndef _WIN32
#include <unistd.h>   /* fsync, close, unlink */
#include <fcntl.h>    /* open, O_RDONLY, O_DIRECTORY */
#endif

#define BACKUP_MAGIC "GVBAK"
#define BACKUP_MAGIC_LEN 5
#define BACKUP_CHECKSUM_LEN 64
#define BACKUP_HEADER_FIXED_SIZE (BACKUP_MAGIC_LEN + sizeof(uint32_t) * 4 + sizeof(uint64_t) * 4)
#define BACKUP_HEADER_SIZE (BACKUP_HEADER_FIXED_SIZE + BACKUP_CHECKSUM_LEN)
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

void backup_options_init(GV_BackupOptions *options) {
    if (!options) return;
    *options = DEFAULT_BACKUP_OPTIONS;
}

void restore_options_init(GV_RestoreOptions *options) {
    if (!options) return;
    *options = DEFAULT_RESTORE_OPTIONS;
}

static GV_BackupResult *create_result(int success, const char *error) {
    GV_BackupResult *result = gv_calloc(1, sizeof(GV_BackupResult));
    if (!result) return NULL;
    result->success = success;
    if (error) {
        result->error_message = gv_strdup(error);
    }
    return result;
}

static double get_time_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* fsync the directory containing `path` so a rename() into it is durable across
 * a crash (mirrors db_save_locked in database.c).  POSIX only; a no-op on
 * _WIN32 where rename durability is handled by the platform.  Returns 0 on
 * success (or on non-POSIX), -1 on failure. */
static int backup_fsync_parent_dir(const char *path) {
#ifndef _WIN32
    char dir_path[1024];
    size_t len = strlen(path);
    const char *slash = NULL;
    for (size_t i = len; i > 0; --i) {
        if (path[i - 1] == '/') { slash = &path[i - 1]; break; }
    }
    if (slash == NULL) {
        dir_path[0] = '.'; dir_path[1] = '\0';
    } else if (slash == path) {
        dir_path[0] = '/'; dir_path[1] = '\0';
    } else {
        size_t dlen = (size_t)(slash - path);
        if (dlen >= sizeof(dir_path)) dlen = sizeof(dir_path) - 1;
        memcpy(dir_path, path, dlen);
        dir_path[dlen] = '\0';
    }
    int dfd = open(dir_path, O_RDONLY | O_DIRECTORY);
    if (dfd < 0) return -1;
    int rc = fsync(dfd);
    close(dfd);
    return rc == 0 ? 0 : -1;
#else
    (void)path;
    return 0;
#endif
}

GV_BackupResult *backup_create(GV_Database *db, const char *backup_path,
                                   const GV_BackupOptions *options,
                                   GV_BackupProgressCallback progress,
                                   void *user_data) {
    if (!db || !backup_path) {
        return create_result(0, "Invalid parameters");
    }

    double start_time = get_time_seconds();
    const GV_BackupOptions *opts = options ? options : &DEFAULT_BACKUP_OPTIONS;

    /* Durability: write to a temp file, fsync it, then atomically rename over the
     * canonical backup path (mirrors db_save).  This ensures a crash or ENOSPC
     * mid-write never corrupts an existing/canonical backup — the final path only
     * appears once the new backup is complete and durable on disk. */
    char tmp_path[1024];
    if (snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", backup_path) >= (int)sizeof(tmp_path)) {
        return create_result(0, "Backup path too long");
    }

    FILE *fp = fopen(tmp_path, "wb");
    if (!fp) {
        return create_result(0, "Failed to create backup file");
    }

    if (fwrite(BACKUP_MAGIC, 1, BACKUP_MAGIC_LEN, fp) != BACKUP_MAGIC_LEN) {
        fclose(fp); remove(tmp_path); return create_result(0, "Failed to write backup magic");
    }

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

#define FWRITE_HDR(expr) \
    if ((expr) != 0) { \
        fclose(fp); remove(tmp_path); return create_result(0, "Failed to write backup header"); \
    }
    FWRITE_HDR(write_u32(fp, header.version))
    FWRITE_HDR(write_u32(fp, header.flags))
    FWRITE_HDR(write_u64(fp, header.created_at))
    FWRITE_HDR(write_u64(fp, header.vector_count))
    FWRITE_HDR(write_u32(fp, header.dimension))
    FWRITE_HDR(write_u32(fp, header.index_type))
#undef FWRITE_HDR

    /* Placeholder for sizes and checksum (will update at end) */
    long sizes_pos = ftell(fp);
    if (sizes_pos < 0) {
        fclose(fp);
        remove(tmp_path);
        return create_result(0, "Failed to get file position");
    }
    uint64_t zero = 0;
    if (write_u64(fp, zero) != 0 ||
        write_u64(fp, zero) != 0) {
        fclose(fp); remove(tmp_path); return create_result(0, "Failed to write backup placeholders");
    }
    char checksum_placeholder[64] = {0};
    if (fwrite(checksum_placeholder, 1, BACKUP_CHECKSUM_LEN, fp) != BACKUP_CHECKSUM_LEN) {
        fclose(fp); remove(tmp_path); return create_result(0, "Failed to write checksum placeholder");
    }

    uint64_t data_size = 0;
    size_t dimension = database_dimension(db);
    size_t count = database_count(db);
    size_t vector_size = dimension * sizeof(float);

    for (size_t i = 0; i < count; i++) {
        const float *vector = database_get_vector(db, i);

        if (vector) {
            if (write_floats(fp, vector, dimension) != 0) {
                fclose(fp);
                remove(tmp_path);
                return create_result(0, "Failed to write vector data (disk full?)");
            }
            data_size += vector_size;
        } else {
            float *zeros = gv_calloc(dimension, sizeof(float));
            if (!zeros) {
                fclose(fp);
                remove(tmp_path);
                return create_result(0, "Failed to allocate zero-vector buffer");
            }
            int zw = write_floats(fp, zeros, dimension);
            gv_free(zeros);
            if (zw != 0) {
                fclose(fp);
                remove(tmp_path);
                return create_result(0, "Failed to write placeholder vector (disk full?)");
            }
            data_size += vector_size;
        }

        if (progress && i % 1000 == 0) {
            progress(i, count, user_data);
        }
    }

    if (progress) {
        progress(count, count, user_data);
    }

    /* v2 metadata section (appended AFTER the raw vectors so the vector layout
     * stays byte-for-byte identical to v1).  We always write the meta_present
     * marker; when include_metadata is off (or there is no SoA storage to read
     * from) we write a zero marker and no records, yielding a valid v2 file with
     * an empty metadata section.  When on, we emit exactly `count` records in
     * vector-index order, each via write_metadata() — the same wire format the
     * main snapshot uses (uint32 count + length-prefixed key/value pairs). */
    uint32_t meta_present = (opts->include_metadata && db->soa_storage) ? 1u : 0u;
    if (write_u32(fp, meta_present) != 0) {
        fclose(fp);
        remove(tmp_path);
        return create_result(0, "Failed to write metadata marker (disk full?)");
    }
    if (meta_present) {
        for (size_t i = 0; i < count; i++) {
            /* soa_storage_get_metadata returns the vector's live GV_Metadata
             * chain (NULL if none); write_metadata handles NULL as count 0. */
            const GV_Metadata *meta = soa_storage_get_metadata(db->soa_storage, i);
            if (write_metadata(fp, meta) != 0) {
                fclose(fp);
                remove(tmp_path);
                return create_result(0, "Failed to write vector metadata (disk full?)");
            }
        }
    }

    long end_pos = ftell(fp);
    if (end_pos < 0) {
        fclose(fp);
        remove(tmp_path);
        return create_result(0, "Failed to get file position after writing vectors");
    }
    fseek(fp, sizes_pos, SEEK_SET);
    write_u64(fp, data_size);
    write_u64(fp, zero);
    fseek(fp, end_pos, SEEK_SET);

    fclose(fp);

    /* Patch the checksum into the (still temp) file's header. */
    char checksum[65];
    if (backup_compute_checksum(tmp_path, checksum) == 0) {
        fp = fopen(tmp_path, "r+b");
        if (fp) {
            fseek(fp, sizes_pos + 16, SEEK_SET);
            fwrite(checksum, 1, BACKUP_CHECKSUM_LEN, fp);
            fclose(fp);
        }
    }

    /* Durably flush the temp file before publishing it: fflush + fsync so a crash
     * after the rename cannot leave a renamed-but-incomplete backup. */
    {
        FILE *sf = fopen(tmp_path, "rb+");
        int sync_ok = 0;
        if (sf) {
            if (fflush(sf) == 0) {
#ifndef _WIN32
                sync_ok = (fsync(fileno(sf)) == 0);
#else
                sync_ok = 1;
#endif
            }
            if (fclose(sf) != 0) sync_ok = 0;
        }
        if (!sync_ok) {
            remove(tmp_path);
            return create_result(0, "Failed to fsync backup file");
        }
    }

    /* Atomically publish the completed backup, then fsync the directory so the
     * rename itself is durable. */
    if (gv_rename_replace(tmp_path, backup_path) != 0) {
        remove(tmp_path);
        return create_result(0, "Failed to publish backup file");
    }
    if (backup_fsync_parent_dir(backup_path) != 0) {
        return create_result(0, "Failed to fsync backup directory");
    }

    if (opts->verify_after) {
        GV_BackupResult *verify = backup_verify(backup_path, NULL);
        if (!verify->success) {
            const char *err = verify->error_message ? verify->error_message : "Verification failed";
            GV_BackupResult *r = create_result(0, err);
            backup_result_free(verify);
            return r;
        }
        backup_result_free(verify);
    }

    GV_BackupResult *result = create_result(1, NULL);
    result->bytes_processed = data_size;
    result->vectors_processed = db->count;
    result->elapsed_seconds = get_time_seconds() - start_time;

    return result;
}

GV_BackupResult *backup_create_from_file(const char *db_path, const char *backup_path,
                                             const GV_BackupOptions *options,
                                             GV_BackupProgressCallback progress,
                                             void *user_data) {
    if (!db_path || !backup_path) {
        return create_result(0, "Invalid parameters");
    }

    GV_Database *db = db_open(db_path, 0, GV_INDEX_TYPE_HNSW);
    if (!db) {
        return create_result(0, "Failed to open database");
    }

    GV_BackupResult *result = backup_create(db, backup_path, options, progress, user_data);
    db_close(db);

    return result;
}

void backup_result_free(GV_BackupResult *result) {
    if (!result) return;
    gv_free(result->error_message);
    gv_free(result);
}

/* Read the v2 metadata section (positioned immediately after the raw vectors)
 * and apply it to the just-restored database, per vector index.
 *
 * Backward compatibility: for v1 backups there is no metadata section, so this
 * is a no-op (callers gate on header->version).  For v2 backups the section
 * begins with a uint32 meta_present marker; when it is 0 (include_metadata was
 * off, or the source had no metadata) there are no records to read.
 *
 * `fp` must be positioned at the start of the metadata section.  Returns 0 on
 * success (including "nothing to apply"); a truncated/garbled section is
 * treated as a soft failure — the vectors are already restored, so we simply
 * stop applying metadata and return 0 rather than failing the whole restore. */
static int backup_apply_metadata_section(FILE *fp, GV_Database *db,
                                         uint32_t version, uint64_t vector_count) {
    if (!fp || !db || version < 2) return 0;

    uint32_t meta_present = 0;
    if (read_u32(fp, &meta_present) != 0 || meta_present == 0) {
        return 0;
    }

    for (uint64_t i = 0; i < vector_count; i++) {
        uint32_t pair_count = 0;
        if (read_u32(fp, &pair_count) != 0) {
            return 0;  /* truncated; stop, keep what we have */
        }
        if (pair_count == 0) {
            continue;  /* vector had no metadata */
        }

        char **keys = gv_calloc(pair_count, sizeof(char *));
        char **vals = gv_calloc(pair_count, sizeof(char *));
        if (!keys || !vals) {
            gv_free(keys);
            gv_free(vals);
            return 0;
        }

        uint32_t got = 0;
        for (; got < pair_count; got++) {
            keys[got] = read_string(fp);
            vals[got] = read_string(fp);
            if (!keys[got] || !vals[got]) {
                /* Truncated pair: free the dangling half and bail out below. */
                gv_free(keys[got]);
                gv_free(vals[got]);
                keys[got] = NULL;
                vals[got] = NULL;
                break;
            }
        }

        if (got > 0) {
            db_update_vector_metadata(db, (size_t)i,
                                      (const char *const *)keys,
                                      (const char *const *)vals, got);
        }

        for (uint32_t j = 0; j < pair_count; j++) {
            gv_free(keys[j]);
            gv_free(vals[j]);
        }
        gv_free(keys);
        gv_free(vals);

        if (got < pair_count) {
            return 0;  /* truncated record; stop applying further metadata */
        }
    }

    return 0;
}

GV_BackupResult *backup_restore(const char *backup_path, const char *db_path,
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
        GV_BackupResult *verify = backup_verify(backup_path, opts->decryption_key);
        if (!verify->success) {
            const char *err = verify->error_message ? verify->error_message : "Checksum verification failed";
            GV_BackupResult *r = create_result(0, err);
            backup_result_free(verify);
            return r;
        }
        backup_result_free(verify);
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
    if (read_u32(fp, &header.version) != 0 ||
        read_u32(fp, &header.flags) != 0 ||
        read_u64(fp, &header.created_at) != 0 ||
        read_u64(fp, &header.vector_count) != 0 ||
        read_u32(fp, &header.dimension) != 0 ||
        read_u32(fp, &header.index_type) != 0 ||
        read_u64(fp, &header.original_size) != 0 ||
        read_u64(fp, &header.compressed_size) != 0 ||
        fread(header.checksum, 1, BACKUP_CHECKSUM_LEN, fp) != BACKUP_CHECKSUM_LEN) {
        fclose(fp);
        return create_result(0, "Backup file truncated — failed to read header");
    }

    GV_Database *db = db_open(NULL, header.dimension, header.index_type);
    if (!db) {
        fclose(fp);
        return create_result(0, "Failed to create database");
    }

    size_t vector_size = header.dimension * sizeof(float);
    float *buffer = gv_alloc(vector_size);
    if (!buffer) {
        db_close(db);
        fclose(fp);
        return create_result(0, "Memory allocation failed");
    }

    uint64_t vectors_read = 0;
    while (vectors_read < header.vector_count) {
        if (read_floats(fp, buffer, header.dimension) != 0) {
            break;
        }

        db_add_vector(db, buffer, header.dimension);
        vectors_read++;

        if (progress && vectors_read % 1000 == 0) {
            progress(vectors_read, header.vector_count, user_data);
        }
    }

    gv_free(buffer);

    /* fp is now positioned right after the raw vectors: for v2 backups the
     * per-vector metadata section follows here.  Apply it before closing fp.
     * Only meaningful when every declared vector was read back. */
    if (vectors_read == header.vector_count) {
        backup_apply_metadata_section(fp, db, header.version, header.vector_count);
    }

    fclose(fp);

    if (progress) {
        progress(header.vector_count, header.vector_count, user_data);
    }

    if (db_save(db, db_path) != 0) {
        db_close(db);
        return create_result(0, "Failed to save database");
    }

    db_close(db);

    GV_BackupResult *result = create_result(1, NULL);
    result->bytes_processed = header.original_size;
    result->vectors_processed = vectors_read;
    result->elapsed_seconds = get_time_seconds() - start_time;

    return result;
}

GV_BackupResult *backup_restore_to_db(const char *backup_path,
                                          const GV_RestoreOptions *options,
                                          GV_Database **db) {
    if (!backup_path || !db) {
        return create_result(0, "Invalid parameters");
    }

    const GV_RestoreOptions *opts = options ? options : &DEFAULT_RESTORE_OPTIONS;

    if (opts->verify_checksum) {
        GV_BackupResult *verify = backup_verify(backup_path, opts->decryption_key);
        if (!verify->success) {
            const char *err = verify->error_message ? verify->error_message : "Checksum verification failed";
            GV_BackupResult *r = create_result(0, err);
            backup_result_free(verify);
            return r;
        }
        backup_result_free(verify);
    }

    GV_BackupHeader header;
    if (backup_read_header(backup_path, &header) != 0) {
        return create_result(0, "Failed to read backup header");
    }

    *db = db_open(NULL, header.dimension, header.index_type);
    if (!*db) {
        return create_result(0, "Failed to create database");
    }

    FILE *fp = fopen(backup_path, "rb");
    if (!fp) {
        db_close(*db);
        *db = NULL;
        return create_result(0, "Failed to open backup file");
    }

    fseek(fp, BACKUP_HEADER_SIZE, SEEK_SET);

    size_t vector_size = header.dimension * sizeof(float);
    float *buffer = gv_alloc(vector_size);
    if (!buffer) {
        fclose(fp);
        db_close(*db);
        *db = NULL;
        return create_result(0, "Memory allocation failed");
    }

    uint64_t vectors_read = 0;
    while (vectors_read < header.vector_count) {
        if (read_floats(fp, buffer, header.dimension) != 0) {
            break;
        }
        db_add_vector(*db, buffer, header.dimension);
        vectors_read++;
    }

    gv_free(buffer);

    /* Apply the v2 per-vector metadata section (no-op for v1 or empty section)
     * before closing the file — fp is positioned right after the raw vectors. */
    if (vectors_read == header.vector_count) {
        backup_apply_metadata_section(fp, *db, header.version, header.vector_count);
    }

    fclose(fp);

    GV_BackupResult *result = create_result(1, NULL);
    result->vectors_processed = vectors_read;

    return result;
}

int backup_read_header(const char *backup_path, GV_BackupHeader *header) {
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

    if (read_u32(fp, &header->version) != 0 ||
        read_u32(fp, &header->flags) != 0 ||
        read_u64(fp, &header->created_at) != 0 ||
        read_u64(fp, &header->vector_count) != 0 ||
        read_u32(fp, &header->dimension) != 0 ||
        read_u32(fp, &header->index_type) != 0 ||
        read_u64(fp, &header->original_size) != 0 ||
        read_u64(fp, &header->compressed_size) != 0 ||
        fread(header->checksum, 1, BACKUP_CHECKSUM_LEN, fp) != BACKUP_CHECKSUM_LEN) {
        fclose(fp);
        return -1;
    }
    header->checksum[64] = '\0';

    fclose(fp);
    return 0;
}

GV_BackupResult *backup_verify(const char *backup_path, const char *decryption_key) {
    if (!backup_path) {
        return create_result(0, "Invalid parameters");
    }

    GV_BackupHeader header;
    if (backup_read_header(backup_path, &header) != 0) {
        return create_result(0, "Failed to read backup header");
    }

    /* Accept the whole supported range so older (v1, no-metadata) backups still
     * verify.  v2 appends a metadata section after the vectors; the checksum
     * (computed over the entire file below) already covers those bytes. */
    if (header.version < GV_BACKUP_VERSION_MIN || header.version > GV_BACKUP_VERSION) {
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
        GV_CryptoContext *ctx = crypto_create(NULL);
        if (!ctx) {
            return create_result(0, "Failed to create crypto context for verification");
        }

        GV_CryptoKey key;
        unsigned char salt[16] = {0};  /* Backup uses zero salt for deterministic derivation */
        if (crypto_derive_key(ctx, decryption_key, strlen(decryption_key),
                                 salt, sizeof(salt), &key) != 0) {
            crypto_destroy(ctx);
            return create_result(0, "Failed to derive decryption key");
        }

        /* Read probe block: first vector's encrypted data */
        FILE *fp = fopen(backup_path, "rb");
        if (!fp) {
            crypto_wipe_key(&key);
            crypto_destroy(ctx);
            return create_result(0, "Failed to open backup file");
        }

        size_t header_total = BACKUP_HEADER_SIZE;
        fseek(fp, (long)header_total, SEEK_SET);

        size_t probe_size = header.dimension * sizeof(float);
        /* Encrypted data may have padding — read extra 16 bytes */
        size_t read_size = probe_size + 16;
        unsigned char *encrypted_probe = gv_alloc(read_size);
        unsigned char *decrypted_probe = gv_alloc(read_size);
        if (!encrypted_probe || !decrypted_probe) {
            gv_free(encrypted_probe);
            gv_free(decrypted_probe);
            fclose(fp);
            crypto_wipe_key(&key);
            crypto_destroy(ctx);
            return create_result(0, "Memory allocation failed during verification");
        }

        size_t bytes_read = fread(encrypted_probe, 1, read_size, fp);
        fclose(fp);

        if (bytes_read < probe_size) {
            gv_free(encrypted_probe);
            gv_free(decrypted_probe);
            crypto_wipe_key(&key);
            crypto_destroy(ctx);
            return create_result(0, "Backup file truncated — cannot read probe block");
        }

        size_t decrypted_len = 0;
        int dec_rc = crypto_decrypt(ctx, &key, encrypted_probe, bytes_read,
                                       decrypted_probe, &decrypted_len);
        gv_free(encrypted_probe);
        gv_free(decrypted_probe);
        crypto_wipe_key(&key);
        crypto_destroy(ctx);

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

    /* Overflow guard: vector_count (uint64) * dimension (uint32) * sizeof(float)
     * can wrap around, producing a tiny expected_min that lets a truncated (or
     * maliciously crafted) file pass the size check.  Reject any header whose
     * declared payload cannot fit in size_t before doing the multiply. */
    size_t per_vector = (size_t)header.dimension * sizeof(float);
    if (header.dimension != 0 && header.vector_count != 0 &&
        header.vector_count > (SIZE_MAX - BACKUP_HEADER_SIZE) / per_vector) {
        return create_result(0, "Backup header declares an implausible size");
    }
    size_t expected_min = BACKUP_HEADER_SIZE +
                          (size_t)header.vector_count * per_vector;

    if (!(header.flags & BACKUP_FLAG_ENCRYPTED) && file_size < (long)expected_min) {
        return create_result(0, "Backup file appears truncated");
    }

    /* Verify checksum if present — force null termination before compare */
    header.checksum[BACKUP_CHECKSUM_LEN - 1] = '\0';
    if (header.checksum[0] != '\0') {
        char computed[65];
        if (backup_compute_checksum(backup_path, computed) == 0) {
            computed[64] = '\0';
            if (strcmp(computed, header.checksum) != 0) {
                return create_result(0, "Checksum mismatch — backup may be corrupted");
            }
        }
    }

    return create_result(1, NULL);
}

int backup_get_info(const char *backup_path, char *info_buf, size_t buf_size) {
    if (!backup_path || !info_buf || buf_size == 0) return -1;

    GV_BackupHeader header;
    if (backup_read_header(backup_path, &header) != 0) {
        snprintf(info_buf, buf_size, "Error: Failed to read backup header");
        return -1;
    }

    time_t created = (time_t)header.created_at;
    struct tm *tm_info = localtime(&created);
    char time_buf[64];
    if (tm_info) {
        strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", tm_info);
    } else {
        /* localtime() can return NULL (e.g. out-of-range created_at); passing
         * that to strftime is undefined behavior. */
        snprintf(time_buf, sizeof(time_buf), "unknown");
    }

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

    if (read_u32(fp, &header->version) != 0 ||
        read_u32(fp, &header->flags) != 0 ||
        read_u64(fp, &header->created_at) != 0 ||
        read_u64(fp, &header->vector_count) != 0 ||
        read_u32(fp, &header->dimension) != 0 ||
        read_u32(fp, &header->index_type) != 0) {
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}

GV_BackupResult *backup_create_incremental(GV_Database *db, const char *backup_path,
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

    write_u32(fp, header.version);
    write_u32(fp, header.flags);
    write_u64(fp, header.created_at);
    write_u64(fp, header.vector_count);
    write_u32(fp, header.dimension);
    write_u32(fp, header.index_type);

    write_u64(fp, start_idx);
    write_u64(fp, base_header.created_at);

    for (uint64_t i = start_idx; i < current_count; i++) {
        const float *vec = database_get_vector(db, (size_t)i);
        if (vec) {
            write_floats(fp, vec, db->dimension);
        }
    }

    fclose(fp);

    GV_BackupResult *result = create_result(1, NULL);
    if (result) {
        result->vectors_processed = vectors_to_backup;
    }

    return result;
}

GV_BackupResult *backup_merge(const char *base_backup_path,
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

    char *buffer = gv_alloc(BUFFER_SIZE);
    if (!buffer) {
        fclose(base_fp);
        fclose(out_fp);
        return create_result(0, "Memory allocation failed");
    }

    size_t bytes;
    while ((bytes = fread(buffer, 1, BUFFER_SIZE, base_fp)) > 0) {
        if (fwrite(buffer, 1, bytes, out_fp) != bytes) {
            /* Silent truncation (e.g. ENOSPC) would corrupt the merged backup. */
            fclose(base_fp);
            fclose(out_fp);
            gv_free(buffer);
            remove(output_path);
            return create_result(0, "Failed to write merged backup (disk full?)");
        }
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

        /* Incremental header layout: magic + version + flags + created_at +
         * vector_count + dimension + index_type + start_idx + base_created_at.
         * No checksum field — do NOT include BACKUP_CHECKSUM_LEN here. */
        size_t header_size = BACKUP_MAGIC_LEN +
            sizeof(inc_header.version) + sizeof(inc_header.flags) +
            sizeof(inc_header.created_at) + sizeof(inc_header.vector_count) +
            sizeof(inc_header.dimension) + sizeof(inc_header.index_type) +
            sizeof(uint64_t) * 2;

        fseek(inc_fp, header_size, SEEK_SET);

        /* Bound the multiplication: vector_count (uint64) * dimension (uint32) *
         * sizeof(float) can overflow size_t.  A wrapped-around vector_bytes would
         * copy the wrong amount of data; skip any incremental that can't fit. */
        size_t inc_per_vector = (size_t)inc_header.dimension * sizeof(float);
        if (inc_header.dimension != 0 && inc_header.vector_count != 0 &&
            inc_header.vector_count > SIZE_MAX / inc_per_vector) {
            fclose(inc_fp);
            continue;
        }
        size_t vector_bytes = (size_t)inc_header.vector_count * inc_per_vector;
        size_t remaining = vector_bytes;

        while (remaining > 0) {
            size_t to_read = remaining < BUFFER_SIZE ? remaining : BUFFER_SIZE;
            bytes = fread(buffer, 1, to_read, inc_fp);
            if (bytes == 0) break;
            if (fwrite(buffer, 1, bytes, out_fp) != bytes) {
                fclose(inc_fp);
                fclose(out_fp);
                gv_free(buffer);
                remove(output_path);
                return create_result(0, "Failed to write merged backup (disk full?)");
            }
            remaining -= bytes;
        }

        total_vectors += inc_header.vector_count;
        fclose(inc_fp);
    }

    gv_free(buffer);

    fseek(out_fp, BACKUP_MAGIC_LEN + sizeof(uint32_t) * 2 + sizeof(uint64_t), SEEK_SET);
    if (write_u64(out_fp, total_vectors) != 0) {
        fclose(out_fp);
        remove(output_path);
        return create_result(0, "Failed to write merged backup vector count");
    }

    fclose(out_fp);

    GV_BackupResult *result = create_result(1, NULL);
    if (result) {
        result->vectors_processed = total_vectors;
    }

    return result;
}

int backup_compute_checksum(const char *backup_path, char *checksum_out) {
    if (!backup_path || !checksum_out) return -1;

    FILE *fp = fopen(backup_path, "rb");
    if (!fp) return -1;

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    /* ftell() returns -1 on error; an empty file yields 0.  Either would lead to
     * a negative/zero-sized gv_alloc (a huge alloc after the implicit conversion
     * to size_t for -1) and an out-of-bounds auth_sha256 read.  Reject both. */
    if (file_size <= 0) {
        fclose(fp);
        return -1;
    }

    size_t checksum_offset = BACKUP_HEADER_FIXED_SIZE;

    unsigned char *buffer = gv_alloc(file_size);
    if (!buffer) {
        fclose(fp);
        return -1;
    }

    size_t to_read = (size_t)file_size;
    if (fread(buffer, 1, to_read, fp) != to_read) {
        gv_free(buffer);
        fclose(fp);
        return -1;
    }
    fclose(fp);

    if ((size_t)file_size > checksum_offset + BACKUP_CHECKSUM_LEN) {
        memset(buffer + checksum_offset, 0, BACKUP_CHECKSUM_LEN);
    }

    unsigned char hash[32];
    auth_sha256(buffer, file_size, hash);
    gv_free(buffer);

    auth_to_hex(hash, 32, checksum_out);

    return 0;
}

const char *backup_compression_string(GV_BackupCompression compression) {
    switch (compression) {
        case GV_BACKUP_COMPRESS_NONE: return "none";
        case GV_BACKUP_COMPRESS_ZLIB: return "zlib";
        case GV_BACKUP_COMPRESS_LZ4: return "lz4";
        default: return "unknown";
    }
}
