/**
 * @file gv_inspect_cli.c
 * @brief Command-line inspection tool for GigaVector databases and backups.
 *
 * Usage: gvinspect <file_path> [options]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#include "gigavector/gv_backup.h"
#include "gigavector/gv_database.h"

static void print_usage(const char *program) {
    printf("GigaVector Inspect Tool\n\n");
    printf("Usage: %s <file_path> [options]\n\n", program);
    printf("Options:\n");
    printf("  -s, --stats             Show detailed statistics\n");
    printf("  -v, --verify            Verify file integrity\n");
    printf("  -j, --json              Output in JSON format\n");
    printf("  -h, --help              Show this help message\n");
    printf("\n");
    printf("Supported file types:\n");
    printf("  .gvdb   - GigaVector database files\n");
    printf("  .gvb    - GigaVector backup files\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s vectors.gvdb --stats\n", program);
    printf("  %s backup.gvb --verify\n", program);
    printf("  %s database.gvdb --json\n", program);
}

static int ends_with(const char *str, const char *suffix) {
    size_t str_len = strlen(str);
    size_t suffix_len = strlen(suffix);
    if (suffix_len > str_len) return 0;
    return strcmp(str + str_len - suffix_len, suffix) == 0;
}

static void inspect_backup(const char *path, int stats, int verify, int json) {
    if (verify) {
        GV_BackupResult *result = gv_backup_verify(path, NULL);
        if (json) {
            printf("{\"valid\": %s", result->success ? "true" : "false");
            if (result->error_message) {
                printf(", \"error\": \"%s\"", result->error_message);
            }
            printf("}\n");
        } else {
            if (result->success) {
                printf("Backup verification: PASSED\n");
            } else {
                printf("Backup verification: FAILED\n");
                if (result->error_message) {
                    printf("  Error: %s\n", result->error_message);
                }
            }
        }
        gv_backup_result_free(result);
        return;
    }

    GV_BackupHeader header;
    if (gv_backup_read_header(path, &header) != 0) {
        if (json) {
            printf("{\"error\": \"Failed to read backup header\"}\n");
        } else {
            fprintf(stderr, "Error: Failed to read backup header\n");
        }
        return;
    }

    if (json) {
        printf("{\n");
        printf("  \"type\": \"backup\",\n");
        printf("  \"version\": %u,\n", header.version);
        printf("  \"created_at\": %llu,\n", (unsigned long long)header.created_at);
        printf("  \"vector_count\": %llu,\n", (unsigned long long)header.vector_count);
        printf("  \"dimension\": %u,\n", header.dimension);
        printf("  \"index_type\": %u,\n", header.index_type);
        printf("  \"original_size\": %llu,\n", (unsigned long long)header.original_size);
        printf("  \"compressed_size\": %llu,\n", (unsigned long long)header.compressed_size);
        printf("  \"checksum\": \"%s\"\n", header.checksum);
        printf("}\n");
    } else {
        char info[1024];
        gv_backup_get_info(path, info, sizeof(info));
        printf("%s\n", info);

        if (stats) {
            printf("\nDetailed Statistics:\n");
            printf("  Data size per vector: %zu bytes\n", header.dimension * sizeof(float));
            printf("  Total data size: %llu bytes\n", (unsigned long long)(header.vector_count * header.dimension * sizeof(float)));
            if (header.compressed_size > 0) {
                double ratio = (double)header.original_size / header.compressed_size;
                printf("  Compression ratio: %.2fx\n", ratio);
            }
        }
    }
}

static void inspect_database(const char *path, int stats, int verify, int json) {
    (void)verify;  /* Not implemented for databases yet */

    /* Try to open database */
    GV_Database *db = gv_db_open(path, 0, GV_INDEX_TYPE_HNSW);
    if (!db) {
        if (json) {
            printf("{\"error\": \"Failed to open database\"}\n");
        } else {
            fprintf(stderr, "Error: Failed to open database\n");
        }
        return;
    }

    if (json) {
        printf("{\n");
        printf("  \"type\": \"database\",\n");
        printf("  \"vector_count\": %zu,\n", db->count);
        printf("  \"dimension\": %zu,\n", db->dimension);
        printf("  \"index_type\": %u,\n", db->index_type);
        printf("  \"memory_usage\": %zu\n", gv_db_get_memory_usage(db));
        printf("}\n");
    } else {
        printf("GigaVector Database\n");
        printf("  Vectors: %zu\n", db->count);
        printf("  Dimension: %zu\n", db->dimension);

        const char *index_type;
        switch (db->index_type) {
            case GV_INDEX_TYPE_KDTREE: index_type = "KD-Tree"; break;
            case GV_INDEX_TYPE_HNSW: index_type = "HNSW"; break;
            case GV_INDEX_TYPE_IVFPQ: index_type = "IVF-PQ"; break;
            case GV_INDEX_TYPE_SPARSE: index_type = "Sparse"; break;
            default: index_type = "Unknown"; break;
        }
        printf("  Index Type: %s\n", index_type);

        if (stats) {
            printf("\nDetailed Statistics:\n");
            printf("  Memory Usage: %zu bytes\n", gv_db_get_memory_usage(db));
            printf("  Data Size: %zu bytes\n", db->count * db->dimension * sizeof(float));
        }
    }

    gv_db_close(db);
}

int main(int argc, char *argv[]) {
    int stats = 0;
    int verify = 0;
    int json = 0;

    static struct option long_options[] = {
        {"stats",   no_argument, 0, 's'},
        {"verify",  no_argument, 0, 'v'},
        {"json",    no_argument, 0, 'j'},
        {"help",    no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "svjh", long_options, NULL)) != -1) {
        switch (opt) {
            case 's': stats = 1; break;
            case 'v': verify = 1; break;
            case 'j': json = 1; break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    if (optind >= argc) {
        fprintf(stderr, "Error: File path required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    const char *path = argv[optind];

    /* Detect file type */
    if (ends_with(path, ".gvb")) {
        inspect_backup(path, stats, verify, json);
    } else if (ends_with(path, ".gvdb") || ends_with(path, ".db")) {
        inspect_database(path, stats, verify, json);
    } else {
        /* Try to detect by magic */
        FILE *fp = fopen(path, "rb");
        if (!fp) {
            fprintf(stderr, "Error: Cannot open file: %s\n", path);
            return 1;
        }

        char magic[5];
        if (fread(magic, 1, 5, fp) == 5 && memcmp(magic, "GVBAK", 5) == 0) {
            fclose(fp);
            inspect_backup(path, stats, verify, json);
        } else {
            fclose(fp);
            inspect_database(path, stats, verify, json);
        }
    }

    return 0;
}
