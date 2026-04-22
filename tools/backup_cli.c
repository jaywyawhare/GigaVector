/**
 * @file gv_backup_cli.c
 * @brief Command-line backup tool for GigaVector.
 *
 * Usage: gvbackup --source <db_path> --dest <backup_path> [options]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#include "gigavector/gv_backup.h"

static void print_usage(const char *program) {
    printf("GigaVector Backup Tool\n\n");
    printf("Usage: %s --source <db_path> --dest <backup_path> [options]\n\n", program);
    printf("Options:\n");
    printf("  -s, --source <path>     Source database file (required)\n");
    printf("  -d, --dest <path>       Destination backup file (required)\n");
    printf("  -c, --compress          Enable compression\n");
    printf("  -e, --encrypt <key>     Encrypt with password\n");
    printf("  -n, --no-verify         Skip verification after backup\n");
    printf("  -v, --verbose           Verbose output\n");
    printf("  -h, --help              Show this help message\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s --source /data/vectors.gvdb --dest /backups/vectors.gvb\n", program);
    printf("  %s -s db.gvdb -d backup.gvb --compress --encrypt mypassword\n", program);
}

static void progress_callback(size_t current, size_t total, void *user_data) {
    int verbose = *(int *)user_data;
    if (verbose && total > 0) {
        int percent = (int)((current * 100) / total);
        printf("\rProgress: %d%% (%zu / %zu vectors)", percent, current, total);
        fflush(stdout);
        if (current == total) {
            printf("\n");
        }
    }
}

int main(int argc, char *argv[]) {
    const char *source = NULL;
    const char *dest = NULL;
    const char *encrypt_key = NULL;
    int compress = 0;
    int no_verify = 0;
    int verbose = 0;

    static struct option long_options[] = {
        {"source",    required_argument, 0, 's'},
        {"dest",      required_argument, 0, 'd'},
        {"compress",  no_argument,       0, 'c'},
        {"encrypt",   required_argument, 0, 'e'},
        {"no-verify", no_argument,       0, 'n'},
        {"verbose",   no_argument,       0, 'v'},
        {"help",      no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "s:d:ce:nvh", long_options, NULL)) != -1) {
        switch (opt) {
            case 's': source = optarg; break;
            case 'd': dest = optarg; break;
            case 'c': compress = 1; break;
            case 'e': encrypt_key = optarg; break;
            case 'n': no_verify = 1; break;
            case 'v': verbose = 1; break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    if (!source || !dest) {
        fprintf(stderr, "Error: --source and --dest are required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    if (verbose) {
        printf("GigaVector Backup\n");
        printf("  Source: %s\n", source);
        printf("  Destination: %s\n", dest);
        printf("  Compression: %s\n", compress ? "Yes" : "No");
        printf("  Encryption: %s\n", encrypt_key ? "Yes" : "No");
        printf("\n");
    }

    /* Configure backup options */
    GV_BackupOptions options;
    gv_backup_options_init(&options);
    options.compression = compress ? GV_BACKUP_COMPRESS_ZLIB : GV_BACKUP_COMPRESS_NONE;
    options.encryption_key = encrypt_key;
    options.verify_after = !no_verify;

    /* Create backup */
    GV_BackupResult *result = gv_backup_create_from_file(
        source, dest, &options,
        verbose ? progress_callback : NULL,
        &verbose
    );

    if (!result) {
        fprintf(stderr, "Error: Backup failed (unknown error)\n");
        return 1;
    }

    if (!result->success) {
        fprintf(stderr, "Error: %s\n", result->error_message ? result->error_message : "Unknown error");
        gv_backup_result_free(result);
        return 1;
    }

    if (verbose) {
        printf("\nBackup completed successfully!\n");
        printf("  Vectors: %llu\n", (unsigned long long)result->vectors_processed);
        printf("  Bytes: %llu\n", (unsigned long long)result->bytes_processed);
        printf("  Time: %.2f seconds\n", result->elapsed_seconds);
    } else {
        printf("Backup created: %s\n", dest);
    }

    gv_backup_result_free(result);
    return 0;
}
