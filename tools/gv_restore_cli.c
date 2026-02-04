/**
 * @file gv_restore_cli.c
 * @brief Command-line restore tool for GigaVector.
 *
 * Usage: gvrestore --source <backup_path> --dest <db_path> [options]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#include "gigavector/gv_backup.h"

static void print_usage(const char *program) {
    printf("GigaVector Restore Tool\n\n");
    printf("Usage: %s --source <backup_path> --dest <db_path> [options]\n\n", program);
    printf("Options:\n");
    printf("  -s, --source <path>     Source backup file (required)\n");
    printf("  -d, --dest <path>       Destination database file (required)\n");
    printf("  -k, --key <password>    Decryption password (if encrypted)\n");
    printf("  -f, --force             Overwrite existing destination\n");
    printf("  -n, --no-verify         Skip checksum verification\n");
    printf("  -v, --verbose           Verbose output\n");
    printf("  -h, --help              Show this help message\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s --source /backups/vectors.gvb --dest /data/vectors.gvdb\n", program);
    printf("  %s -s backup.gvb -d db.gvdb --key mypassword --force\n", program);
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
    const char *decrypt_key = NULL;
    int force = 0;
    int no_verify = 0;
    int verbose = 0;

    static struct option long_options[] = {
        {"source",    required_argument, 0, 's'},
        {"dest",      required_argument, 0, 'd'},
        {"key",       required_argument, 0, 'k'},
        {"force",     no_argument,       0, 'f'},
        {"no-verify", no_argument,       0, 'n'},
        {"verbose",   no_argument,       0, 'v'},
        {"help",      no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "s:d:k:fnvh", long_options, NULL)) != -1) {
        switch (opt) {
            case 's': source = optarg; break;
            case 'd': dest = optarg; break;
            case 'k': decrypt_key = optarg; break;
            case 'f': force = 1; break;
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
        printf("GigaVector Restore\n");
        printf("  Source: %s\n", source);
        printf("  Destination: %s\n", dest);
        printf("  Overwrite: %s\n", force ? "Yes" : "No");
        printf("\n");
    }

    /* Configure restore options */
    GV_RestoreOptions options;
    gv_restore_options_init(&options);
    options.overwrite = force;
    options.verify_checksum = !no_verify;
    options.decryption_key = decrypt_key;

    /* Restore backup */
    GV_BackupResult *result = gv_backup_restore(
        source, dest, &options,
        verbose ? progress_callback : NULL,
        &verbose
    );

    if (!result) {
        fprintf(stderr, "Error: Restore failed (unknown error)\n");
        return 1;
    }

    if (!result->success) {
        fprintf(stderr, "Error: %s\n", result->error_message ? result->error_message : "Unknown error");
        gv_backup_result_free(result);
        return 1;
    }

    if (verbose) {
        printf("\nRestore completed successfully!\n");
        printf("  Vectors: %llu\n", (unsigned long long)result->vectors_processed);
        printf("  Bytes: %llu\n", (unsigned long long)result->bytes_processed);
        printf("  Time: %.2f seconds\n", result->elapsed_seconds);
    } else {
        printf("Database restored: %s\n", dest);
    }

    gv_backup_result_free(result);
    return 0;
}
