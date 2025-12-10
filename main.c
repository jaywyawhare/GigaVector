#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <time.h>

#include "gigavector/gigavector.h"

static const char *gv_demo_data_dir(void) {
    const char *dir = getenv("GV_DATA_DIR");
    return (dir && dir[0] != '\0') ? dir : "snapshots";
}

static int gv_demo_mkpath(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        return S_ISDIR(st.st_mode) ? 0 : -1;
    }
    if (errno != ENOENT) {
        return -1;
    }
    return mkdir(path, 0755);
}

static int gv_demo_join(char *out, size_t out_sz, const char *dir, const char *name) {
    int n = snprintf(out, out_sz, "%s/%s", dir, name);
    return (n > 0 && (size_t)n < out_sz) ? 0 : -1;
}

static void gv_demo_usage(const char *prog) {
    printf("Usage: %s [--index {kdtree|hnsw|ivfpq}] [--dim N]\n", prog);
    printf("             [--ivf-nlist N] [--ivf-m N] [--ivf-nbits N]\n");
    printf("             [--ivf-nprobe N] [--ivf-rerank N] [--ivf-cosine]\n");
    printf("\nDefaults: kdtree index, dim=3; IVF-PQ defaults only used when --index ivfpq.\n");
}

static void gv_demo_fill_random(float *data, size_t count, size_t dim) {
    for (size_t i = 0; i < count * dim; ++i) {
        data[i] = (float)rand() / (float)RAND_MAX;
    }
}

int main(int argc, char **argv) {
    printf("=== GigaVector Database Demo ===\n\n");

    GV_IndexType index_type = GV_INDEX_TYPE_KDTREE;
    size_t dim = 3;
    size_t ivf_nlist = 256;
    size_t ivf_m = 8;
    uint8_t ivf_nbits = 8;
    size_t ivf_nprobe = 16;
    size_t ivf_rerank = 32;
    int ivf_cosine = 0;

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (strcmp(arg, "--help") == 0) {
            gv_demo_usage(argv[0]);
            return 0;
        } else if (strcmp(arg, "--index") == 0 && i + 1 < argc) {
            const char *v = argv[++i];
            if (strcmp(v, "kdtree") == 0) index_type = GV_INDEX_TYPE_KDTREE;
            else if (strcmp(v, "hnsw") == 0) index_type = GV_INDEX_TYPE_HNSW;
            else if (strcmp(v, "ivfpq") == 0) { index_type = GV_INDEX_TYPE_IVFPQ; dim = 64; }
        } else if (strcmp(arg, "--dim") == 0 && i + 1 < argc) {
            dim = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(arg, "--ivf-nlist") == 0 && i + 1 < argc) {
            ivf_nlist = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(arg, "--ivf-m") == 0 && i + 1 < argc) {
            ivf_m = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(arg, "--ivf-nbits") == 0 && i + 1 < argc) {
            ivf_nbits = (uint8_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(arg, "--ivf-nprobe") == 0 && i + 1 < argc) {
            ivf_nprobe = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(arg, "--ivf-rerank") == 0 && i + 1 < argc) {
            ivf_rerank = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(arg, "--ivf-cosine") == 0) {
            ivf_cosine = 1;
        } else {
            fprintf(stderr, "Unknown or incomplete option: %s\n", arg);
            gv_demo_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    srand((unsigned int)time(NULL));

    const char *data_dir = gv_demo_data_dir();
    if (gv_demo_mkpath(data_dir) != 0) {
        fprintf(stderr, "Error: Failed to create data dir %s\n", data_dir);
        return EXIT_FAILURE;
    }
    if (getenv("GV_WAL_DIR") == NULL) {
        setenv("GV_WAL_DIR", data_dir, 1);
    }

    char db_path[512];
    if (gv_demo_join(db_path, sizeof(db_path), data_dir,
                     (index_type == GV_INDEX_TYPE_KDTREE) ? "database.bin" :
                     (index_type == GV_INDEX_TYPE_HNSW) ? "hnsw_database.bin" :
                     "ivfpq_database.bin") != 0) {
        fprintf(stderr, "Error: Path construction failed\n");
        return EXIT_FAILURE;
    }

    printf("Index: %s | dim=%zu\n", index_type == GV_INDEX_TYPE_KDTREE ? "kdtree" :
                                   index_type == GV_INDEX_TYPE_HNSW ? "hnsw" : "ivfpq", dim);
    GV_Database *db = gv_db_open(db_path, dim, index_type);
    if (db == NULL) {
        fprintf(stderr, "Error: Failed to create database (check WAL/index compatibility)\n");
        return EXIT_FAILURE;
    }

    if (index_type == GV_INDEX_TYPE_IVFPQ) {
        GV_IVFPQConfig cfg = {.nlist = ivf_nlist, .m = ivf_m, .nbits = ivf_nbits,
                              .nprobe = ivf_nprobe, .train_iters = 20,
                              .default_rerank = ivf_rerank, .use_cosine = ivf_cosine};
        gv_ivfpq_destroy(db->hnsw_index);
        db->hnsw_index = gv_ivfpq_create(dim, &cfg);
        if (!db->hnsw_index) {
            fprintf(stderr, "Error: IVF-PQ create failed\n");
            gv_db_close(db);
            return EXIT_FAILURE;
        }
    }

    size_t train_count = (index_type == GV_INDEX_TYPE_IVFPQ) ? 2048 : 0;
    size_t vec_count = 16;
    float *train = NULL;
    if (train_count > 0) {
        train = (float *)malloc(train_count * dim * sizeof(float));
        if (!train) {
            gv_db_close(db);
            return EXIT_FAILURE;
        }
        gv_demo_fill_random(train, train_count, dim);
        if (gv_ivfpq_train(db->hnsw_index, train, train_count) != 0) {
            fprintf(stderr, "Error: IVF-PQ training failed\n");
            free(train);
            gv_db_close(db);
            return EXIT_FAILURE;
        }
    }

    float *data = (float *)malloc(vec_count * dim * sizeof(float));
    if (!data) {
        free(train);
        gv_db_close(db);
        return EXIT_FAILURE;
    }
    gv_demo_fill_random(data, vec_count, dim);
    for (size_t i = 0; i < vec_count; ++i) {
        char idbuf[32];
        snprintf(idbuf, sizeof(idbuf), "%zu", i);
        if (gv_db_add_vector_with_metadata(db, data + i * dim, dim, "id", idbuf) != 0) {
            fprintf(stderr, "Error: insert failed at %zu\n", i);
            free(data);
            free(train);
            gv_db_close(db);
            return EXIT_FAILURE;
        }
    }

    printf("Inserted %zu vectors%s.\n", vec_count,
           index_type == GV_INDEX_TYPE_IVFPQ ? " (IVF-PQ trained)" : "");

    float *qbuf = (float *)malloc(dim * sizeof(float));
    if (!qbuf) {
        free(data);
        free(train);
        gv_db_close(db);
        return EXIT_FAILURE;
    }
    gv_demo_fill_random(qbuf, 1, dim);

    GV_SearchResult results[5] = {0};
    int found = -1;
    if (index_type == GV_INDEX_TYPE_IVFPQ) {
        found = gv_db_search_ivfpq_opts(db, qbuf, 5, results,
                                        ivf_cosine ? GV_DISTANCE_COSINE : GV_DISTANCE_EUCLIDEAN,
                                        ivf_nprobe, ivf_rerank);
    } else {
        found = gv_db_search(db, qbuf, 5, results, GV_DISTANCE_EUCLIDEAN);
    }

    if (found < 0) {
        fprintf(stderr, "Error: Search failed\n");
        free(qbuf);
        free(data);
        free(train);
        gv_db_close(db);
        return EXIT_FAILURE;
    }

    printf("\nSearch results (top %d):\n", found);
    for (int i = 0; i < found; ++i) {
        printf("Rank %d: Distance = %f\n", i + 1, results[i].distance);
    }

    printf("\nSaving database to %s\n", db_path);
    if (gv_db_save(db, NULL) != 0) {
        fprintf(stderr, "Error: Failed to save database\n");
        free(qbuf);
        free(data);
        free(train);
        gv_db_close(db);
        return EXIT_FAILURE;
    }
    printf("Database saved successfully.\n");

    free(qbuf);
    free(data);
    free(train);
    gv_db_close(db);
    printf("\nDemo completed successfully.\n");
    return EXIT_SUCCESS;
}

