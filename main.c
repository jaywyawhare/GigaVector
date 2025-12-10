#include <stdio.h>
#include <stdlib.h>

#include "gigavector/gigavector.h"

int main(void) {
    printf("=== GigaVector Database Demo ===\n\n");
    
    printf("1. Testing KD-Tree Index:\n");
    printf("------------------------\n");
    GV_Database *db = gv_db_open("in-memory", 3, GV_INDEX_TYPE_KDTREE);
    if (db == NULL) {
        fprintf(stderr, "Error: Failed to create database\n");
        return EXIT_FAILURE;
    }

    float v1[] = {1.0f, 2.0f, 3.0f};
    float v2[] = {4.0f, 1.5f, -0.5f};
    float v3[] = {0.0f, 0.0f, 0.0f};
    float v4[] = {2.0f, 2.5f, 3.5f};
    float v5[] = {5.0f, 0.0f, 1.0f};

    if (gv_db_add_vector_with_metadata(db, v1, 3, "category", "A") != 0) {
        fprintf(stderr, "Error: Failed to insert first vector\n");
        gv_db_close(db);
        return EXIT_FAILURE;
    }

    if (gv_db_add_vector_with_metadata(db, v2, 3, "category", "B") != 0) {
        fprintf(stderr, "Error: Failed to insert second vector\n");
        gv_db_close(db);
        return EXIT_FAILURE;
    }

    if (gv_db_add_vector_with_metadata(db, v3, 3, "category", "A") != 0) {
        fprintf(stderr, "Error: Failed to insert third vector\n");
        gv_db_close(db);
        return EXIT_FAILURE;
    }

    if (gv_db_add_vector_with_metadata(db, v4, 3, "category", "A") != 0) {
        fprintf(stderr, "Error: Failed to insert fourth vector\n");
        gv_db_close(db);
        return EXIT_FAILURE;
    }

    if (gv_db_add_vector_with_metadata(db, v5, 3, "category", "B") != 0) {
        fprintf(stderr, "Error: Failed to insert fifth vector\n");
        gv_db_close(db);
        return EXIT_FAILURE;
    }

    printf("Successfully inserted %zu vectors with metadata into KD-tree (root axis=%zu)\n",
           db->count, db->root ? db->root->axis : 0U);

    if (gv_db_save(db, "database.bin") != 0) {
        fprintf(stderr, "Error: Failed to save database to database.bin\n");
        gv_db_close(db);
        return EXIT_FAILURE;
    }

    printf("Successfully saved database to database.bin\n");

    GV_Database *db2 = gv_db_open("database.bin", 3, GV_INDEX_TYPE_KDTREE);
    if (db2 == NULL) {
        fprintf(stderr, "Error: Failed to load database from database.bin\n");
        gv_db_close(db);
        return EXIT_FAILURE;
    }

    printf("Successfully loaded database with %zu vectors\n", db2->count);
    if (db2->count != 5) {
        fprintf(stderr, "Warning: Expected 5 vectors, got %zu\n", db2->count);
    }

    float query[] = {1.5f, 2.0f, 2.5f};
    GV_SearchResult results[5];
    int found = gv_db_search(db2, query, 3, results, GV_DISTANCE_EUCLIDEAN);
    if (found < 0) {
        fprintf(stderr, "Error: Search failed\n");
        gv_db_close(db);
        gv_db_close(db2);
        return EXIT_FAILURE;
    }

    printf("\nEuclidean distance search results (query: [%.2f, %.2f, %.2f]):\n",
           query[0], query[1], query[2]);
    for (int i = 0; i < found; ++i) {
        const char *category = gv_vector_get_metadata(results[i].vector, "category");
        printf("  %d. Distance: %.4f, Vector: [", i + 1, results[i].distance);
        for (size_t j = 0; j < db2->dimension; ++j) {
            printf("%s%.2f", (j == 0) ? "" : ", ", results[i].vector->data[j]);
        }
        printf("], Category: %s\n", category ? category : "none");
    }

    found = gv_db_search(db2, query, 5, results, GV_DISTANCE_COSINE);
    if (found < 0) {
        fprintf(stderr, "Error: Cosine search failed\n");
        gv_db_close(db);
        gv_db_close(db2);
        return EXIT_FAILURE;
    }

    printf("\nCosine similarity search results:\n");
    for (int i = 0; i < found; ++i) {
        float similarity = 1.0f - results[i].distance;
        const char *category = gv_vector_get_metadata(results[i].vector, "category");
        printf("  %d. Similarity: %.4f, Vector: [", i + 1, similarity);
        for (size_t j = 0; j < db2->dimension; ++j) {
            printf("%s%.2f", (j == 0) ? "" : ", ", results[i].vector->data[j]);
        }
        printf("], Category: %s\n", category ? category : "none");
    }

    found = gv_db_search_filtered(db2, query, 5, results, GV_DISTANCE_EUCLIDEAN,
                                   "category", "A");
    if (found < 0) {
        fprintf(stderr, "Error: Filtered search failed\n");
        gv_db_close(db);
        gv_db_close(db2);
        return EXIT_FAILURE;
    }

    printf("\nFiltered search (category='A') results:\n");
    for (int i = 0; i < found; ++i) {
        const char *category = gv_vector_get_metadata(results[i].vector, "category");
        printf("  %d. Distance: %.4f, Vector: [", i + 1, results[i].distance);
        for (size_t j = 0; j < db2->dimension; ++j) {
            printf("%s%.2f", (j == 0) ? "" : ", ", results[i].vector->data[j]);
        }
        printf("], Category: %s\n", category ? category : "none");
    }

    gv_db_close(db);
    gv_db_close(db2);
    printf("\nKD-Tree database lifecycle completed successfully\n\n");

    printf("2. Testing HNSW Index:\n");
    printf("----------------------\n");
    GV_Database *db_hnsw = gv_db_open("hnsw-demo", 3, GV_INDEX_TYPE_HNSW);
    if (db_hnsw == NULL) {
        fprintf(stderr, "Error: Failed to create HNSW database\n");
        return EXIT_FAILURE;
    }

    float v1_hnsw[] = {1.0f, 2.0f, 3.0f};
    float v2_hnsw[] = {4.0f, 1.5f, -0.5f};
    float v3_hnsw[] = {0.0f, 0.0f, 0.0f};
    float v4_hnsw[] = {2.0f, 2.5f, 3.5f};
    float v5_hnsw[] = {5.0f, 0.0f, 1.0f};

    if (gv_db_add_vector_with_metadata(db_hnsw, v1_hnsw, 3, "category", "A") != 0 ||
        gv_db_add_vector_with_metadata(db_hnsw, v2_hnsw, 3, "category", "B") != 0 ||
        gv_db_add_vector_with_metadata(db_hnsw, v3_hnsw, 3, "category", "A") != 0 ||
        gv_db_add_vector_with_metadata(db_hnsw, v4_hnsw, 3, "category", "A") != 0 ||
        gv_db_add_vector_with_metadata(db_hnsw, v5_hnsw, 3, "category", "B") != 0) {
        fprintf(stderr, "Error: Failed to insert vectors into HNSW\n");
        gv_db_close(db_hnsw);
        return EXIT_FAILURE;
    }

    printf("Successfully inserted %zu vectors into HNSW index\n", db_hnsw->count);

    if (gv_db_save(db_hnsw, "hnsw_database.bin") != 0) {
        fprintf(stderr, "Error: Failed to save HNSW database\n");
        gv_db_close(db_hnsw);
        return EXIT_FAILURE;
    }
    printf("Successfully saved HNSW database to hnsw_database.bin\n");

    GV_Database *db_hnsw2 = gv_db_open("hnsw_database.bin", 3, GV_INDEX_TYPE_HNSW);
    if (db_hnsw2 == NULL) {
        fprintf(stderr, "Error: Failed to load HNSW database from file\n");
        gv_db_close(db_hnsw);
        return EXIT_FAILURE;
    }
    printf("Successfully loaded HNSW database with %zu vectors\n\n", db_hnsw2->count);

    float query_hnsw[] = {1.5f, 2.0f, 2.5f};
    GV_SearchResult results_hnsw[5];
    int found_hnsw = gv_db_search(db_hnsw2, query_hnsw, 5, results_hnsw, GV_DISTANCE_EUCLIDEAN);
    
    if (found_hnsw < 0) {
        fprintf(stderr, "Error: HNSW search failed (returned %d)\n", found_hnsw);
        gv_db_close(db_hnsw);
        return EXIT_FAILURE;
    }
    
    if (found_hnsw == 0) {
        printf("HNSW search returned 0 results (index may be empty or search failed)\n");
        gv_db_close(db_hnsw);
        return EXIT_FAILURE;
    }

    printf("\nHNSW Search Results (k=5, query: [%.2f, %.2f, %.2f]):\n",
           query_hnsw[0], query_hnsw[1], query_hnsw[2]);
    for (int i = 0; i < found_hnsw; ++i) {
        const char *category = gv_vector_get_metadata(results_hnsw[i].vector, "category");
        printf("  %d. Distance: %.4f, Vector: [", i + 1, results_hnsw[i].distance);
        for (size_t j = 0; j < db_hnsw->dimension; ++j) {
            printf("%s%.2f", (j == 0) ? "" : ", ", results_hnsw[i].vector->data[j]);
        }
        printf("], Category: %s\n", category ? category : "none");
    }

    found_hnsw = gv_db_search_filtered(db_hnsw2, query_hnsw, 5, results_hnsw, GV_DISTANCE_EUCLIDEAN,
                                       "category", "A");
    if (found_hnsw >= 0) {
        printf("\nHNSW Filtered Search (category='A'):\n");
        for (int i = 0; i < found_hnsw; ++i) {
            const char *category = gv_vector_get_metadata(results_hnsw[i].vector, "category");
            printf("  %d. Distance: %.4f, Vector: [", i + 1, results_hnsw[i].distance);
            for (size_t j = 0; j < db_hnsw->dimension; ++j) {
                printf("%s%.2f", (j == 0) ? "" : ", ", results_hnsw[i].vector->data[j]);
            }
            printf("], Category: %s\n", category ? category : "none");
        }
    }

    gv_db_close(db_hnsw);
    gv_db_close(db_hnsw2);
    printf("\nHNSW database lifecycle completed successfully\n");
    printf("\n=== All tests completed successfully ===\n");
    return EXIT_SUCCESS;
}