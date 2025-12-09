#include <stdio.h>
#include <stdlib.h>

#include "gigavector/gigavector.h"

int main(void) {
    GV_Database *db = gv_db_open("in-memory", 3);
    if (db == NULL) {
        fprintf(stderr, "Error: Failed to create database\n");
        return EXIT_FAILURE;
    }

    float v1[] = {1.0f, 2.0f, 3.0f};
    float v2[] = {4.0f, 1.5f, -0.5f};
    float v3[] = {0.0f, 0.0f, 0.0f};

    if (gv_db_add_vector(db, v1, 3) != 0) {
        fprintf(stderr, "Error: Failed to insert first vector\n");
        gv_db_close(db);
        return EXIT_FAILURE;
    }

    if (gv_db_add_vector(db, v2, 3) != 0) {
        fprintf(stderr, "Error: Failed to insert second vector\n");
        gv_db_close(db);
        return EXIT_FAILURE;
    }

    if (gv_db_add_vector(db, v3, 3) != 0) {
        fprintf(stderr, "Error: Failed to insert third vector\n");
        gv_db_close(db);
        return EXIT_FAILURE;
    }

    printf("Successfully inserted %zu vectors into KD-tree (root axis=%zu)\n",
           db->count, db->root ? db->root->axis : 0U);

    if (gv_db_save(db, "database.bin") != 0) {
        fprintf(stderr, "Error: Failed to save database to database.bin\n");
        gv_db_close(db);
        return EXIT_FAILURE;
    }

    printf("Successfully saved database to database.bin\n");
    gv_db_close(db);
    db = NULL;

    GV_Database *db2 = gv_db_open("database.bin", 3);
    if (db2 == NULL) {
        fprintf(stderr, "Error: Failed to load database from database.bin\n");
        return EXIT_FAILURE;
    }

    printf("Successfully loaded database with %zu vectors\n", db2->count);
    if (db2->count != 3) {
        fprintf(stderr, "Warning: Expected 3 vectors, got %zu\n", db2->count);
    }

    float query[] = {1.5f, 2.0f, 2.5f};
    GV_SearchResult results[3];
    int found = gv_db_search(db2, query, 3, results, GV_DISTANCE_EUCLIDEAN);
    if (found < 0) {
        fprintf(stderr, "Error: Search failed\n");
        gv_db_close(db2);
        return EXIT_FAILURE;
    }

    printf("\nEuclidean distance search results (query: [%.2f, %.2f, %.2f]):\n",
           query[0], query[1], query[2]);
    for (int i = 0; i < found; ++i) {
        printf("  %d. Distance: %.4f, Vector: [", i + 1, results[i].distance);
        for (size_t j = 0; j < db2->dimension; ++j) {
            printf("%s%.2f", (j == 0) ? "" : ", ", results[i].vector->data[j]);
        }
        printf("]\n");
    }

    found = gv_db_search(db2, query, 3, results, GV_DISTANCE_COSINE);
    if (found < 0) {
        fprintf(stderr, "Error: Cosine search failed\n");
        gv_db_close(db2);
        return EXIT_FAILURE;
    }

    printf("\nCosine similarity search results:\n");
    for (int i = 0; i < found; ++i) {
        float similarity = 1.0f - results[i].distance;
        printf("  %d. Similarity: %.4f, Vector: [", i + 1, similarity);
        for (size_t j = 0; j < db2->dimension; ++j) {
            printf("%s%.2f", (j == 0) ? "" : ", ", results[i].vector->data[j]);
        }
        printf("]\n");
    }

    gv_db_close(db2);
    printf("\nDatabase lifecycle completed successfully\n");
    return EXIT_SUCCESS;
}