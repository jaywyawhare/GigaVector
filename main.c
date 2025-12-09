#include <stdio.h>

#include "gigavector/gigavector.h"

int main(void) {
    GV_Database *db = gv_db_open("in-memory", 3);
    if (db == NULL) {
        fprintf(stderr, "Failed to create database\n");
        return 1;
    }

    float v1[] = {1.0f, 2.0f, 3.0f};
    float v2[] = {4.0f, 1.5f, -0.5f};

    if (gv_db_add_vector(db, v1, 3) != 0 || gv_db_add_vector(db, v2, 3) != 0) {
        fprintf(stderr, "Failed to insert vectors\n");
        gv_db_close(db);
        return 1;
    }

    printf("Inserted %zu vectors into KD-tree (root axis=%zu)\n", db->count, db->root ? db->root->axis : 0U);

    if (gv_db_save(db, "database.bin") != 0) {
        fprintf(stderr, "Failed to save database\n");
        gv_db_close(db);
        return 1;
    }

    printf("Saved database to database.bin\n");

    gv_db_close(db);

    GV_Database *db2 = gv_db_open("database.bin", 3);
    if (db2 == NULL) {
        fprintf(stderr, "Failed to open database\n");
        return 1;
    }
    printf("Opened database with %zu vectors\n", db2->count);
    gv_db_close(db2);
    return 0;
}