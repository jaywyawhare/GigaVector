#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gigavector/gv_database.h"
#include "gigavector/gv_kdtree.h"
#include "gigavector/gv_vector.h"

static char *gv_db_strdup(const char *src) {
    if (src == NULL) {
        return NULL;
    }
    size_t len = strlen(src) + 1;
    char *copy = (char *)malloc(len);
    if (copy == NULL) {
        return NULL;
    }
    memcpy(copy, src, len);
    return copy;
}

static int gv_db_write_header(FILE *out, uint32_t dimension, uint64_t count) {
    const uint32_t magic = 0x47564442; /* "GVDB" in hex */
    const uint32_t version = 1;
    if (fwrite(&magic, sizeof(uint32_t), 1, out) != 1) {
        return -1;
    }
    if (fwrite(&version, sizeof(uint32_t), 1, out) != 1) {
        return -1;
    }
    if (fwrite(&dimension, sizeof(uint32_t), 1, out) != 1) {
        return -1;
    }
    if (fwrite(&count, sizeof(uint64_t), 1, out) != 1) {
        return -1;
    }
    return 0;
}

static int gv_db_read_header(FILE *in, uint32_t *dimension_out, uint64_t *count_out) {
    uint32_t magic = 0;
    uint32_t version = 0;
    if (fread(&magic, sizeof(uint32_t), 1, in) != 1) {
        return -1;
    }
    if (fread(&version, sizeof(uint32_t), 1, in) != 1) {
        return -1;
    }
    if (magic != 0x47564442 /* "GVDB" */ || version != 1) {
        return -1;
    }
    if (fread(dimension_out, sizeof(uint32_t), 1, in) != 1) {
        return -1;
    }
    if (fread(count_out, sizeof(uint64_t), 1, in) != 1) {
        return -1;
    }
    return 0;
}

GV_Database *gv_db_open(const char *filepath, size_t dimension) {
    if (dimension == 0 && filepath == NULL) {
        return NULL;
    }

    GV_Database *db = (GV_Database *)malloc(sizeof(GV_Database));
    if (db == NULL) {
        return NULL;
    }

    db->dimension = dimension;
    db->root = NULL;
    db->filepath = NULL;
    db->count = 0;

    if (filepath != NULL) {
        db->filepath = gv_db_strdup(filepath);
        if (db->filepath == NULL) {
            free(db);
            return NULL;
        }
    }

    if (filepath == NULL) {
        return db;
    }

    FILE *in = fopen(filepath, "rb");
    if (in == NULL) {
        if (errno == ENOENT) {
            return db;
        }
        free(db->filepath);
        free(db);
        return NULL;
    }

    uint32_t file_dim = 0;
    uint64_t file_count = 0;
    if (gv_db_read_header(in, &file_dim, &file_count) != 0) {
        fclose(in);
        free(db->filepath);
        free(db);
        return NULL;
    }

    if (dimension != 0 && dimension != (size_t)file_dim) {
        fclose(in);
        free(db->filepath);
        free(db);
        return NULL;
    }

    db->dimension = (size_t)file_dim;

    if (gv_kdtree_load_recursive(&(db->root), in, db->dimension) != 0) {
        fclose(in);
        gv_kdtree_destroy_recursive(db->root);
        free(db->filepath);
        free(db);
        return NULL;
    }

    if (fclose(in) != 0) {
        gv_kdtree_destroy_recursive(db->root);
        free(db->filepath);
        free(db);
        return NULL;
    }

    db->count = file_count;
    return db;
}

void gv_db_close(GV_Database *db) {
    if (db == NULL) {
        return;
    }

    gv_kdtree_destroy_recursive(db->root);
    free(db->filepath);
    free(db);
}

int gv_db_add_vector(GV_Database *db, const float *data, size_t dimension) {
    if (db == NULL || data == NULL || dimension == 0 || dimension != db->dimension) {
        return -1;
    }

    GV_Vector *vector = gv_vector_create_from_data(dimension, data);
    if (vector == NULL) {
        return -1;
    }

    if (gv_kdtree_insert(&(db->root), vector, 0) != 0) {
        gv_vector_destroy(vector);
        return -1;
    }

    db->count += 1;
    return 0;
}

int gv_db_save(const GV_Database *db, const char *filepath) {
    if (db == NULL) {
        return -1;
    }

    const char *out_path = filepath != NULL ? filepath : db->filepath;
    if (out_path == NULL) {
        return -1;
    }

    if (db->dimension == 0 || db->dimension > UINT32_MAX) {
        return -1;
    }

    FILE *out = fopen(out_path, "wb");
    if (out == NULL) {
        return -1;
    }

    int status = gv_db_write_header(out, (uint32_t)db->dimension, db->count);
    if (status == 0) {
        status = gv_kdtree_save_recursive(db->root, out);
    }

    if (fclose(out) != 0) {
        status = -1;
    }

    if (status != 0) {
        return -1;
    }

    return 0;
}

