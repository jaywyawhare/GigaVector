#include <stdlib.h>
#include "core/memory.h"
#include <string.h>

#include "schema/metadata.h"
#include "core/utils.h"

int vector_set_metadata(GV_Vector *vector, const char *key, const char *value) {
    if (vector == NULL || key == NULL || value == NULL) {
        return -1;
    }

    GV_Metadata *current = vector->metadata;
    while (current != NULL) {
        if (strcmp(current->key, key) == 0) {
            char *new_value = gv_dup_cstr(value);
            if (new_value == NULL) {
                return -1;
            }
            gv_free(current->value);
            current->value = new_value;
            return 0;
        }
        current = current->next;
    }

    GV_Metadata *new_meta = (GV_Metadata *)gv_alloc(sizeof(GV_Metadata));
    if (new_meta == NULL) {
        return -1;
    }

    new_meta->key = gv_dup_cstr(key);
    if (new_meta->key == NULL) {
        gv_free(new_meta);
        return -1;
    }

    new_meta->value = gv_dup_cstr(value);
    if (new_meta->value == NULL) {
        gv_free(new_meta->key);
        gv_free(new_meta);
        return -1;
    }

    new_meta->next = vector->metadata;
    vector->metadata = new_meta;
    return 0;
}

const char *vector_get_metadata(const GV_Vector *vector, const char *key) {
    if (vector == NULL || key == NULL) {
        return NULL;
    }

    GV_Metadata *current = vector->metadata;
    while (current != NULL) {
        if (strcmp(current->key, key) == 0) {
            return current->value;
        }
        current = current->next;
    }
    return NULL;
}

int vector_remove_metadata(GV_Vector *vector, const char *key) {
    if (vector == NULL || key == NULL) {
        return -1;
    }

    GV_Metadata *current = vector->metadata;
    GV_Metadata *prev = NULL;

    while (current != NULL) {
        if (strcmp(current->key, key) == 0) {
            if (prev == NULL) {
                vector->metadata = current->next;
            } else {
                prev->next = current->next;
            }
            gv_free(current->key);
            gv_free(current->value);
            gv_free(current);
            return 0;
        }
        prev = current;
        current = current->next;
    }
    return 0;
}

void vector_clear_metadata(GV_Vector *vector) {
    if (vector == NULL) {
        return;
    }

    GV_Metadata *current = vector->metadata;
    while (current != NULL) {
        GV_Metadata *next = current->next;
        gv_free(current->key);
        gv_free(current->value);
        gv_free(current);
        current = next;
    }
    vector->metadata = NULL;
}

void metadata_free(GV_Metadata *meta) {
    GV_Metadata *current = meta;
    while (current != NULL) {
        GV_Metadata *next = current->next;
        gv_free(current->key);
        gv_free(current->value);
        gv_free(current);
        current = next;
    }
}

GV_Metadata *metadata_from_keys_values(const char **keys, const char **values, size_t count) {
    GV_Metadata *head = NULL;
    GV_Metadata **tail = &head;
    for (size_t i = 0; i < count; i++) {
        GV_Metadata *item = gv_alloc(sizeof(GV_Metadata));
        if (!item) {
            metadata_free(head);
            return NULL;
        }
        item->key = gv_dup_cstr(keys[i]);
        if (!item->key) {
            gv_free(item);
            metadata_free(head);
            return NULL;
        }
        item->value = gv_dup_cstr(values[i]);
        if (!item->value) {
            gv_free(item->key);
            gv_free(item);
            metadata_free(head);
            return NULL;
        }
        item->next = NULL;
        *tail = item;
        tail = &item->next;
    }
    return head;
}

int read_metadata_into_vector(FILE *in, GV_Vector *vector) {
    uint32_t count = 0;
    if (read_u32(in, &count) != 0) return -1;
    for (uint32_t i = 0; i < count; i++) {
        char *key = read_string(in);
        char *value = read_string(in);
        int rc = (key && value) ? vector_set_metadata(vector, key, value) : -1;
        gv_free(key);
        gv_free(value);
        if (rc != 0) return -1;
    }
    return 0;
}

void vector_apply_metadata(GV_Vector *dst, const GV_Metadata *src) {
    if (!dst) return;
    for (const GV_Metadata *m = src; m; m = m->next) {
        if (m->key && m->value) vector_set_metadata(dst, m->key, m->value);
    }
}

void vector_copy_metadata(GV_Vector *dst, const GV_Vector *src) {
    if (src) vector_apply_metadata(dst, src->metadata);
}
