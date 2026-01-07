#include <stdlib.h>
#include <string.h>

#include "gigavector/gv_metadata.h"

static char *gv_metadata_strdup(const char *src) {
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

int gv_vector_set_metadata(GV_Vector *vector, const char *key, const char *value) {
    if (vector == NULL || key == NULL || value == NULL) {
        return -1;
    }

    GV_Metadata *current = vector->metadata;
    while (current != NULL) {
        if (strcmp(current->key, key) == 0) {
            char *new_value = gv_metadata_strdup(value);
            if (new_value == NULL) {
                return -1;
            }
            free(current->value);
            current->value = new_value;
            return 0;
        }
        current = current->next;
    }

    GV_Metadata *new_meta = (GV_Metadata *)malloc(sizeof(GV_Metadata));
    if (new_meta == NULL) {
        return -1;
    }

    new_meta->key = gv_metadata_strdup(key);
    if (new_meta->key == NULL) {
        free(new_meta);
        return -1;
    }

    new_meta->value = gv_metadata_strdup(value);
    if (new_meta->value == NULL) {
        free(new_meta->key);
        free(new_meta);
        return -1;
    }

    new_meta->next = vector->metadata;
    vector->metadata = new_meta;
    return 0;
}

const char *gv_vector_get_metadata(const GV_Vector *vector, const char *key) {
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

int gv_vector_remove_metadata(GV_Vector *vector, const char *key) {
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
            free(current->key);
            free(current->value);
            free(current);
            return 0;
        }
        prev = current;
        current = current->next;
    }
    return 0;
}

void gv_vector_clear_metadata(GV_Vector *vector) {
    if (vector == NULL) {
        return;
    }

    GV_Metadata *current = vector->metadata;
    while (current != NULL) {
        GV_Metadata *next = current->next;
        free(current->key);
        free(current->value);
        free(current);
        current = next;
    }
    vector->metadata = NULL;
}

void gv_metadata_free(GV_Metadata *meta) {
    GV_Metadata *current = meta;
    while (current != NULL) {
        GV_Metadata *next = current->next;
        free(current->key);
        free(current->value);
        free(current);
        current = next;
    }
}

GV_Metadata *gv_metadata_from_keys_values(const char **keys, const char **values, size_t count) {
    GV_Metadata *head = NULL;
    GV_Metadata **tail = &head;
    for (size_t i = 0; i < count; i++) {
        GV_Metadata *item = malloc(sizeof(GV_Metadata));
        if (!item) {
            gv_metadata_free(head);
            return NULL;
        }
        item->key = strdup(keys[i]);
        if (!item->key) {
            free(item);
            gv_metadata_free(head);
            return NULL;
        }
        item->value = strdup(values[i]);
        if (!item->value) {
            free(item->key);
            free(item);
            gv_metadata_free(head);
            return NULL;
        }
        item->next = NULL;
        *tail = item;
        tail = &item->next;
    }
    return head;
}
