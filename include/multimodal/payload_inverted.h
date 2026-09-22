#ifndef GIGAVECTOR_GV_PAYLOAD_INVERTED_H
#define GIGAVECTOR_GV_PAYLOAD_INVERTED_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "core/id_bitmap.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GV_PayloadInvertedIndex GV_PayloadInvertedIndex;

GV_PayloadInvertedIndex *payload_inverted_create(void);
void payload_inverted_destroy(GV_PayloadInvertedIndex *idx);

int payload_inverted_add_field(GV_PayloadInvertedIndex *idx, const char *field, int type);

int payload_inverted_index(GV_PayloadInvertedIndex *idx, const char *field,
                           uint64_t vector_id, const char *value);

int payload_inverted_remove(GV_PayloadInvertedIndex *idx, uint64_t vector_id);

GV_IdBitmap *payload_inverted_find(const GV_PayloadInvertedIndex *idx,
                                   const char *field, const char *value);

GV_IdBitmap *payload_inverted_find_prefix(const GV_PayloadInvertedIndex *idx,
                                          const char *field, const char *prefix);

GV_IdBitmap *payload_inverted_find_range(const GV_PayloadInvertedIndex *idx,
                                         const char *field,
                                         double min_val, double max_val);

int payload_inverted_save(const GV_PayloadInvertedIndex *idx, FILE *out);
int payload_inverted_load(GV_PayloadInvertedIndex **idx, FILE *in);

size_t payload_inverted_field_count(const GV_PayloadInvertedIndex *idx);
size_t payload_inverted_entry_count(const GV_PayloadInvertedIndex *idx);

#ifdef __cplusplus
}
#endif
#endif
