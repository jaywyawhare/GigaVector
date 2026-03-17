#ifndef GV_UTILS_H
#define GV_UTILS_H

#include <stdlib.h>
#include <string.h>

static inline char *gv_strdup(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char *copy = (char *)malloc(len);
    if (!copy) return NULL;
    memcpy(copy, s, len);
    return copy;
}

#endif /* GV_UTILS_H */
