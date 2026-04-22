#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "specialized/point_id.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_point_id_create_destroy(void) {
    GV_PointIDMap *map = point_id_create(0);
    ASSERT(map != NULL, "point ID map creation with default capacity");
    point_id_destroy(map);

    map = point_id_create(128);
    ASSERT(map != NULL, "point ID map creation with capacity=128");
    point_id_destroy(map);

    point_id_destroy(NULL);
    return 0;
}

static int test_point_id_set_and_get(void) {
    GV_PointIDMap *map = point_id_create(0);
    ASSERT(map != NULL, "map creation");

    ASSERT(point_id_set(map, "vec-001", 0) == 0, "set vec-001 -> 0");
    ASSERT(point_id_set(map, "vec-002", 1) == 0, "set vec-002 -> 1");
    ASSERT(point_id_set(map, "vec-003", 2) == 0, "set vec-003 -> 2");

    size_t idx;
    ASSERT(point_id_get(map, "vec-001", &idx) == 0, "get vec-001");
    ASSERT(idx == 0, "vec-001 maps to 0");

    ASSERT(point_id_get(map, "vec-002", &idx) == 0, "get vec-002");
    ASSERT(idx == 1, "vec-002 maps to 1");

    ASSERT(point_id_get(map, "vec-003", &idx) == 0, "get vec-003");
    ASSERT(idx == 2, "vec-003 maps to 2");

    ASSERT(point_id_get(map, "nonexistent", &idx) == -1, "get nonexistent returns -1");

    point_id_destroy(map);
    return 0;
}

static int test_point_id_update(void) {
    GV_PointIDMap *map = point_id_create(0);
    ASSERT(map != NULL, "map creation");

    ASSERT(point_id_set(map, "my-id", 10) == 0, "set my-id -> 10");

    size_t idx;
    ASSERT(point_id_get(map, "my-id", &idx) == 0, "get my-id");
    ASSERT(idx == 10, "my-id initially maps to 10");

    ASSERT(point_id_set(map, "my-id", 42) == 0, "update my-id -> 42");
    ASSERT(point_id_get(map, "my-id", &idx) == 0, "get my-id after update");
    ASSERT(idx == 42, "my-id now maps to 42");
    ASSERT(point_id_count(map) == 1, "count is 1 after update");

    point_id_destroy(map);
    return 0;
}

static int test_point_id_remove(void) {
    GV_PointIDMap *map = point_id_create(0);
    ASSERT(map != NULL, "map creation");

    point_id_set(map, "alpha", 0);
    point_id_set(map, "beta", 1);
    point_id_set(map, "gamma", 2);
    ASSERT(point_id_count(map) == 3, "count is 3");

    ASSERT(point_id_remove(map, "beta") == 0, "remove beta");
    ASSERT(point_id_count(map) == 2, "count is 2 after removal");
    ASSERT(point_id_has(map, "beta") == 0, "beta is absent after removal");

    ASSERT(point_id_remove(map, "nonexistent") == -1, "remove nonexistent returns -1");

    ASSERT(point_id_has(map, "alpha") == 1, "alpha still present");
    ASSERT(point_id_has(map, "gamma") == 1, "gamma still present");

    point_id_destroy(map);
    return 0;
}

static int test_point_id_has(void) {
    GV_PointIDMap *map = point_id_create(0);
    ASSERT(map != NULL, "map creation");

    ASSERT(point_id_has(map, "test") == 0, "has returns 0 for empty map");

    point_id_set(map, "test", 5);
    ASSERT(point_id_has(map, "test") == 1, "has returns 1 after set");

    point_id_remove(map, "test");
    ASSERT(point_id_has(map, "test") == 0, "has returns 0 after remove");

    point_id_destroy(map);
    return 0;
}

static int test_point_id_reverse_lookup(void) {
    GV_PointIDMap *map = point_id_create(0);
    ASSERT(map != NULL, "map creation");

    point_id_set(map, "uuid-abc-123", 7);
    point_id_set(map, "uuid-def-456", 12);

    const char *str = point_id_reverse_lookup(map, 7);
    ASSERT(str != NULL, "reverse lookup for index 7");
    ASSERT(strcmp(str, "uuid-abc-123") == 0, "reverse lookup returns correct string");

    str = point_id_reverse_lookup(map, 12);
    ASSERT(str != NULL, "reverse lookup for index 12");
    ASSERT(strcmp(str, "uuid-def-456") == 0, "reverse lookup returns correct string for 12");

    str = point_id_reverse_lookup(map, 999);
    ASSERT(str == NULL, "reverse lookup for missing index returns NULL");

    point_id_destroy(map);
    return 0;
}

static int test_point_id_generate_uuid(void) {
    char buf[37];
    ASSERT(point_id_generate_uuid(buf, sizeof(buf)) == 0, "generate UUID");

    ASSERT(strlen(buf) == 36, "UUID is 36 characters");
    ASSERT(buf[8] == '-', "dash at position 8");
    ASSERT(buf[13] == '-', "dash at position 13");
    ASSERT(buf[14] == '4', "version nibble is '4'");
    ASSERT(buf[18] == '-', "dash at position 18");
    ASSERT(buf[23] == '-', "dash at position 23");

    char variant = buf[19];
    ASSERT(variant == '8' || variant == '9' || variant == 'a' || variant == 'b',
           "variant nibble is valid");

    char buf2[37];
    point_id_generate_uuid(buf2, sizeof(buf2));
    ASSERT(strcmp(buf, buf2) != 0, "two UUIDs are different");

    char tiny[10];
    ASSERT(point_id_generate_uuid(tiny, sizeof(tiny)) == -1, "too-small buffer fails");

    ASSERT(point_id_generate_uuid(NULL, 37) == -1, "NULL buffer fails");

    return 0;
}

static int test_point_id_save_load(void) {
    const char *path = "/tmp/test_point_id_save_load.bin";
    GV_PointIDMap *map = point_id_create(0);
    ASSERT(map != NULL, "map creation");

    point_id_set(map, "first", 100);
    point_id_set(map, "second", 200);
    point_id_set(map, "third", 300);

    ASSERT(point_id_save(map, path) == 0, "save point ID map");
    point_id_destroy(map);

    GV_PointIDMap *loaded = point_id_load(path);
    ASSERT(loaded != NULL, "load point ID map");
    ASSERT(point_id_count(loaded) == 3, "loaded map has 3 entries");

    size_t idx;
    ASSERT(point_id_get(loaded, "first", &idx) == 0 && idx == 100, "loaded 'first' -> 100");
    ASSERT(point_id_get(loaded, "second", &idx) == 0 && idx == 200, "loaded 'second' -> 200");
    ASSERT(point_id_get(loaded, "third", &idx) == 0 && idx == 300, "loaded 'third' -> 300");

    point_id_destroy(loaded);
    remove(path);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing point_id create/destroy...", test_point_id_create_destroy},
        {"Testing point_id set and get...", test_point_id_set_and_get},
        {"Testing point_id update...", test_point_id_update},
        {"Testing point_id remove...", test_point_id_remove},
        {"Testing point_id has...", test_point_id_has},
        {"Testing point_id reverse lookup...", test_point_id_reverse_lookup},
        {"Testing point_id generate UUID...", test_point_id_generate_uuid},
        {"Testing point_id save/load...", test_point_id_save_load},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
