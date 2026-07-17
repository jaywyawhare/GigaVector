#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "core/id_bitmap.h"
#include "core/memory.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* ---- iteration collector --------------------------------------------- */

typedef struct {
    uint64_t *ids;
    size_t    n;
    size_t    cap;
    int       ordered; /* set to 0 if a non-ascending step is seen */
    uint64_t  last;
    int       have_last;
} Collect;

static int collect_cb(uint64_t id, void *ctx)
{
    Collect *c = (Collect *)ctx;
    if (c->have_last && id <= c->last) c->ordered = 0;
    c->last = id;
    c->have_last = 1;
    if (c->n == c->cap) {
        size_t ncap = c->cap ? c->cap * 2 : 16;
        uint64_t *ni = (uint64_t *)realloc(c->ids, ncap * sizeof(uint64_t));
        if (!ni) return 1; /* stop */
        c->ids = ni; c->cap = ncap;
    }
    c->ids[c->n++] = id;
    return 0;
}

/* ---- tests ------------------------------------------------------------ */

static int test_add_contains_card(void)
{
    GV_IdBitmap *b = gv_id_bitmap_create();
    ASSERT(b != NULL, "create");

    static const uint64_t ids[] = {
        0, 1, 65535, 65536, (1ull << 20), (1ull << 40) + 7, UINT64_MAX
    };
    const size_t n = sizeof(ids) / sizeof(ids[0]);

    for (size_t i = 0; i < n; i++)
        ASSERT(gv_id_bitmap_add(b, ids[i]) == 0, "add");

    /* Idempotent re-add. */
    for (size_t i = 0; i < n; i++)
        ASSERT(gv_id_bitmap_add(b, ids[i]) == 0, "re-add");

    ASSERT(gv_id_bitmap_cardinality(b) == n, "cardinality after dedup");

    for (size_t i = 0; i < n; i++)
        ASSERT(gv_id_bitmap_contains(b, ids[i]) == 1, "contains member");

    ASSERT(gv_id_bitmap_contains(b, 2) == 0, "absent 2");
    ASSERT(gv_id_bitmap_contains(b, 65537) == 0, "absent 65537");
    ASSERT(gv_id_bitmap_contains(b, UINT64_MAX - 1) == 0, "absent max-1");

    gv_id_bitmap_free(b);
    gv_id_bitmap_free(NULL); /* safe */
    return 0;
}

static int test_array_to_bitmap_promotion(void)
{
    GV_IdBitmap *b = gv_id_bitmap_create();
    ASSERT(b != NULL, "create");

    /* All within one high-bit chunk (key = 3): base = 3 << 16. */
    const uint64_t base = (uint64_t)3 << 16;
    const uint32_t count = 5000; /* > 4096 forces array->bitmap */
    for (uint32_t i = 0; i < count; i++)
        ASSERT(gv_id_bitmap_add(b, base + i) == 0, "add promotion");

    ASSERT(gv_id_bitmap_cardinality(b) == count, "cardinality after promotion");
    for (uint32_t i = 0; i < count; i++)
        ASSERT(gv_id_bitmap_contains(b, base + i) == 1, "contains after promotion");
    ASSERT(gv_id_bitmap_contains(b, base + count) == 0, "absent past range");

    /* Re-add is still idempotent post-promotion. */
    ASSERT(gv_id_bitmap_add(b, base + 100) == 0, "re-add promoted");
    ASSERT(gv_id_bitmap_cardinality(b) == count, "cardinality stable");

    gv_id_bitmap_free(b);
    return 0;
}

static int test_remove(void)
{
    GV_IdBitmap *b = gv_id_bitmap_create();
    ASSERT(b != NULL, "create");

    for (uint64_t i = 0; i < 100; i++)
        ASSERT(gv_id_bitmap_add(b, i * 700) == 0, "add");
    ASSERT(gv_id_bitmap_cardinality(b) == 100, "card 100");

    /* Remove present. */
    ASSERT(gv_id_bitmap_remove(b, 0) == 0, "remove present 0");
    ASSERT(gv_id_bitmap_contains(b, 0) == 0, "0 gone");
    ASSERT(gv_id_bitmap_remove(b, 50 * 700) == 0, "remove present mid");
    ASSERT(gv_id_bitmap_contains(b, 50 * 700) == 0, "mid gone");
    ASSERT(gv_id_bitmap_cardinality(b) == 98, "card 98");

    /* Remove absent is a no-op. */
    ASSERT(gv_id_bitmap_remove(b, 123456789) == 0, "remove absent");
    ASSERT(gv_id_bitmap_cardinality(b) == 98, "card unchanged");
    gv_id_bitmap_remove(NULL, 5); /* safe */

    gv_id_bitmap_free(b);
    return 0;
}

/* ---- brute-force reference sets --------------------------------------- */

static int u64_cmp(const void *x, const void *y)
{
    uint64_t a = *(const uint64_t *)x, b = *(const uint64_t *)y;
    return (a < b) ? -1 : (a > b) ? 1 : 0;
}

static int test_union_intersect(void)
{
    GV_IdBitmap *a = gv_id_bitmap_create();
    GV_IdBitmap *b = gv_id_bitmap_create();
    ASSERT(a && b, "create a,b");

    /* Deterministic pseudo-random ids spread across chunks. */
    uint64_t seed = 0x1234567u;
    uint64_t setA[300], setB[300];
    for (int i = 0; i < 300; i++) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        uint64_t v = seed % 200000ull; /* dense enough for overlap */
        setA[i] = v;
        ASSERT(gv_id_bitmap_add(a, v) == 0, "add a");
    }
    for (int i = 0; i < 300; i++) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        uint64_t v = seed % 200000ull;
        setB[i] = v;
        ASSERT(gv_id_bitmap_add(b, v) == 0, "add b");
    }

    /* Reference union / intersect via sorted arrays + membership scan. */
    qsort(setA, 300, sizeof(uint64_t), u64_cmp);
    qsort(setB, 300, sizeof(uint64_t), u64_cmp);

    GV_IdBitmap *u = gv_id_bitmap_or(a, b);
    GV_IdBitmap *in = gv_id_bitmap_and(a, b);
    ASSERT(u && in, "or/and results");

    /* Every element of A and B is in union. */
    for (int i = 0; i < 300; i++) {
        ASSERT(gv_id_bitmap_contains(u, setA[i]) == 1, "union has A");
        ASSERT(gv_id_bitmap_contains(u, setB[i]) == 1, "union has B");
    }
    /* Intersection: id present iff in both. */
    for (int i = 0; i < 300; i++) {
        int inA = gv_id_bitmap_contains(a, setB[i]);
        int inB = 1; /* setB[i] is in b by construction */
        int expect = inA && inB;
        ASSERT(gv_id_bitmap_contains(in, setB[i]) == expect, "intersect member match");
    }
    for (int i = 0; i < 300; i++) {
        int expect = gv_id_bitmap_contains(b, setA[i]) ? 1 : 0;
        ASSERT(gv_id_bitmap_contains(in, setA[i]) == expect, "intersect A match");
    }

    /* Cardinality of intersection must not exceed either input. */
    ASSERT(gv_id_bitmap_cardinality(in) <= gv_id_bitmap_cardinality(a), "and <= a");
    ASSERT(gv_id_bitmap_cardinality(in) <= gv_id_bitmap_cardinality(b), "and <= b");
    /* union >= both inputs. */
    ASSERT(gv_id_bitmap_cardinality(u) >= gv_id_bitmap_cardinality(a), "or >= a");
    ASSERT(gv_id_bitmap_cardinality(u) >= gv_id_bitmap_cardinality(b), "or >= b");

    /* or_into equivalence. */
    GV_IdBitmap *u2 = gv_id_bitmap_clone(a);
    ASSERT(u2 != NULL, "clone a");
    ASSERT(gv_id_bitmap_or_into(u2, b) == 0, "or_into");
    ASSERT(gv_id_bitmap_cardinality(u2) == gv_id_bitmap_cardinality(u), "or_into card matches or");
    for (int i = 0; i < 300; i++)
        ASSERT(gv_id_bitmap_contains(u2, setB[i]) == 1, "or_into has B");

    gv_id_bitmap_free(a);
    gv_id_bitmap_free(b);
    gv_id_bitmap_free(u);
    gv_id_bitmap_free(u2);
    gv_id_bitmap_free(in);
    return 0;
}

static int test_iterate_ascending(void)
{
    GV_IdBitmap *b = gv_id_bitmap_create();
    ASSERT(b != NULL, "create");

    static const uint64_t ids[] = {
        UINT64_MAX, 0, 65536, 65535, (1ull << 40) + 7, 1, (1ull << 20)
    };
    const size_t n = sizeof(ids) / sizeof(ids[0]);
    for (size_t i = 0; i < n; i++)
        ASSERT(gv_id_bitmap_add(b, ids[i]) == 0, "add");

    Collect c;
    memset(&c, 0, sizeof(c));
    c.ordered = 1;
    ASSERT(gv_id_bitmap_iterate(b, collect_cb, &c) == 0, "iterate ok");
    ASSERT(c.n == n, "iterate visits all members exactly once");
    ASSERT(c.ordered == 1, "iterate ascending");

    /* Sorted copy of ids matches visited order. */
    uint64_t sorted[7];
    memcpy(sorted, ids, sizeof(ids));
    qsort(sorted, n, sizeof(uint64_t), u64_cmp);
    for (size_t i = 0; i < n; i++)
        ASSERT(c.ids[i] == sorted[i], "iterate order matches sorted");

    free(c.ids);
    gv_id_bitmap_free(b);
    return 0;
}

static int test_serialize_roundtrip(void)
{
    GV_IdBitmap *b = gv_id_bitmap_create();
    ASSERT(b != NULL, "create");

    /* Mixed: a dense chunk (bitmap) + sparse ids across chunks. */
    const uint64_t base = (uint64_t)7 << 16;
    for (uint32_t i = 0; i < 5000; i++)
        ASSERT(gv_id_bitmap_add(b, base + i) == 0, "add dense");
    static const uint64_t sparse[] = { 0, 1, 65535, (1ull << 33) + 9, UINT64_MAX };
    for (size_t i = 0; i < sizeof(sparse) / sizeof(sparse[0]); i++)
        ASSERT(gv_id_bitmap_add(b, sparse[i]) == 0, "add sparse");

    uint8_t *buf = NULL; size_t len = 0;
    ASSERT(gv_id_bitmap_serialize(b, &buf, &len) == 0, "serialize");
    ASSERT(buf != NULL && len > 0, "buffer produced");

    GV_IdBitmap *r = gv_id_bitmap_deserialize(buf, len);
    ASSERT(r != NULL, "deserialize");
    ASSERT(gv_id_bitmap_cardinality(r) == gv_id_bitmap_cardinality(b), "card round-trip");

    for (uint32_t i = 0; i < 5000; i++)
        ASSERT(gv_id_bitmap_contains(r, base + i) == 1, "dense round-trip");
    for (size_t i = 0; i < sizeof(sparse) / sizeof(sparse[0]); i++)
        ASSERT(gv_id_bitmap_contains(r, sparse[i]) == 1, "sparse round-trip");
    ASSERT(gv_id_bitmap_contains(r, base + 5000) == 0, "absent round-trip");

    /* Truncated buffer => NULL. */
    ASSERT(gv_id_bitmap_deserialize(buf, len - 1) == NULL, "truncated tail");
    ASSERT(gv_id_bitmap_deserialize(buf, 10) == NULL, "truncated header");
    ASSERT(gv_id_bitmap_deserialize(buf, 0) == NULL, "zero length");
    ASSERT(gv_id_bitmap_deserialize(NULL, 100) == NULL, "null data");

    /* Corrupt magic. */
    uint8_t *bad = (uint8_t *)malloc(len);
    ASSERT(bad != NULL, "alloc bad");
    memcpy(bad, buf, len);
    bad[0] ^= 0xFFu;
    ASSERT(gv_id_bitmap_deserialize(bad, len) == NULL, "bad magic");

    /* Corrupt a payload byte => CRC mismatch => NULL. */
    memcpy(bad, buf, len);
    bad[len / 2] ^= 0xFFu;
    ASSERT(gv_id_bitmap_deserialize(bad, len) == NULL, "crc mismatch");

    /* Pure garbage of a plausible length. */
    uint8_t garbage[64];
    for (size_t i = 0; i < sizeof(garbage); i++) garbage[i] = (uint8_t)(i * 37 + 3);
    ASSERT(gv_id_bitmap_deserialize(garbage, sizeof(garbage)) == NULL, "garbage");

    /* Empty bitmap round-trips too. */
    GV_IdBitmap *e = gv_id_bitmap_create();
    ASSERT(e != NULL, "create empty");
    uint8_t *ebuf = NULL; size_t elen = 0;
    ASSERT(gv_id_bitmap_serialize(e, &ebuf, &elen) == 0, "serialize empty");
    GV_IdBitmap *er = gv_id_bitmap_deserialize(ebuf, elen);
    ASSERT(er != NULL, "deserialize empty");
    ASSERT(gv_id_bitmap_cardinality(er) == 0, "empty card 0");

    free(bad);
    gv_free(buf);
    gv_free(ebuf);
    gv_id_bitmap_free(e);
    gv_id_bitmap_free(er);
    gv_id_bitmap_free(r);
    gv_id_bitmap_free(b);
    return 0;
}

static int test_clone_independence(void)
{
    GV_IdBitmap *b = gv_id_bitmap_create();
    ASSERT(b != NULL, "create");
    for (uint64_t i = 0; i < 200; i++)
        ASSERT(gv_id_bitmap_add(b, i * 1000) == 0, "add");

    GV_IdBitmap *c = gv_id_bitmap_clone(b);
    ASSERT(c != NULL, "clone");
    ASSERT(gv_id_bitmap_cardinality(c) == gv_id_bitmap_cardinality(b), "clone card");

    /* Mutate clone; original unchanged. */
    ASSERT(gv_id_bitmap_add(c, 999999999ull) == 0, "add to clone");
    ASSERT(gv_id_bitmap_remove(c, 0) == 0, "remove from clone");
    ASSERT(gv_id_bitmap_contains(c, 999999999ull) == 1, "clone has new");
    ASSERT(gv_id_bitmap_contains(b, 999999999ull) == 0, "orig lacks new");
    ASSERT(gv_id_bitmap_contains(c, 0) == 0, "clone lacks removed");
    ASSERT(gv_id_bitmap_contains(b, 0) == 1, "orig keeps removed");

    /* Mutate original; clone unchanged. */
    ASSERT(gv_id_bitmap_add(b, 77) == 0, "add to orig");
    ASSERT(gv_id_bitmap_contains(c, 77) == 0, "clone lacks orig's new");

    gv_id_bitmap_free(c);
    gv_id_bitmap_free(b);
    ASSERT(gv_id_bitmap_clone(NULL) == NULL, "clone NULL is NULL");
    return 0;
}

static int test_clear(void)
{
    GV_IdBitmap *b = gv_id_bitmap_create();
    ASSERT(b != NULL, "create");
    for (uint64_t i = 0; i < 6000; i++)
        ASSERT(gv_id_bitmap_add(b, i) == 0, "add");
    ASSERT(gv_id_bitmap_cardinality(b) == 6000, "card before clear");

    gv_id_bitmap_clear(b);
    ASSERT(gv_id_bitmap_cardinality(b) == 0, "card 0 after clear");
    ASSERT(gv_id_bitmap_contains(b, 5) == 0, "empty after clear");

    /* Reusable after clear. */
    ASSERT(gv_id_bitmap_add(b, 42) == 0, "add after clear");
    ASSERT(gv_id_bitmap_contains(b, 42) == 1, "contains after reuse");
    ASSERT(gv_id_bitmap_cardinality(b) == 1, "card 1 after reuse");

    gv_id_bitmap_clear(NULL); /* safe */
    gv_id_bitmap_free(b);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void)
{
    TestCase tests[] = {
        {"Testing id_bitmap add/contains/cardinality...", test_add_contains_card},
        {"Testing id_bitmap array->bitmap promotion...",  test_array_to_bitmap_promotion},
        {"Testing id_bitmap remove...",                   test_remove},
        {"Testing id_bitmap union/intersect...",          test_union_intersect},
        {"Testing id_bitmap iterate ascending...",        test_iterate_ascending},
        {"Testing id_bitmap serialize round-trip...",     test_serialize_roundtrip},
        {"Testing id_bitmap clone independence...",       test_clone_independence},
        {"Testing id_bitmap clear/reuse...",              test_clear},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
