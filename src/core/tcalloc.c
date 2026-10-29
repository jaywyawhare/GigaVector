#include "core/tcalloc.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>

/* AddressSanitizer / ThreadSanitizer instrument libc malloc to catch bugs; a
 * custom allocator hides those allocations from them. Compile to libc forwarders
 * so the sanitizer gates keep their coverage. */
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define GV_TC_FORCE_LIBC 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || \
    __has_feature(memory_sanitizer)
#define GV_TC_FORCE_LIBC 1
#endif
#endif

#ifdef GV_TC_FORCE_LIBC

/* Sanitizer builds: stay on libc so the tools keep intercepting allocations. */
#include <malloc.h>
void  *gv_tc_malloc(size_t size) { return size ? malloc(size) : NULL; }
void  *gv_tc_calloc(size_t n, size_t s) { return (n && s) ? calloc(n, s) : NULL; }
void  *gv_tc_realloc(void *p, size_t s) { return realloc(p, s); }
void   gv_tc_free(void *p) { free(p); }
size_t gv_tc_usable_size(void *p) { return p ? malloc_usable_size(p) : 0; }
int    gv_tc_active(void) { return 0; }
void   gv_tc_set_enabled(int on) { (void)on; }

#else /* real allocator */

#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>

/* Slab geometry. 1 MiB slabs keep large small-classes at >=15 objects per slab
 * and let free() find a slab header by masking the pointer to this boundary. */
#define GV_TC_SLAB_SIZE   (1u << 20) /* 1 MiB, also the slab alignment */
#define GV_TC_SLAB_MASK   ((uintptr_t)(GV_TC_SLAB_SIZE - 1))
#define GV_TC_MAX_SMALL   (64u * 1024u) /* above this -> direct mmap */
#define GV_TC_MAX_CLASSES 80            /* ladder is 65 classes; leaves headroom */
#define GV_TC_ALIGN       16u           /* matches malloc's min alignment */

/* Slab header at every slab base (GV_TC_SLAB_SIZE-aligned). The size class lives
 * in the ownership registry, not here, so the only field anyone reads back is the
 * mapped length of a large allocation, needed to munmap the exact region. */
typedef struct GV_Slab {
    size_t map_len; /* total mmap length (large allocations) */
} GV_Slab;

/* First object sits 16-byte aligned after the header. */
#define GV_TC_HDR_BYTES   ((sizeof(GV_Slab) + (GV_TC_ALIGN - 1)) & ~(size_t)(GV_TC_ALIGN - 1))

/* Per-size-class central store: a free list shared by all threads, refilled by
 * carving fresh slabs. Sharded one lock per class, so distinct classes never
 * contend and same-class traffic is batched (threads touch it once per batch). */
typedef struct GV_Central {
    pthread_mutex_t lock;
    void           *free_list;
    uint32_t        batch; /* objects moved per thread refill/flush */
} GV_Central;

/* Per-thread cache: a singly linked free list per class (next pointer lives in
 * the first 8 bytes of each free object). Alloc pops and free pushes with no
 * lock; the list is bounded by cap[] and spills a batch back to central. */
typedef struct GV_TCache {
    void    *list[GV_TC_MAX_CLASSES];
    uint32_t count[GV_TC_MAX_CLASSES];
} GV_TCache;

static uint32_t       g_class_size[GV_TC_MAX_CLASSES]; /* object size per class */
static uint32_t       g_class_count;                   /* number of active classes */
static uint8_t        g_size_to_class[GV_TC_MAX_SMALL / GV_TC_ALIGN + 1];
static GV_Central     g_central[GV_TC_MAX_CLASSES];
static int            g_inited; /* 0 until gv_tc_init runs; guarded by g_init_lock */
static pthread_mutex_t g_init_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_key_t  g_tcache_key;
static int            g_use_libc; /* GV_ALLOC=libc opt-out, decided once at init */
static int            g_check;    /* GV_TC_CHECK=1: poison + verify freed objects */
static size_t         g_page;

#include <malloc.h> /* libc fallback path (GV_ALLOC=libc) */

/*
 * Ownership + class registry. The codebase mixes raw libc malloc with gv_alloc,
 * and a libc-malloc'd pointer can reach gv_free, so free() must tell our pointers
 * from libc's before touching any slab metadata (masking a foreign pointer could
 * read an unmapped page). Every slab base is GV_TC_SLAB_SIZE-aligned, so the
 * base's slab index (addr >> 20) keys a two-level radix table: a 128 KiB L1
 * pointer table, 8 KiB L2 leaves allocated on demand. Each leaf byte is a tag:
 * 0 = foreign, GV_TC_TAG_LARGE = a large mmap, else (size class + 1). Storing the
 * class here means the small free/realloc path never reads the distant slab
 * header (a near-guaranteed cache miss) - the tag byte shared by every object in
 * a slab stays hot. Lookup is two relaxed atomic loads; tagging happens once per
 * slab under g_reg_lock. Covers the 47-bit user address space Linux hands out.
 */
#define GV_TC_L2_BITS   13
#define GV_TC_L1_BITS   14
#define GV_TC_IDX_BITS  (GV_TC_L1_BITS + GV_TC_L2_BITS)
#define GV_TC_TAG_LARGE 0xFFu
static uint8_t         *g_reg[1u << GV_TC_L1_BITS];
static pthread_mutex_t   g_reg_lock = PTHREAD_MUTEX_INITIALIZER;

static inline uintptr_t gv_tc_slab_idx(const void *p) {
    return (uintptr_t)p >> 20; /* 1 MiB slabs */
}

/* Registry tag for @p p: 0 foreign, GV_TC_TAG_LARGE large, else class+1. Safe on
 * any pointer. */
static inline uint8_t gv_tc_tag(const void *p) {
    uintptr_t idx = gv_tc_slab_idx(p);
    if (idx >> GV_TC_IDX_BITS) return 0;
    uint8_t *leaf = __atomic_load_n(&g_reg[idx >> GV_TC_L2_BITS], __ATOMIC_ACQUIRE);
    if (!leaf) return 0;
    return __atomic_load_n(&leaf[idx & ((1u << GV_TC_L2_BITS) - 1)], __ATOMIC_RELAXED);
}

/* Tag the slab base at @p p with @p tag (0 releases it). */
static void gv_tc_mark(const void *p, uint8_t tag) {
    uintptr_t idx = gv_tc_slab_idx(p);
    if (idx >> GV_TC_IDX_BITS) return;
    uint32_t l1 = (uint32_t)(idx >> GV_TC_L2_BITS);
    uint32_t l2 = (uint32_t)(idx & ((1u << GV_TC_L2_BITS) - 1));
    pthread_mutex_lock(&g_reg_lock);
    uint8_t *leaf = g_reg[l1];
    if (!leaf) {
        if (!tag) { pthread_mutex_unlock(&g_reg_lock); return; }
        leaf = (uint8_t *)calloc(1u << GV_TC_L2_BITS, 1);
        if (!leaf) { pthread_mutex_unlock(&g_reg_lock); return; }
        __atomic_store_n(&g_reg[l1], leaf, __ATOMIC_RELEASE);
    }
    __atomic_store_n(&leaf[l2], tag, __ATOMIC_RELEASE);
    pthread_mutex_unlock(&g_reg_lock);
}

/* Round @p v up to the next multiple of @p a (a is a power of two). */
static inline size_t gv_tc_align_up(size_t v, size_t a) {
    return (v + (a - 1)) & ~(a - 1);
}

/* Explicit size-class ladder (ported from C-ML's cml_allocator): 16-byte
 * granularity for tiny sizes, widening in stages up to GV_TC_MAX_SMALL. Every
 * value is a multiple of 16 so objects stay 16-byte aligned, and the fine low end
 * keeps internal fragmentation small for the many small allocations a vector DB
 * makes (graph adjacency, metadata, candidate heaps). */
static const uint32_t GV_TC_SIZE_CLASSES[] = {
    16,   32,   48,   64,   80,   96,   112,  128,  144,  160, 176,  192,
    208,  224,  240,  256,  /* 16B steps */
    288,  320,  352,  384,  416,  448,  480,  512,  /* 32B steps */
    576,  640,  704,  768,  832,  896,              /* 64B steps */
    1024, 1152, 1280, 1408,                         /* 128B steps */
    1536, 1792, 2048, 2304,                         /* 256B steps */
    2560, 3072, 3584, 4096,                         /* 512B steps */
    4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192,
    9216, 10240, 11264, 12288,                      /* 1KiB steps */
    14336, 16384, 18432, 20480, 22528, 24576,       /* 2KiB steps */
    28672, 32768, 40960,                            /* medium */
    49152, 65536};                                  /* up to GV_TC_MAX_SMALL */

/* Install the ladder into g_class_size and build the request->class lookup. */
static void gv_tc_build_classes(void) {
    uint32_t n = sizeof(GV_TC_SIZE_CLASSES) / sizeof(GV_TC_SIZE_CLASSES[0]);
    if (n > GV_TC_MAX_CLASSES) n = GV_TC_MAX_CLASSES;
    for (uint32_t i = 0; i < n; ++i) g_class_size[i] = GV_TC_SIZE_CLASSES[i];
    g_class_count = n;

    /* size_to_class[k] maps a request of k*16 bytes to its class index. */
    uint32_t cls = 0;
    for (uint32_t k = 0; k <= GV_TC_MAX_SMALL / GV_TC_ALIGN; ++k) {
        uint32_t bytes = k * GV_TC_ALIGN;
        while (cls < n && g_class_size[cls] < bytes) cls++;
        g_size_to_class[k] = (uint8_t)(cls < n ? cls : n - 1);
    }
}

static void gv_tc_tcache_destroy(void *arg);

static void gv_tc_init(void) {
    /* Opt-in: default to libc so the whole DB and its test suite are unaffected
     * until a user asks for the custom allocator. GV_ALLOC=tc (or gv/custom)
     * routes the gv_alloc choke point through the slab cache; GV_ALLOC=libc is
     * the explicit default. Compile with -DGV_TCALLOC_DEFAULT_ON to flip it. */
#ifdef GV_TCALLOC_DEFAULT_ON
    g_use_libc = 0;
#else
    g_use_libc = 1;
#endif
    const char *env = getenv("GV_ALLOC");
    if (env) {
        if (strcmp(env, "libc") == 0) g_use_libc = 1;
        else if (strcmp(env, "tc") == 0 || strcmp(env, "gv") == 0 ||
                 strcmp(env, "custom") == 0) g_use_libc = 0;
    }
    const char *chk = getenv("GV_TC_CHECK");
    if (chk && chk[0] == '1') g_check = 1;
    g_page = (size_t)sysconf(_SC_PAGESIZE);
    if (g_page == 0) g_page = 4096;
    gv_tc_build_classes();
    for (uint32_t i = 0; i < g_class_count; ++i) {
        pthread_mutex_init(&g_central[i].lock, NULL);
        /* Move ~16 KiB worth of objects per batch, clamped to [8, 1024]. */
        uint32_t b = (16u * 1024u) / g_class_size[i];
        if (b < 8) b = 8;
        if (b > 1024) b = 1024;
        g_central[i].batch = b;
    }
    pthread_key_create(&g_tcache_key, gv_tc_tcache_destroy);
}

/* One-time init behind an inlined acquire-load guard, so the hot path pays a
 * single predicted-taken branch instead of a cross-library pthread_once call. */
static inline void gv_tc_ensure(void) {
    if (__atomic_load_n(&g_inited, __ATOMIC_ACQUIRE)) return;
    pthread_mutex_lock(&g_init_lock);
    if (!g_inited) {
        gv_tc_init();
        __atomic_store_n(&g_inited, 1, __ATOMIC_RELEASE);
    }
    pthread_mutex_unlock(&g_init_lock);
}

/* Initialise at load so the hot-path guard above is always already satisfied. */
__attribute__((constructor)) static void gv_tc_ctor(void) { gv_tc_ensure(); }

/* mmap @p len bytes aligned to GV_TC_SLAB_SIZE by over-mapping and trimming the
 * unaligned head/tail. Returns NULL on failure. */
static void *gv_tc_map_aligned(size_t len) {
    size_t over = len + GV_TC_SLAB_SIZE;
    char *raw = mmap(NULL, over, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) return NULL;
    uintptr_t base = ((uintptr_t)raw + GV_TC_SLAB_MASK) & ~GV_TC_SLAB_MASK;
    size_t head = (size_t)(base - (uintptr_t)raw);
    if (head) munmap(raw, head);
    size_t tail = over - head - len;
    if (tail) munmap((void *)(base + len), tail);
    return (void *)base;
}

/* GV_TC_CHECK integrity poison: every free object past its next-pointer is held
 * at 0xEF, so a write-after-free or an overflow from the neighbouring object is
 * caught the moment the object is handed back out, with its address. */
#define GV_TC_POISON 0xEF
static void gv_tc_poison(void *p, uint32_t obj) {
    if (obj > 8) memset((char *)p + 8, GV_TC_POISON, obj - 8);
}
static void gv_tc_verify(const void *p, uint32_t obj) {
    const unsigned char *b = (const unsigned char *)p;
    for (uint32_t i = 8; i < obj; i++) {
        if (b[i] != GV_TC_POISON) {
            fprintf(stderr,
                    "gv_tcalloc: object %p written while free (obj=%u, byte %u=0x%02x)\n",
                    p, obj, i, b[i]);
            abort();
        }
    }
}

/* Carve a fresh slab for @p class_idx and push all its objects onto the central
 * free list. Caller holds g_central[class_idx].lock. Returns objects added. */
static uint32_t gv_tc_grow_central(uint32_t class_idx) {
    GV_Slab *slab = (GV_Slab *)gv_tc_map_aligned(GV_TC_SLAB_SIZE);
    if (!slab) return 0;
    uint32_t obj = g_class_size[class_idx];
    gv_tc_mark(slab, (uint8_t)(class_idx + 1));

    char *p = (char *)slab + GV_TC_HDR_BYTES;
    char *end = (char *)slab + GV_TC_SLAB_SIZE;
    uint32_t added = 0;
    void *head = g_central[class_idx].free_list;
    for (; p + obj <= end; p += obj) {
        *(void **)p = head;
        if (g_check) gv_tc_poison(p, obj);
        head = p;
        added++;
    }
    g_central[class_idx].free_list = head;
    return added;
}

/* The thread cache lives in __thread storage so the alloc hot path reaches it
 * with a single TLS access (no pthread_getspecific call, which dominated an
 * earlier design). g_tcache_key holds only a sentinel per thread to arm the
 * exit destructor; the destructor flushes this same storage. */
static __thread GV_TCache t_cache;
static __thread int        t_armed;
/* One-entry memo of the last small slab freed into. Small slabs never change
 * class or get unmapped, so a base->tag hit on free skips the registry lookup
 * entirely - and consecutive frees cluster in a handful of slabs. */
static __thread uintptr_t  t_last_base;
static __thread uint8_t    t_last_tag;

static inline GV_TCache *gv_tc_get_cache(void) {
    if (!t_armed) {
        pthread_setspecific(g_tcache_key, (void *)1);
        t_armed = 1;
    }
    return &t_cache;
}

/* Return a whole free list (len objects) to the class's central store. */
static void gv_tc_flush_list(uint32_t class_idx, void *head, void *tail) {
    pthread_mutex_lock(&g_central[class_idx].lock);
    *(void **)tail = g_central[class_idx].free_list;
    g_central[class_idx].free_list = head;
    pthread_mutex_unlock(&g_central[class_idx].lock);
}

/* On thread exit, hand every cached object back to central so it stays usable.
 * Runs in the exiting thread's context, so t_cache (its TLS) is still live. */
static void gv_tc_tcache_destroy(void *arg) {
    (void)arg;
    for (uint32_t i = 0; i < g_class_count; ++i) {
        void *head = t_cache.list[i];
        if (!head) continue;
        void *tail = head;
        while (*(void **)tail) tail = *(void **)tail;
        gv_tc_flush_list(i, head, tail);
        t_cache.list[i] = NULL;
        t_cache.count[i] = 0;
    }
}

/* Pull up to batch objects for @p class_idx from central into @p c. */
static void *gv_tc_refill(GV_TCache *c, uint32_t class_idx) {
    GV_Central *ce = &g_central[class_idx];
    pthread_mutex_lock(&ce->lock);
    if (!ce->free_list && gv_tc_grow_central(class_idx) == 0) {
        pthread_mutex_unlock(&ce->lock);
        return NULL;
    }
    void *head = ce->free_list;
    void *p = head;
    uint32_t got = 1;
    while (got < ce->batch && *(void **)p) {
        p = *(void **)p;
        got++;
    }
    ce->free_list = *(void **)p; /* detach [head..p] */
    pthread_mutex_unlock(&ce->lock);

    *(void **)p = NULL;
    /* Keep head to return; cache the rest. */
    c->list[class_idx] = *(void **)head;
    c->count[class_idx] = got - 1;
    if (g_check) gv_tc_verify(head, g_class_size[class_idx]);
    return head;
}

void *gv_tc_malloc(size_t size) {
    if (size == 0) return NULL;
    gv_tc_ensure();
    if (g_use_libc) return malloc(size);

    if (size <= GV_TC_MAX_SMALL) {
        uint32_t k = (uint32_t)((size + GV_TC_ALIGN - 1) / GV_TC_ALIGN);
        uint32_t ci = g_size_to_class[k];
        GV_TCache *c = gv_tc_get_cache();
        void *p = c->list[ci];
        if (p) {
            c->list[ci] = *(void **)p;
            c->count[ci]--;
            if (g_check) gv_tc_verify(p, g_class_size[ci]);
            return p;
        }
        return gv_tc_refill(c, ci);
    }

    /* Large: one slab-aligned mmap carrying a header before the payload. */
    size_t len = gv_tc_align_up(GV_TC_HDR_BYTES + size, g_page);
    GV_Slab *slab = (GV_Slab *)gv_tc_map_aligned(len);
    if (!slab) return NULL;
    slab->map_len = len;
    gv_tc_mark(slab, GV_TC_TAG_LARGE);
    return (char *)slab + GV_TC_HDR_BYTES;
}

void *gv_tc_calloc(size_t nmemb, size_t size) {
    if (nmemb == 0 || size == 0) return NULL;
    if (nmemb > SIZE_MAX / size) return NULL;
    size_t total = nmemb * size;
    gv_tc_ensure();
    if (g_use_libc) return calloc(nmemb, size);
    void *p = gv_tc_malloc(total);
    /* Small objects come from recycled slabs and must be cleared; fresh mmap
     * (large path) is already zero but memset keeps the contract uniform. */
    if (p) memset(p, 0, total);
    return p;
}

size_t gv_tc_usable_size(void *ptr) {
    if (!ptr) return 0;
    gv_tc_ensure();
    uint8_t tag = g_use_libc ? 0 : gv_tc_tag(ptr);
    if (tag == 0) return malloc_usable_size(ptr);
    if (tag == GV_TC_TAG_LARGE) {
        GV_Slab *slab = (GV_Slab *)((uintptr_t)ptr & ~GV_TC_SLAB_MASK);
        return slab->map_len - GV_TC_HDR_BYTES;
    }
    return g_class_size[tag - 1];
}

void gv_tc_free(void *ptr) {
    if (!ptr) return;
    gv_tc_ensure();
    /* Foreign (raw libc) pointers reach gv_free too: the registry tag forwards
     * them to libc rather than masking to a header on a possibly unmapped page.
     * The tag also carries the size class, so the small path reads no slab header. */
    uint8_t tag;
    if (g_use_libc) {
        free(ptr);
        return;
    }
    uintptr_t base = (uintptr_t)ptr & ~GV_TC_SLAB_MASK;
    if (base == t_last_base && t_last_tag) {
        tag = t_last_tag; /* memo hit: immutable small-slab class, no registry load */
    } else {
        tag = gv_tc_tag(ptr);
        if (tag != 0 && tag != GV_TC_TAG_LARGE) { t_last_base = base; t_last_tag = tag; }
    }
    if (tag == 0) { free(ptr); return; }
    if (tag == GV_TC_TAG_LARGE) {
        GV_Slab *slab = (GV_Slab *)base;
        gv_tc_mark(slab, 0);
        munmap(slab, slab->map_len);
        return;
    }
    /* Small: push back onto the thread cache; spill a batch to central if full. */
    uint32_t ci = tag - 1;
    GV_TCache *c = gv_tc_get_cache();
    *(void **)ptr = c->list[ci];
    if (g_check) gv_tc_poison(ptr, g_class_size[ci]);
    c->list[ci] = ptr;
    c->count[ci]++;

    uint32_t cap = g_central[ci].batch * 2;
    if (c->count[ci] > cap) {
        /* Detach the oldest batch (tail of the list) and return it to central. */
        uint32_t keep = g_central[ci].batch;
        void *p = c->list[ci];
        for (uint32_t i = 1; i < keep; ++i) p = *(void **)p;
        void *spill_head = *(void **)p;
        *(void **)p = NULL; /* new tail of the kept portion */
        void *tail = spill_head;
        while (*(void **)tail) tail = *(void **)tail;
        gv_tc_flush_list(ci, spill_head, tail);
        c->count[ci] = keep;
    }
}

void *gv_tc_realloc(void *ptr, size_t size) {
    if (!ptr) return gv_tc_malloc(size);
    if (size == 0) { gv_tc_free(ptr); return NULL; }
    gv_tc_ensure();
    /* A foreign pointer stays with libc so its allocator pairing is preserved. */
    uint8_t tag = g_use_libc ? 0 : gv_tc_tag(ptr);
    if (tag == 0) return realloc(ptr, size);

    size_t old;
    if (tag == GV_TC_TAG_LARGE) {
        GV_Slab *slab = (GV_Slab *)((uintptr_t)ptr & ~GV_TC_SLAB_MASK);
        old = slab->map_len - GV_TC_HDR_BYTES;
    } else {
        old = g_class_size[tag - 1];
    }
    /* Small -> same class, or large -> still fits the mapped payload: reuse. */
    if (size <= old) {
        if (tag != GV_TC_TAG_LARGE) {
            uint32_t k = (uint32_t)((size + GV_TC_ALIGN - 1) / GV_TC_ALIGN);
            if (g_class_size[g_size_to_class[k]] == old) return ptr;
        } else {
            return ptr;
        }
    }
    void *np = gv_tc_malloc(size);
    if (!np) return NULL;
    memcpy(np, ptr, old < size ? old : size);
    gv_tc_free(ptr);
    return np;
}

int gv_tc_active(void) {
    gv_tc_ensure();
    return !g_use_libc;
}

void gv_tc_set_enabled(int on) {
    gv_tc_ensure();
    g_use_libc = on ? 0 : 1;
}

#endif /* GV_TC_FORCE_LIBC */
