/**
 * test_serialization_portable.c — verifies the core/utils.h serialization
 * helpers emit a DETERMINISTIC little-endian, fixed-width, IEEE-754 on-disk
 * format. This guards endianness/width portability: the assertions on the raw
 * emitted bytes hold identically on little- and big-endian hosts (the helpers
 * byte-serialize explicitly), so a file written on one architecture is readable
 * on another. Also checks round-trip identity for every scalar/bulk helper.
 */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "core/utils.h"

#define ASSERT(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); return -1; } \
} while (0)

/* Read the whole tmpfile back into buf; returns byte count. */
static long slurp(FILE *f, uint8_t *buf, size_t cap) {
    fflush(f);
    if (fseek(f, 0, SEEK_END) != 0) return -1;
    long n = ftell(f);
    if (n < 0 || (size_t)n > cap) return -1;
    if (fseek(f, 0, SEEK_SET) != 0) return -1;
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) return -1;
    return n;
}

/* Assert the exact little-endian byte pattern of each scalar width. */
static int test_le_byte_layout(void) {
    uint8_t buf[64];

    { /* u16 */
        FILE *f = tmpfile(); ASSERT(f, "tmpfile u16");
        ASSERT(write_u16(f, 0x1122) == 0, "write_u16");
        long n = slurp(f, buf, sizeof(buf));
        ASSERT(n == 2, "u16 width==2");
        ASSERT(buf[0] == 0x22 && buf[1] == 0x11, "u16 little-endian bytes");
        fclose(f);
    }
    { /* u32 */
        FILE *f = tmpfile(); ASSERT(f, "tmpfile u32");
        ASSERT(write_u32(f, 0x11223344u) == 0, "write_u32");
        long n = slurp(f, buf, sizeof(buf));
        ASSERT(n == 4, "u32 width==4");
        ASSERT(buf[0]==0x44 && buf[1]==0x33 && buf[2]==0x22 && buf[3]==0x11, "u32 little-endian bytes");
        fclose(f);
    }
    { /* u64 */
        FILE *f = tmpfile(); ASSERT(f, "tmpfile u64");
        ASSERT(write_u64(f, 0x0102030405060708ull) == 0, "write_u64");
        long n = slurp(f, buf, sizeof(buf));
        ASSERT(n == 8, "u64 width==8");
        for (int i = 0; i < 8; i++)
            ASSERT(buf[i] == (uint8_t)(8 - i), "u64 little-endian bytes");
        fclose(f);
    }
    { /* size_t is normalized to a fixed 8 bytes on disk */
        FILE *f = tmpfile(); ASSERT(f, "tmpfile size");
        ASSERT(write_size(f, (size_t)0xABCD) == 0, "write_size");
        long n = slurp(f, buf, sizeof(buf));
        ASSERT(n == 8, "size width==8 (fixed)");
        ASSERT(buf[0]==0xCD && buf[1]==0xAB && buf[2]==0 && buf[7]==0, "size little-endian 8-byte");
        fclose(f);
    }
    { /* float via IEEE-754 bit pattern: 1.0f == 0x3F800000 */
        FILE *f = tmpfile(); ASSERT(f, "tmpfile f32");
        ASSERT(write_f32(f, 1.0f) == 0, "write_f32");
        long n = slurp(f, buf, sizeof(buf));
        ASSERT(n == 4, "f32 width==4");
        ASSERT(buf[0]==0x00 && buf[1]==0x00 && buf[2]==0x80 && buf[3]==0x3F, "f32 IEEE-754 LE bytes");
        fclose(f);
    }
    { /* double: 1.0 == 0x3FF0000000000000 */
        FILE *f = tmpfile(); ASSERT(f, "tmpfile f64");
        ASSERT(write_f64(f, 1.0) == 0, "write_f64");
        long n = slurp(f, buf, sizeof(buf));
        ASSERT(n == 8, "f64 width==8");
        ASSERT(buf[7]==0x3F && buf[6]==0xF0 && buf[0]==0x00, "f64 IEEE-754 LE bytes");
        fclose(f);
    }
    return 0;
}

/* Round-trip identity for every scalar/bulk helper. */
static int test_roundtrip(void) {
    FILE *f = tmpfile(); ASSERT(f, "tmpfile rt");

    const uint16_t u16 = 0xBEEF;
    const uint32_t u32 = 0xDEADBEEFu;
    const uint64_t u64 = 0x1122334455667788ull;
    const size_t   sz  = (size_t)123456789;
    const float    fv  = 3.14159265f;
    const double   dv  = 2.718281828459045;
    const float    arr[5] = { -1.5f, 0.0f, 42.0f, 1e-9f, 1e9f };

    ASSERT(write_u16(f, u16) == 0, "w u16");
    ASSERT(write_u32(f, u32) == 0, "w u32");
    ASSERT(write_u64(f, u64) == 0, "w u64");
    ASSERT(write_size(f, sz) == 0, "w size");
    ASSERT(write_f32(f, fv) == 0, "w f32");
    ASSERT(write_f64(f, dv) == 0, "w f64");
    ASSERT(write_floats(f, arr, 5) == 0, "w floats");

    ASSERT(fseek(f, 0, SEEK_SET) == 0, "rewind");

    uint16_t r16; uint32_t r32; uint64_t r64; size_t rsz; float rf; double rd; float rarr[5];
    ASSERT(read_u16(f, &r16) == 0 && r16 == u16, "rt u16");
    ASSERT(read_u32(f, &r32) == 0 && r32 == u32, "rt u32");
    ASSERT(read_u64(f, &r64) == 0 && r64 == u64, "rt u64");
    ASSERT(read_size(f, &rsz) == 0 && rsz == sz, "rt size");
    ASSERT(read_f32(f, &rf) == 0 && rf == fv, "rt f32");
    ASSERT(read_f64(f, &rd) == 0 && rd == dv, "rt f64");
    ASSERT(read_floats(f, rarr, 5) == 0, "rt floats read");
    for (int i = 0; i < 5; i++) ASSERT(rarr[i] == arr[i], "rt floats value");

    fclose(f);
    return 0;
}

/* read_size must reject an on-disk value that overflows this platform's size_t
 * only on 32-bit; on 64-bit any u64 fits. Verify a normal value round-trips and
 * that a truncated stream fails cleanly. */
static int test_edge(void) {
    FILE *f = tmpfile(); ASSERT(f, "tmpfile edge");
    /* Truncated read: write 3 bytes, then read_u32 must fail (needs 4). */
    uint8_t partial[3] = { 1, 2, 3 };
    ASSERT(fwrite(partial, 1, 3, f) == 3, "write partial");
    ASSERT(fseek(f, 0, SEEK_SET) == 0, "rewind edge");
    uint32_t v;
    ASSERT(read_u32(f, &v) != 0, "read_u32 on short stream must fail");
    fclose(f);
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= test_le_byte_layout();
    rc |= test_roundtrip();
    rc |= test_edge();
    if (rc == 0) printf("All serialization-portability tests PASSED.\n");
    return rc != 0;
}
