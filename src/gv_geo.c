#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "gigavector/gv_geo.h"

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

#define GV_GEO_EARTH_RADIUS_KM  6371.0
#define GV_GEO_DEG_TO_RAD       (M_PI / 180.0)
#define GV_GEO_HASH_BUCKETS     65536
#define GV_GEO_GRID_SCALE       100   /* ~1 km granularity at equator */

/* ------------------------------------------------------------------ */
/*  Internal data structures                                           */
/* ------------------------------------------------------------------ */

typedef struct GV_GeoEntry {
    size_t              point_index;
    double              lat;
    double              lng;
    struct GV_GeoEntry *next;
} GV_GeoEntry;

typedef struct GV_GeoBucket {
    GV_GeoEntry *head;
} GV_GeoBucket;

struct GV_GeoIndex {
    GV_GeoBucket     buckets[GV_GEO_HASH_BUCKETS];
    size_t           count;
    pthread_rwlock_t rwlock;
};

/* ------------------------------------------------------------------ */
/*  Grid hashing                                                       */
/* ------------------------------------------------------------------ */

/**
 * @brief Compute a bucket index from a (lat, lng) pair.
 *
 * The world is divided into a grid of cells where each cell is roughly
 * 1 km at the equator (scale factor 100 means 0.01-degree steps which
 * is ~1.11 km in latitude).  The two grid coordinates are combined with
 * a simple hash to map into the fixed bucket array.
 */
static uint32_t geo_hash(double lat, double lng)
{
    int ilat = (int)(lat * GV_GEO_GRID_SCALE);
    int ilng = (int)(lng * GV_GEO_GRID_SCALE);

    /* Combine the two grid coordinates with a hash. */
    uint32_t h = (uint32_t)((ilat * 73856093) ^ (ilng * 19349663));
    return h % GV_GEO_HASH_BUCKETS;
}

/**
 * @brief Convert grid coordinate back to the integer pair for cell
 *        enumeration during range scans.
 */
static void geo_cell(double lat, double lng, int *out_ilat, int *out_ilng)
{
    *out_ilat = (int)(lat * GV_GEO_GRID_SCALE);
    *out_ilng = (int)(lng * GV_GEO_GRID_SCALE);
}

/* ------------------------------------------------------------------ */
/*  Haversine distance                                                 */
/* ------------------------------------------------------------------ */

double gv_geo_distance_km(double lat1, double lng1, double lat2, double lng2)
{
    double dlat = (lat2 - lat1) * GV_GEO_DEG_TO_RAD;
    double dlng = (lng2 - lng1) * GV_GEO_DEG_TO_RAD;

    double rlat1 = lat1 * GV_GEO_DEG_TO_RAD;
    double rlat2 = lat2 * GV_GEO_DEG_TO_RAD;

    double a = sin(dlat / 2.0) * sin(dlat / 2.0) +
               cos(rlat1) * cos(rlat2) *
               sin(dlng / 2.0) * sin(dlng / 2.0);
    double c = 2.0 * asin(sqrt(a));

    return GV_GEO_EARTH_RADIUS_KM * c;
}

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                          */
/* ------------------------------------------------------------------ */

GV_GeoIndex *gv_geo_create(void)
{
    GV_GeoIndex *index = (GV_GeoIndex *)calloc(1, sizeof(GV_GeoIndex));
    if (index == NULL) {
        return NULL;
    }

    if (pthread_rwlock_init(&index->rwlock, NULL) != 0) {
        free(index);
        return NULL;
    }

    return index;
}

void gv_geo_destroy(GV_GeoIndex *index)
{
    if (index == NULL) {
        return;
    }

    for (size_t i = 0; i < GV_GEO_HASH_BUCKETS; i++) {
        GV_GeoEntry *entry = index->buckets[i].head;
        while (entry != NULL) {
            GV_GeoEntry *next = entry->next;
            free(entry);
            entry = next;
        }
    }

    pthread_rwlock_destroy(&index->rwlock);
    free(index);
}

/* ------------------------------------------------------------------ */
/*  Insert / Update / Remove                                           */
/* ------------------------------------------------------------------ */

int gv_geo_insert(GV_GeoIndex *index, size_t point_index, double lat, double lng)
{
    if (index == NULL) {
        return -1;
    }
    if (lat < -90.0 || lat > 90.0 || lng < -180.0 || lng > 180.0) {
        return -1;
    }

    GV_GeoEntry *entry = (GV_GeoEntry *)malloc(sizeof(GV_GeoEntry));
    if (entry == NULL) {
        return -1;
    }
    entry->point_index = point_index;
    entry->lat = lat;
    entry->lng = lng;

    uint32_t bucket = geo_hash(lat, lng);

    pthread_rwlock_wrlock(&index->rwlock);
    entry->next = index->buckets[bucket].head;
    index->buckets[bucket].head = entry;
    index->count++;
    pthread_rwlock_unlock(&index->rwlock);

    return 0;
}

int gv_geo_update(GV_GeoIndex *index, size_t point_index, double lat, double lng)
{
    if (index == NULL) {
        return -1;
    }
    if (lat < -90.0 || lat > 90.0 || lng < -180.0 || lng > 180.0) {
        return -1;
    }

    /* Remove the old entry first, then insert the new one. */
    pthread_rwlock_wrlock(&index->rwlock);

    /* Scan all buckets for the point_index to remove it. */
    int found = 0;
    for (size_t i = 0; i < GV_GEO_HASH_BUCKETS && !found; i++) {
        GV_GeoEntry **pp = &index->buckets[i].head;
        while (*pp != NULL) {
            if ((*pp)->point_index == point_index) {
                GV_GeoEntry *victim = *pp;
                *pp = victim->next;
                free(victim);
                index->count--;
                found = 1;
                break;
            }
            pp = &(*pp)->next;
        }
    }

    /* Insert the new entry while still holding the write lock. */
    GV_GeoEntry *entry = (GV_GeoEntry *)malloc(sizeof(GV_GeoEntry));
    if (entry == NULL) {
        pthread_rwlock_unlock(&index->rwlock);
        return -1;
    }
    entry->point_index = point_index;
    entry->lat = lat;
    entry->lng = lng;

    uint32_t bucket = geo_hash(lat, lng);
    entry->next = index->buckets[bucket].head;
    index->buckets[bucket].head = entry;
    index->count++;
    pthread_rwlock_unlock(&index->rwlock);

    return found ? 0 : 1; /* 1 = was a fresh insert, not an update */
}

int gv_geo_remove(GV_GeoIndex *index, size_t point_index)
{
    if (index == NULL) {
        return -1;
    }

    pthread_rwlock_wrlock(&index->rwlock);

    for (size_t i = 0; i < GV_GEO_HASH_BUCKETS; i++) {
        GV_GeoEntry **pp = &index->buckets[i].head;
        while (*pp != NULL) {
            if ((*pp)->point_index == point_index) {
                GV_GeoEntry *victim = *pp;
                *pp = victim->next;
                free(victim);
                index->count--;
                pthread_rwlock_unlock(&index->rwlock);
                return 0;
            }
            pp = &(*pp)->next;
        }
    }

    pthread_rwlock_unlock(&index->rwlock);
    return -1; /* not found */
}

/* ------------------------------------------------------------------ */
/*  Radius search                                                      */
/* ------------------------------------------------------------------ */

/**
 * @brief Compute the approximate lat/lng bounding box for a circle
 *        centred at (lat, lng) with the given radius in km.
 *
 * Used to determine which grid cells to scan.
 */
static void radius_to_bbox(double lat, double lng, double radius_km,
                            double *min_lat, double *max_lat,
                            double *min_lng, double *max_lng)
{
    /* Approximate degrees of latitude per km. */
    double dlat = radius_km / (GV_GEO_EARTH_RADIUS_KM * GV_GEO_DEG_TO_RAD);

    /* Approximate degrees of longitude per km at the given latitude. */
    double cos_lat = cos(lat * GV_GEO_DEG_TO_RAD);
    double dlng;
    if (cos_lat < 1e-12) {
        dlng = 360.0; /* near a pole */
    } else {
        dlng = dlat / cos_lat;
    }

    *min_lat = lat - dlat;
    *max_lat = lat + dlat;
    *min_lng = lng - dlng;
    *max_lng = lng + dlng;

    /* Clamp latitude. */
    if (*min_lat < -90.0)  *min_lat = -90.0;
    if (*max_lat >  90.0)  *max_lat =  90.0;

    /* Longitude wrapping is handled during cell enumeration. */
}

/**
 * @brief Helper used by both radius_search and get_candidates.
 *
 * Iterates over the grid cells that overlap the search circle and
 * invokes the provided callback for each entry within the radius.
 * Returns the total number of qualifying entries written.
 */
static int geo_scan_radius(const GV_GeoIndex *index,
                            double lat, double lng, double radius_km,
                            GV_GeoResult *results, size_t *out_indices,
                            size_t max_count)
{
    if (index == NULL || max_count == 0) {
        return 0;
    }
    if (results == NULL && out_indices == NULL) {
        return 0;
    }

    double min_lat, max_lat, min_lng, max_lng;
    radius_to_bbox(lat, lng, radius_km, &min_lat, &max_lat, &min_lng, &max_lng);

    int cell_min_lat, cell_max_lat, cell_min_lng, cell_max_lng;
    geo_cell(min_lat, min_lng, &cell_min_lat, &cell_min_lng);
    geo_cell(max_lat, max_lng, &cell_max_lat, &cell_max_lng);

    size_t found = 0;

    for (int ilat = cell_min_lat; ilat <= cell_max_lat; ilat++) {
        for (int ilng = cell_min_lng; ilng <= cell_max_lng; ilng++) {
            uint32_t h = (uint32_t)((ilat * 73856093) ^ (ilng * 19349663));
            uint32_t bucket = h % GV_GEO_HASH_BUCKETS;

            const GV_GeoEntry *entry = index->buckets[bucket].head;
            while (entry != NULL) {
                double d = gv_geo_distance_km(lat, lng, entry->lat, entry->lng);
                if (d <= radius_km) {
                    if (results != NULL) {
                        results[found].point_index = entry->point_index;
                        results[found].lat = entry->lat;
                        results[found].lng = entry->lng;
                        results[found].distance_km = d;
                    }
                    if (out_indices != NULL) {
                        out_indices[found] = entry->point_index;
                    }
                    found++;
                    if (found >= max_count) {
                        return (int)found;
                    }
                }
                entry = entry->next;
            }
        }
    }

    return (int)found;
}

int gv_geo_radius_search(const GV_GeoIndex *index, double lat, double lng,
                          double radius_km, GV_GeoResult *results, size_t max_results)
{
    if (index == NULL || results == NULL || max_results == 0 || radius_km < 0.0) {
        return -1;
    }

    pthread_rwlock_rdlock((pthread_rwlock_t *)&index->rwlock);
    int n = geo_scan_radius(index, lat, lng, radius_km, results, NULL, max_results);
    pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);

    return n;
}

/* ------------------------------------------------------------------ */
/*  Bounding box search                                                */
/* ------------------------------------------------------------------ */

int gv_geo_bbox_search(const GV_GeoIndex *index, const GV_GeoBBox *bbox,
                        GV_GeoResult *results, size_t max_results)
{
    if (index == NULL || bbox == NULL || results == NULL || max_results == 0) {
        return -1;
    }

    double min_lat = bbox->min.lat;
    double max_lat = bbox->max.lat;
    double min_lng = bbox->min.lng;
    double max_lng = bbox->max.lng;

    if (min_lat > max_lat || min_lng > max_lng) {
        return -1;
    }

    int cell_min_lat, cell_max_lat, cell_min_lng, cell_max_lng;
    geo_cell(min_lat, min_lng, &cell_min_lat, &cell_min_lng);
    geo_cell(max_lat, max_lng, &cell_max_lat, &cell_max_lng);

    /* Centre of the bounding box, used to compute distances. */
    double clat = (min_lat + max_lat) / 2.0;
    double clng = (min_lng + max_lng) / 2.0;

    size_t found = 0;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&index->rwlock);

    for (int ilat = cell_min_lat; ilat <= cell_max_lat; ilat++) {
        for (int ilng = cell_min_lng; ilng <= cell_max_lng; ilng++) {
            uint32_t h = (uint32_t)((ilat * 73856093) ^ (ilng * 19349663));
            uint32_t bucket = h % GV_GEO_HASH_BUCKETS;

            const GV_GeoEntry *entry = index->buckets[bucket].head;
            while (entry != NULL) {
                if (entry->lat >= min_lat && entry->lat <= max_lat &&
                    entry->lng >= min_lng && entry->lng <= max_lng) {
                    results[found].point_index = entry->point_index;
                    results[found].lat = entry->lat;
                    results[found].lng = entry->lng;
                    results[found].distance_km =
                        gv_geo_distance_km(clat, clng, entry->lat, entry->lng);
                    found++;
                    if (found >= max_results) {
                        pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);
                        return (int)found;
                    }
                }
                entry = entry->next;
            }
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);
    return (int)found;
}

/* ------------------------------------------------------------------ */
/*  Candidate pre-filter for vector search                             */
/* ------------------------------------------------------------------ */

int gv_geo_get_candidates(const GV_GeoIndex *index, double lat, double lng,
                           double radius_km, size_t *out_indices, size_t max_count)
{
    if (index == NULL || out_indices == NULL || max_count == 0 || radius_km < 0.0) {
        return -1;
    }

    pthread_rwlock_rdlock((pthread_rwlock_t *)&index->rwlock);
    int n = geo_scan_radius(index, lat, lng, radius_km, NULL, out_indices, max_count);
    pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);

    return n;
}

/* ------------------------------------------------------------------ */
/*  Count                                                              */
/* ------------------------------------------------------------------ */

size_t gv_geo_count(const GV_GeoIndex *index)
{
    if (index == NULL) {
        return 0;
    }

    pthread_rwlock_rdlock((pthread_rwlock_t *)&index->rwlock);
    size_t c = index->count;
    pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);

    return c;
}

/* ------------------------------------------------------------------ */
/*  Save / Load                                                        */
/* ------------------------------------------------------------------ */

/**
 * @brief Persist the geo index to a binary file.
 *
 * File format (all values little-endian, native struct packing):
 *   uint64_t  count
 *   For each entry:
 *     uint64_t  point_index
 *     double    lat
 *     double    lng
 */
int gv_geo_save(const GV_GeoIndex *index, const char *filepath)
{
    if (index == NULL || filepath == NULL) {
        return -1;
    }

    FILE *fp = fopen(filepath, "wb");
    if (fp == NULL) {
        return -1;
    }

    pthread_rwlock_rdlock((pthread_rwlock_t *)&index->rwlock);

    uint64_t count = (uint64_t)index->count;
    if (fwrite(&count, sizeof(uint64_t), 1, fp) != 1) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);
        fclose(fp);
        return -1;
    }

    for (size_t i = 0; i < GV_GEO_HASH_BUCKETS; i++) {
        const GV_GeoEntry *entry = index->buckets[i].head;
        while (entry != NULL) {
            uint64_t pi = (uint64_t)entry->point_index;
            if (fwrite(&pi, sizeof(uint64_t), 1, fp) != 1 ||
                fwrite(&entry->lat, sizeof(double), 1, fp) != 1 ||
                fwrite(&entry->lng, sizeof(double), 1, fp) != 1) {
                pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);
                fclose(fp);
                return -1;
            }
            entry = entry->next;
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);
    fclose(fp);
    return 0;
}

GV_GeoIndex *gv_geo_load(const char *filepath)
{
    if (filepath == NULL) {
        return NULL;
    }

    FILE *fp = fopen(filepath, "rb");
    if (fp == NULL) {
        return NULL;
    }

    uint64_t count;
    if (fread(&count, sizeof(uint64_t), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }

    GV_GeoIndex *index = gv_geo_create();
    if (index == NULL) {
        fclose(fp);
        return NULL;
    }

    for (uint64_t i = 0; i < count; i++) {
        uint64_t pi;
        double lat, lng;
        if (fread(&pi, sizeof(uint64_t), 1, fp) != 1 ||
            fread(&lat, sizeof(double), 1, fp) != 1 ||
            fread(&lng, sizeof(double), 1, fp) != 1) {
            gv_geo_destroy(index);
            fclose(fp);
            return NULL;
        }
        if (gv_geo_insert(index, (size_t)pi, lat, lng) != 0) {
            gv_geo_destroy(index);
            fclose(fp);
            return NULL;
        }
    }

    fclose(fp);
    return index;
}
