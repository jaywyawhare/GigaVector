#ifndef GIGAVECTOR_GV_GEO_H
#define GIGAVECTOR_GV_GEO_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct { double lat; double lng; } GV_GeoPoint;
typedef struct { GV_GeoPoint min; GV_GeoPoint max; } GV_GeoBBox;

typedef struct GV_GeoIndex GV_GeoIndex;

typedef struct {
    size_t point_index;    /* Index of the vector point */
    double lat;
    double lng;
    double distance_km;    /* Distance from query point */
} GV_GeoResult;

/* Lifecycle */
GV_GeoIndex *gv_geo_create(void);
void gv_geo_destroy(GV_GeoIndex *index);

/* Insert/remove geo coordinates for a point */
int gv_geo_insert(GV_GeoIndex *index, size_t point_index, double lat, double lng);
int gv_geo_update(GV_GeoIndex *index, size_t point_index, double lat, double lng);
int gv_geo_remove(GV_GeoIndex *index, size_t point_index);

/* Radius search: find all points within radius_km of center */
int gv_geo_radius_search(const GV_GeoIndex *index, double lat, double lng,
                          double radius_km, GV_GeoResult *results, size_t max_results);

/* Bounding box search */
int gv_geo_bbox_search(const GV_GeoIndex *index, const GV_GeoBBox *bbox,
                        GV_GeoResult *results, size_t max_results);

/* Get candidate indices for use as pre-filter in vector search */
int gv_geo_get_candidates(const GV_GeoIndex *index, double lat, double lng,
                           double radius_km, size_t *out_indices, size_t max_count);

/* Haversine distance between two points in km */
double gv_geo_distance_km(double lat1, double lng1, double lat2, double lng2);

/* Count */
size_t gv_geo_count(const GV_GeoIndex *index);

/* Save/load */
int gv_geo_save(const GV_GeoIndex *index, const char *filepath);
GV_GeoIndex *gv_geo_load(const char *filepath);

#ifdef __cplusplus
}
#endif
#endif
