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

GV_GeoIndex *geo_create(void);
/**
 * @brief Destroy an instance and free associated resources.
 *
 * @param index Index instance.
 */
void geo_destroy(GV_GeoIndex *index);

/**
 * @brief Perform the operation.
 *
 * @param index Index instance.
 * @param point_index Index value.
 * @param lat lat.
 * @param lng lng.
 * @return 0 on success, -1 on error.
 */
int geo_insert(GV_GeoIndex *index, size_t point_index, double lat, double lng);
/**
 * @brief Update an item.
 *
 * @param index Index instance.
 * @param point_index Index value.
 * @param lat lat.
 * @param lng lng.
 * @return 0 on success, -1 on error.
 */
int geo_update(GV_GeoIndex *index, size_t point_index, double lat, double lng);
/**
 * @brief Remove an item.
 *
 * @param index Index instance.
 * @param point_index Index value.
 * @return 0 on success, -1 on error.
 */
int geo_remove(GV_GeoIndex *index, size_t point_index);

int geo_radius_search(const GV_GeoIndex *index, double lat, double lng,
                          double radius_km, GV_GeoResult *results, size_t max_results);

int geo_bbox_search(const GV_GeoIndex *index, const GV_GeoBBox *bbox,
                        GV_GeoResult *results, size_t max_results);

/* Returns candidate indices for use as a pre-filter in vector search. */
int geo_get_candidates(const GV_GeoIndex *index, double lat, double lng,
                           double radius_km, size_t *out_indices, size_t max_count);

/* Haversine distance between two points in km. */
/**
 * @brief Perform the operation.
 *
 * @param lat1 lat1.
 * @param lng1 lng1.
 * @param lat2 lat2.
 * @param lng2 lng2.
 * @return Result value.
 */
double geo_distance_km(double lat1, double lng1, double lat2, double lng2);

/**
 * @brief Return the number of stored items.
 *
 * @param index Index instance.
 * @return Count value.
 */
size_t geo_count(const GV_GeoIndex *index);
/**
 * @brief Save state to a file.
 *
 * @param index Index instance.
 * @param filepath Filesystem path.
 * @return 0 on success, -1 on error.
 */
int geo_save(const GV_GeoIndex *index, const char *filepath);
GV_GeoIndex *geo_load(const char *filepath);

#ifdef __cplusplus
}
#endif
#endif
