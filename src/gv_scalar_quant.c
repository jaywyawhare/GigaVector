#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

#include "gigavector/gv_scalar_quant.h"
#include "gigavector/gv_distance.h"

size_t gv_scalar_quant_bytes_needed(size_t dimension, uint8_t bits) {
    if (bits == 0 || dimension == 0) {
        return 0;
    }
    return (dimension * bits + 7) / 8;
}

static void find_min_max_global(const float *data, size_t count, size_t dimension,
                                 float *min_out, float *max_out) {
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < dimension; ++j) {
            float val = data[i * dimension + j];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
    }
    
    *min_out = min_val;
    *max_out = max_val;
}

static void find_min_max_per_dimension(const float *data, size_t count, size_t dimension,
                                       float *min_vals, float *max_vals) {
    for (size_t j = 0; j < dimension; ++j) {
        min_vals[j] = FLT_MAX;
        max_vals[j] = -FLT_MAX;
    }
    
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < dimension; ++j) {
            float val = data[i * dimension + j];
            if (val < min_vals[j]) min_vals[j] = val;
            if (val > max_vals[j]) max_vals[j] = val;
        }
    }
}

GV_ScalarQuantVector *gv_scalar_quantize_train(const float *data, size_t count, size_t dimension,
                                                const GV_ScalarQuantConfig *config) {
    if (data == NULL || count == 0 || dimension == 0 || config == NULL) {
        return NULL;
    }
    
    if (config->bits != 4 && config->bits != 8 && config->bits != 16) {
        return NULL;
    }
    
    GV_ScalarQuantVector *sqv = (GV_ScalarQuantVector *)malloc(sizeof(GV_ScalarQuantVector));
    if (sqv == NULL) {
        return NULL;
    }
    
    sqv->dimension = dimension;
    sqv->bits = config->bits;
    sqv->per_dimension = config->per_dimension;
    sqv->bytes_per_vector = gv_scalar_quant_bytes_needed(dimension, config->bits);
    
    if (sqv->per_dimension) {
        sqv->min_vals = (float *)malloc(dimension * sizeof(float));
        sqv->max_vals = (float *)malloc(dimension * sizeof(float));
        if (sqv->min_vals == NULL || sqv->max_vals == NULL) {
            free(sqv->min_vals);
            free(sqv->max_vals);
            free(sqv);
            return NULL;
        }
        find_min_max_per_dimension(data, count, dimension, sqv->min_vals, sqv->max_vals);
    } else {
        sqv->min_vals = (float *)malloc(sizeof(float));
        sqv->max_vals = (float *)malloc(sizeof(float));
        if (sqv->min_vals == NULL || sqv->max_vals == NULL) {
            free(sqv->min_vals);
            free(sqv->max_vals);
            free(sqv);
            return NULL;
        }
        find_min_max_global(data, count, dimension, sqv->min_vals, sqv->max_vals);
    }
    
    sqv->quantized = NULL;
    return sqv;
}

GV_ScalarQuantVector *gv_scalar_quantize(const float *data, size_t dimension,
                                         const GV_ScalarQuantConfig *config) {
    if (data == NULL || dimension == 0 || config == NULL) {
        return NULL;
    }
    
    if (config->bits != 4 && config->bits != 8 && config->bits != 16) {
        return NULL;
    }
    
    GV_ScalarQuantVector *sqv = (GV_ScalarQuantVector *)malloc(sizeof(GV_ScalarQuantVector));
    if (sqv == NULL) {
        return NULL;
    }
    
    sqv->dimension = dimension;
    sqv->bits = config->bits;
    sqv->per_dimension = config->per_dimension;
    sqv->bytes_per_vector = gv_scalar_quant_bytes_needed(dimension, config->bits);
    
    sqv->quantized = (uint8_t *)calloc(sqv->bytes_per_vector, sizeof(uint8_t));
    if (sqv->quantized == NULL) {
        free(sqv);
        return NULL;
    }
    
    if (sqv->per_dimension) {
        sqv->min_vals = (float *)malloc(dimension * sizeof(float));
        sqv->max_vals = (float *)malloc(dimension * sizeof(float));
        if (sqv->min_vals == NULL || sqv->max_vals == NULL) {
            free(sqv->quantized);
            free(sqv->min_vals);
            free(sqv->max_vals);
            free(sqv);
            return NULL;
        }
        
        for (size_t i = 0; i < dimension; ++i) {
            sqv->min_vals[i] = data[i];
            sqv->max_vals[i] = data[i];
        }
    } else {
        sqv->min_vals = (float *)malloc(sizeof(float));
        sqv->max_vals = (float *)malloc(sizeof(float));
        if (sqv->min_vals == NULL || sqv->max_vals == NULL) {
            free(sqv->quantized);
            free(sqv->min_vals);
            free(sqv->max_vals);
            free(sqv);
            return NULL;
        }
        
        sqv->min_vals[0] = data[0];
        sqv->max_vals[0] = data[0];
        for (size_t i = 1; i < dimension; ++i) {
            if (data[i] < sqv->min_vals[0]) sqv->min_vals[0] = data[i];
            if (data[i] > sqv->max_vals[0]) sqv->max_vals[0] = data[i];
        }
    }
    
    size_t max_quant = (1ULL << config->bits) - 1;
    
    for (size_t i = 0; i < dimension; ++i) {
        float min_val = sqv->per_dimension ? sqv->min_vals[i] : sqv->min_vals[0];
        float max_val = sqv->per_dimension ? sqv->max_vals[i] : sqv->max_vals[0];
        float range = max_val - min_val;
        
        if (range <= 0.0f) {
            continue;
        }
        
        float normalized = (data[i] - min_val) / range;
        normalized = (normalized < 0.0f) ? 0.0f : (normalized > 1.0f) ? 1.0f : normalized;
        size_t quantized_val = (size_t)(normalized * max_quant + 0.5f);
        if (quantized_val > max_quant) quantized_val = max_quant;
        
        if (config->bits == 4) {
            size_t byte_idx = i / 2;
            size_t bit_offset = (i % 2) * 4;
            sqv->quantized[byte_idx] |= (uint8_t)(quantized_val << (4 - bit_offset));
        } else if (config->bits == 8) {
            sqv->quantized[i] = (uint8_t)quantized_val;
        } else if (config->bits == 16) {
            ((uint16_t *)sqv->quantized)[i] = (uint16_t)quantized_val;
        }
    }
    
    return sqv;
}

int gv_scalar_dequantize(const GV_ScalarQuantVector *sqv, float *output) {
    if (sqv == NULL || output == NULL || sqv->quantized == NULL) {
        return -1;
    }
    
    size_t max_quant = (1ULL << sqv->bits) - 1;
    
    for (size_t i = 0; i < sqv->dimension; ++i) {
        size_t quantized_val = 0;
        
        if (sqv->bits == 4) {
            size_t byte_idx = i / 2;
            size_t bit_offset = (i % 2) * 4;
            quantized_val = (sqv->quantized[byte_idx] >> (4 - bit_offset)) & 0x0F;
        } else if (sqv->bits == 8) {
            quantized_val = sqv->quantized[i];
        } else if (sqv->bits == 16) {
            quantized_val = ((uint16_t *)sqv->quantized)[i];
        }
        
        float min_val = sqv->per_dimension ? sqv->min_vals[i] : sqv->min_vals[0];
        float max_val = sqv->per_dimension ? sqv->max_vals[i] : sqv->max_vals[0];
        float range = max_val - min_val;
        
        float normalized = (float)quantized_val / (float)max_quant;
        output[i] = min_val + normalized * range;
    }
    
    return 0;
}

float gv_scalar_quant_distance(const float *query, const GV_ScalarQuantVector *sqv, int distance_type) {
    if (query == NULL || sqv == NULL || sqv->quantized == NULL) {
        return -1.0f;
    }
    
    float *dequantized = (float *)malloc(sqv->dimension * sizeof(float));
    if (dequantized == NULL) {
        return -1.0f;
    }
    
    if (gv_scalar_dequantize(sqv, dequantized) != 0) {
        free(dequantized);
        return -1.0f;
    }
    
    GV_Vector query_vec = {
        .dimension = sqv->dimension,
        .data = (float *)query,
        .metadata = NULL
    };
    
    GV_Vector dequant_vec = {
        .dimension = sqv->dimension,
        .data = dequantized,
        .metadata = NULL
    };
    
    float distance = gv_distance(&query_vec, &dequant_vec, (GV_DistanceType)distance_type);
    
    free(dequantized);
    return distance;
}

void gv_scalar_quant_vector_destroy(GV_ScalarQuantVector *sqv) {
    if (sqv == NULL) {
        return;
    }
    if (sqv->quantized != NULL) {
        free(sqv->quantized);
    }
    if (sqv->min_vals != NULL) {
        free(sqv->min_vals);
    }
    if (sqv->max_vals != NULL) {
        free(sqv->max_vals);
    }
    free(sqv);
}

