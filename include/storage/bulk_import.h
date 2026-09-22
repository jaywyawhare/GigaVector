#ifndef GIGAVECTOR_GV_BULK_IMPORT_H
#define GIGAVECTOR_GV_BULK_IMPORT_H

#include <stddef.h>

#include "storage/database.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file bulk_import.h
 * @brief Dependency-free bulk loading of vectors into a database from CSV/JSONL.
 */

typedef struct {
    size_t imported;   /**< Rows successfully added. */
    size_t skipped;    /**< Blank/comment rows skipped. */
    size_t errors;     /**< Rows that failed to parse or add. */
} GV_ImportReport;

/**
 * @brief Import vectors from a CSV file.
 *
 * Each data row holds exactly @p db's dimension numeric columns, in order. If
 * @p id_column >= 0, that column is treated as a string primary key (added via
 * db_add_vector_with_id) and excluded from the vector.
 *
 * @param db         Target database.
 * @param path       CSV file path.
 * @param delimiter  Field delimiter (0 defaults to ',').
 * @param has_header Non-zero to skip the first line.
 * @param id_column  0-based column holding a string id, or -1 for none.
 * @param out        Optional report (may be NULL).
 * @return 0 on success (file opened and processed), -1 on open/argument error.
 */
int db_import_csv(GV_Database *db, const char *path, char delimiter,
                  int has_header, int id_column, GV_ImportReport *out);

/**
 * @brief Import vectors from a JSONL file (one JSON object per line).
 *
 * Each object must have a "vector" array (length = db dimension); optional
 * "id" (string) and a flat "metadata" object of string values are honoured.
 *
 * @return 0 on success, -1 on open/argument error.
 */
int db_import_jsonl(GV_Database *db, const char *path, GV_ImportReport *out);

#ifdef __cplusplus
}
#endif

#endif
