// Package gigavector is an embedded Go client for GigaVector: it links the
// GigaVector C library directly via cgo (no server required) and exposes an
// idiomatic Go API over the core vector database.
//
// Build/test requires the compiled C library. From the repository root:
//
//	make lib
//	cd clients/go && go test ./...
//
// The cgo directives below locate the headers and shared library relative to
// this package; set CGO_LDFLAGS / LD_LIBRARY_PATH if your layout differs.
package gigavector

/*
#cgo CFLAGS: -I${SRCDIR}/../../include
#cgo LDFLAGS: -L${SRCDIR}/../../build/lib -lGigaVector -lm
#include <stdlib.h>
#include "storage/database.h"
#include "search/distance.h"
#include "core/types.h"
*/
import "C"

import (
	"errors"
	"runtime"
	"unsafe"
)

// IndexType selects the vector index implementation.
type IndexType int

const (
	IndexKDTree IndexType = C.GV_INDEX_TYPE_KDTREE
	IndexHNSW   IndexType = C.GV_INDEX_TYPE_HNSW
	IndexIVFPQ  IndexType = C.GV_INDEX_TYPE_IVFPQ
	IndexFlat   IndexType = C.GV_INDEX_TYPE_FLAT
)

// Distance selects the similarity metric used by Search.
type Distance int

const (
	Euclidean  Distance = C.GV_DISTANCE_EUCLIDEAN
	Cosine     Distance = C.GV_DISTANCE_COSINE
	DotProduct Distance = C.GV_DISTANCE_DOT_PRODUCT
	Manhattan  Distance = C.GV_DISTANCE_MANHATTAN
)

// DB is a handle to an open GigaVector database.
type DB struct {
	ptr *C.GV_Database
	dim int
}

// SearchResult is one hit returned by Search.
type SearchResult struct {
	ID       uint64
	Distance float32
}

// ErrClosed is returned when operating on a closed database.
var ErrClosed = errors.New("gigavector: database is closed")

// Open opens (or creates) a database. Pass path == "" for an in-memory database.
func Open(path string, dimension int, index IndexType) (*DB, error) {
	var cpath *C.char
	if path != "" {
		cpath = C.CString(path)
		defer C.free(unsafe.Pointer(cpath))
	}
	ptr := C.db_open(cpath, C.size_t(dimension), C.GV_IndexType(index))
	if ptr == nil {
		return nil, errors.New("gigavector: db_open failed")
	}
	db := &DB{ptr: ptr, dim: dimension}
	runtime.SetFinalizer(db, (*DB).Close)
	return db, nil
}

// Close releases the database. Safe to call more than once.
func (db *DB) Close() error {
	if db.ptr != nil {
		C.db_close(db.ptr)
		db.ptr = nil
		runtime.SetFinalizer(db, nil)
	}
	return nil
}

func (db *DB) cvec(v []float32) *C.float {
	if len(v) == 0 {
		return nil
	}
	return (*C.float)(unsafe.Pointer(&v[0]))
}

// AddVector inserts a vector. Its length must equal the database dimension.
func (db *DB) AddVector(vec []float32) error {
	if db.ptr == nil {
		return ErrClosed
	}
	if len(vec) != db.dim {
		return errors.New("gigavector: vector length does not match dimension")
	}
	if rc := C.db_add_vector(db.ptr, db.cvec(vec), C.size_t(len(vec))); rc != 0 {
		return errors.New("gigavector: db_add_vector failed")
	}
	return nil
}

// AddVectorWithMetadata inserts a vector with a single key/value metadata pair.
func (db *DB) AddVectorWithMetadata(vec []float32, key, value string) error {
	if db.ptr == nil {
		return ErrClosed
	}
	if len(vec) != db.dim {
		return errors.New("gigavector: vector length does not match dimension")
	}
	ckey, cval := C.CString(key), C.CString(value)
	defer C.free(unsafe.Pointer(ckey))
	defer C.free(unsafe.Pointer(cval))
	if rc := C.db_add_vector_with_metadata(db.ptr, db.cvec(vec), C.size_t(len(vec)), ckey, cval); rc != 0 {
		return errors.New("gigavector: db_add_vector_with_metadata failed")
	}
	return nil
}

// Search returns the k nearest neighbours of query under the given metric.
func (db *DB) Search(query []float32, k int, metric Distance) ([]SearchResult, error) {
	if db.ptr == nil {
		return nil, ErrClosed
	}
	if len(query) != db.dim {
		return nil, errors.New("gigavector: query length does not match dimension")
	}
	if k <= 0 {
		return nil, nil
	}
	out := make([]C.GV_SearchResult, k)
	found := C.db_search(db.ptr, db.cvec(query), C.size_t(k), &out[0], C.GV_DistanceType(metric))
	if found < 0 {
		return nil, errors.New("gigavector: db_search failed")
	}
	results := make([]SearchResult, int(found))
	for i := 0; i < int(found); i++ {
		results[i] = SearchResult{
			ID:       uint64(out[i].id),
			Distance: float32(out[i].distance),
		}
	}
	return results, nil
}
