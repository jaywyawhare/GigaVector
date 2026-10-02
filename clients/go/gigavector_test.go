package gigavector

import "testing"

func TestOpenAddSearch(t *testing.T) {
	db, err := Open("", 4, IndexFlat)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	vectors := [][]float32{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0.9, 0.1, 0, 0},
	}
	for _, v := range vectors {
		if err := db.AddVector(v); err != nil {
			t.Fatalf("AddVector: %v", err)
		}
	}

	hits, err := db.Search([]float32{1, 0, 0, 0}, 2, Euclidean)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 2 {
		t.Fatalf("expected 2 hits, got %d", len(hits))
	}
	// The exact match (vector 0) must be the nearest neighbour.
	if hits[0].ID != 0 {
		t.Errorf("expected nearest id 0, got %d", hits[0].ID)
	}
	if hits[0].Distance > hits[1].Distance {
		t.Errorf("results not ordered by distance: %v", hits)
	}
}

func TestAddVectorWithMetadata(t *testing.T) {
	db, err := Open("", 2, IndexFlat)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	if err := db.AddVectorWithMetadata([]float32{1, 2}, "color", "red"); err != nil {
		t.Fatalf("AddVectorWithMetadata: %v", err)
	}
	hits, err := db.Search([]float32{1, 2}, 1, Euclidean)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 1 {
		t.Fatalf("expected 1 hit, got %d", len(hits))
	}
}

func TestDimensionMismatch(t *testing.T) {
	db, err := Open("", 3, IndexFlat)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	if err := db.AddVector([]float32{1, 2}); err == nil {
		t.Error("expected dimension-mismatch error, got nil")
	}
}

func TestUseAfterClose(t *testing.T) {
	db, err := Open("", 2, IndexFlat)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	db.Close()
	if err := db.AddVector([]float32{1, 2}); err != ErrClosed {
		t.Errorf("expected ErrClosed, got %v", err)
	}
}
