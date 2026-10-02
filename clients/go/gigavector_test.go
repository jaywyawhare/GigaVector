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

func TestRangeSearch(t *testing.T) {
	db, err := Open("", 2, IndexFlat)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	for _, v := range [][]float32{{0, 0}, {0, 1}, {10, 10}} {
		if err := db.AddVector(v); err != nil {
			t.Fatalf("AddVector: %v", err)
		}
	}
	hits, err := db.RangeSearch([]float32{0, 0}, 2.0, 10, Euclidean)
	if err != nil {
		t.Fatalf("RangeSearch: %v", err)
	}
	// {0,0} and {0,1} are within radius 2; {10,10} is not.
	if len(hits) != 2 {
		t.Errorf("expected 2 in-radius hits, got %d", len(hits))
	}
}

func TestSaveAndReopen(t *testing.T) {
	path := t.TempDir() + "/snap.gvdb"
	db, err := Open(path, 3, IndexFlat)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if err := db.AddVector([]float32{1, 2, 3}); err != nil {
		t.Fatalf("AddVector: %v", err)
	}
	if err := db.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	db.Close()

	reopened, err := Open(path, 3, IndexFlat)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer reopened.Close()
	hits, err := reopened.Search([]float32{1, 2, 3}, 1, Euclidean)
	if err != nil {
		t.Fatalf("Search after reopen: %v", err)
	}
	if len(hits) != 1 {
		t.Errorf("expected 1 hit after reopen, got %d", len(hits))
	}
}

func TestDeleteAndUpdate(t *testing.T) {
	db, err := Open("", 2, IndexFlat)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	if err := db.AddVector([]float32{1, 1}); err != nil {
		t.Fatalf("AddVector: %v", err)
	}
	if err := db.UpdateVector(0, []float32{2, 2}); err != nil {
		t.Fatalf("UpdateVector: %v", err)
	}
	if err := db.DeleteVector(0); err != nil {
		t.Fatalf("DeleteVector: %v", err)
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
