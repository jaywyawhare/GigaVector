package gigavector

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
)

// Unit tests run against an httptest mock (no C server needed); they verify the
// client's request shaping, auth header, and response decoding.

func TestHealth(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" || r.Method != http.MethodGet {
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"status":"healthy","vector_count":3}`))
	}))
	defer srv.Close()

	h, err := New(srv.URL).Health()
	if err != nil {
		t.Fatal(err)
	}
	if h.Status != "healthy" || h.VectorCount != 3 {
		t.Fatalf("got %+v", h)
	}
}

func TestAddVectorSendsDataAndAuth(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("X-API-Key"); got != "secret" {
			t.Errorf("missing/wrong API key: %q", got)
		}
		var body struct {
			Data     []float32         `json:"data"`
			Metadata map[string]string `json:"metadata"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if len(body.Data) != 4 || body.Data[0] != 1 {
			t.Errorf("bad data: %v", body.Data)
		}
		if body.Metadata["tag"] != "a" {
			t.Errorf("bad metadata: %v", body.Metadata)
		}
		_, _ = w.Write([]byte(`{"success":true,"inserted":1,"indices":[7]}`))
	}))
	defer srv.Close()

	c := New(srv.URL, WithAPIKey("secret"))
	resp, err := c.AddVector([]float32{1, 0, 0, 0}, map[string]string{"tag": "a"})
	if err != nil {
		t.Fatal(err)
	}
	if !resp.Success || resp.Inserted != 1 || len(resp.Indices) != 1 || resp.Indices[0] != 7 {
		t.Fatalf("got %+v", resp)
	}
}

func TestSearchDecodesResults(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Query    []float32 `json:"query"`
			K        int       `json:"k"`
			Distance string    `json:"distance"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		if body.K != 2 || body.Distance != Cosine {
			t.Errorf("bad search req: %+v", body)
		}
		_, _ = w.Write([]byte(`{"results":[{"id":1,"distance":0.1,"data":[1,0]},{"id":2,"distance":0.4,"data":[0,1]}],"count":2}`))
	}))
	defer srv.Close()

	res, err := New(srv.URL).Search([]float32{1, 0}, 2, Cosine)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 2 || res[0].ID != 1 || res[0].Distance != 0.1 {
		t.Fatalf("got %+v", res)
	}
}

func TestAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"dimension_mismatch","message":"bad dim"}`))
	}))
	defer srv.Close()

	_, err := New(srv.URL).AddVector([]float32{1}, nil)
	apiErr, ok := err.(*APIError)
	if !ok {
		t.Fatalf("want *APIError, got %T (%v)", err, err)
	}
	if apiErr.Status != 400 || apiErr.Code != "dimension_mismatch" {
		t.Fatalf("got %+v", apiErr)
	}
}

// Optional end-to-end test against a running gvserver. Set GIGAVECTOR_TEST_ADDR
// (e.g. http://localhost:8080) to enable; skipped otherwise.
func TestIntegration(t *testing.T) {
	addr := os.Getenv("GIGAVECTOR_TEST_ADDR")
	if addr == "" {
		t.Skip("set GIGAVECTOR_TEST_ADDR to run the integration test")
	}
	c := New(addr, WithAPIKey(os.Getenv("GIGAVECTOR_TEST_KEY")))
	if _, err := c.Health(); err != nil {
		t.Fatal(err)
	}
	if _, err := c.AddVector([]float32{1, 0, 0, 0, 0, 0, 0, 0}, nil); err != nil {
		t.Fatal(err)
	}
	res, err := c.Search([]float32{1, 0, 0, 0, 0, 0, 0, 0}, 1, Euclidean)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) < 1 {
		t.Fatalf("expected >=1 result, got %d", len(res))
	}
}
