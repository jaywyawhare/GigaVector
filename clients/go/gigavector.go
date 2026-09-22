// Package gigavector is a Go client for the GigaVector HTTP (REST) API.
//
// Example:
//
//	c := gigavector.New("http://localhost:8080", gigavector.WithAPIKey("secret"))
//	if _, err := c.AddVector([]float32{1, 0, 0, 0}, map[string]string{"tag": "a"}); err != nil {
//	    log.Fatal(err)
//	}
//	res, err := c.Search([]float32{1, 0, 0, 0}, 5, gigavector.Euclidean)
package gigavector

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Distance metrics accepted by the server.
const (
	Euclidean  = "euclidean"
	Cosine     = "cosine"
	DotProduct = "dot_product"
	Manhattan  = "manhattan"
	Hamming    = "hamming"
)

// Client talks to a GigaVector server.
type Client struct {
	baseURL string
	apiKey  string
	http    *http.Client
}

// Option configures a Client.
type Option func(*Client)

// WithAPIKey sets the API key sent as the X-API-Key header.
func WithAPIKey(key string) Option { return func(c *Client) { c.apiKey = key } }

// WithHTTPClient overrides the underlying *http.Client.
func WithHTTPClient(h *http.Client) Option { return func(c *Client) { c.http = h } }

// New creates a client for the server at baseURL (e.g. "http://localhost:8080").
func New(baseURL string, opts ...Option) *Client {
	c := &Client{baseURL: baseURL, http: &http.Client{Timeout: 30 * time.Second}}
	for _, o := range opts {
		o(c)
	}
	return c
}

// APIError is returned for non-2xx responses.
type APIError struct {
	Status  int
	Code    string `json:"error"`
	Message string `json:"message"`
}

func (e *APIError) Error() string {
	return fmt.Sprintf("gigavector: %d %s: %s", e.Status, e.Code, e.Message)
}

func (c *Client) do(method, path string, reqBody, respOut any) error {
	var body io.Reader
	if reqBody != nil {
		b, err := json.Marshal(reqBody)
		if err != nil {
			return err
		}
		body = bytes.NewReader(b)
	}
	req, err := http.NewRequest(method, c.baseURL+path, body)
	if err != nil {
		return err
	}
	if reqBody != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.apiKey != "" {
		req.Header.Set("X-API-Key", c.apiKey)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode >= 400 {
		apiErr := &APIError{Status: resp.StatusCode}
		_ = json.Unmarshal(data, apiErr) // best effort
		return apiErr
	}
	if respOut != nil && len(data) > 0 {
		return json.Unmarshal(data, respOut)
	}
	return nil
}

// Health is the server's health snapshot.
type Health struct {
	Status      string `json:"status"`
	VectorCount int    `json:"vector_count"`
}

// Health returns the server health.
func (c *Client) Health() (*Health, error) {
	var h Health
	if err := c.do(http.MethodGet, "/health", nil, &h); err != nil {
		return nil, err
	}
	return &h, nil
}

// AddResponse is the result of adding vectors.
type AddResponse struct {
	Success  bool  `json:"success"`
	Inserted int   `json:"inserted"`
	Indices  []int `json:"indices"`
}

// AddVector inserts a single vector with optional metadata.
func (c *Client) AddVector(data []float32, metadata map[string]string) (*AddResponse, error) {
	body := map[string]any{"data": data}
	if len(metadata) > 0 {
		body["metadata"] = metadata
	}
	var out AddResponse
	if err := c.do(http.MethodPost, "/vectors", body, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// SearchResult is one neighbour returned by Search.
type SearchResult struct {
	ID       int       `json:"id"`
	Distance float64   `json:"distance"`
	Data     []float32 `json:"data"`
}

type searchResponse struct {
	Results []SearchResult `json:"results"`
	Count   int            `json:"count"`
}

// Search returns the k nearest neighbours of query using the given distance
// metric (use the package constants; "" defaults to Euclidean).
func (c *Client) Search(query []float32, k int, distance string) ([]SearchResult, error) {
	if distance == "" {
		distance = Euclidean
	}
	body := map[string]any{"query": query, "k": k, "distance": distance}
	var out searchResponse
	if err := c.do(http.MethodPost, "/search", body, &out); err != nil {
		return nil, err
	}
	return out.Results, nil
}

// Stats returns raw server statistics as a decoded JSON object.
func (c *Client) Stats() (map[string]any, error) {
	var out map[string]any
	if err := c.do(http.MethodGet, "/stats", nil, &out); err != nil {
		return nil, err
	}
	return out, nil
}
