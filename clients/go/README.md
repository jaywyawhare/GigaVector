# GigaVector Go client

A small Go SDK for the GigaVector HTTP (REST) API.

```go
import "gigavector"

c := gigavector.New("http://localhost:8080", gigavector.WithAPIKey("secret"))

if _, err := c.AddVector([]float32{1, 0, 0, 0}, map[string]string{"tag": "a"}); err != nil {
    log.Fatal(err)
}

results, err := c.Search([]float32{1, 0, 0, 0}, 5, gigavector.Cosine)
for _, r := range results {
    fmt.Println(r.ID, r.Distance)
}
```

## Auth

Pass `WithAPIKey(key)` — it is sent as the `X-API-Key` header. When the server is
started without an API key it runs read-only (or, with `GV_ALLOW_UNAUTH=1`,
allows unauthenticated writes for local development).

## Testing

Unit tests run against an in-process mock and need no server:

```sh
go test ./...
```

To run the end-to-end test against a real `gvserver`:

```sh
docker run -d -p 8080:8080 -e GV_DIMENSION=8 -e GV_INDEX=flat -e GV_ALLOW_UNAUTH=1 gigavector
GIGAVECTOR_TEST_ADDR=http://localhost:8080 go test -run TestIntegration ./...
```
