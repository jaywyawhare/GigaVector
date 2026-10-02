# GigaVector Go client (embedded)

An idiomatic Go binding for GigaVector. It links the GigaVector C library
directly via cgo, so it runs **embedded** in your process — no server required.

## Requirements

- Go 1.21+
- A compiled GigaVector C library. From the repository root:

  ```sh
  make lib
  ```

## Usage

```go
import gv "github.com/jaywyawhare/GigaVector/clients/go"

db, err := gv.Open("", 4, gv.IndexFlat) // "" => in-memory
if err != nil {
    log.Fatal(err)
}
defer db.Close()

_ = db.AddVector([]float32{1, 0, 0, 0})
_ = db.AddVectorWithMetadata([]float32{0, 1, 0, 0}, "color", "blue")

hits, _ := db.Search([]float32{1, 0, 0, 0}, 2, gv.Euclidean)
for _, h := range hits {
    fmt.Printf("id=%d distance=%.4f\n", h.ID, h.Distance)
}
```

## Building & testing

The cgo directives in `gigavector.go` locate the headers (`../../include`) and
shared library (`../../build/lib`) relative to this package. Point the loader at
the library at run time:

```sh
cd clients/go
LD_LIBRARY_PATH=../../build/lib go test ./...
```

If your layout differs, override `CGO_CFLAGS` / `CGO_LDFLAGS` and
`LD_LIBRARY_PATH` accordingly.

## Supported API

| Method                     | Description                              |
| -------------------------- | ---------------------------------------- |
| `Open`                     | Open/create a database (in-memory or on disk) |
| `Close`                    | Release the database                     |
| `AddVector`                | Insert a vector                          |
| `AddVectorWithMetadata`    | Insert a vector with a key/value pair    |
| `Search`                   | k-nearest-neighbour search (Euclidean / Cosine / DotProduct / Manhattan) |
| `RangeSearch`              | All vectors within a radius               |
| `UpdateVector`            | Replace a vector by index                 |
| `DeleteVector`            | Remove a vector by index                  |
| `Save`                     | Write a durable snapshot (reopen with `Open`) |

This is the first non-Python client. A network client against the REST/gRPC
server can be layered on the same package later.
