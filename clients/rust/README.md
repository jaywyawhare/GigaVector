# GigaVector Rust client (REST)

An idiomatic, **dependency-free** Rust client for the GigaVector REST API. It
speaks HTTP/1.1 and JSON using only the standard library, and keeps a bounded
pool of **keep-alive TCP connections** so repeated calls reuse sockets instead
of reconnecting - connection pooling at the protocol level.

## Requirements

- Rust 1.70+ (stable). No third-party crates.
- A running GigaVector REST server (see the project server docs).

## Usage

```rust
use gigavector::{Client, Distance};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Default pool: up to 8 idle keep-alive connections.
    let client = Client::new("http://localhost:8080")?;

    client.add_vector(&[0.1, 0.2, 0.3, 0.4], &[("tag", "red")])?;

    let hits = client.search(&[0.1, 0.2, 0.3, 0.4], 5, Distance::Euclidean)?;
    for h in hits {
        println!("distance={} metadata={:?}", h.distance, h.metadata);
    }

    let stats = client.stats()?;
    println!("{} vectors, dim {}", stats.vector_count, stats.dimension);
    Ok(())
}
```

### Configuration

```rust
use std::time::Duration;
use gigavector::Client;

let client = Client::builder("http://localhost:8080")
    .api_key("secret")               // sent as X-API-Key
    .max_idle_connections(32)        // pool cap (default 8)
    .timeout(Duration::from_secs(5)) // per-socket read/write timeout
    .build()?;
```

## API

| Method | REST endpoint |
| --- | --- |
| `health()` | `GET /health` |
| `stats()` | `GET /stats` |
| `add_vector(data, metadata)` | `POST /vectors` |
| `search(query, k, distance)` | `POST /search` |
| `range_search(query, radius, max, distance)` | `POST /search/range` |
| `save(path)` | `POST /save` |

`Distance` is one of `Euclidean`, `Cosine`, `DotProduct`, `Manhattan`.

Errors surface as the `gigavector::Error` enum (`Http`, `Io`, `Parse`,
`Protocol`, `BadUrl`).

## Connection pooling

Each request checks an idle connection out of the pool (or opens a new one),
sends the request with `Connection: keep-alive`, reads the response, and - when
the server keeps the connection open - returns it to the pool for reuse. A
request that fails on a stale pooled socket is transparently retried once on a
fresh connection. `client.pooled_connections()` reports the current idle count.

## Testing

```sh
cargo test     # unit tests (JSON, HTTP framing, pool) - no server needed
cargo clippy   # lint-clean
```
