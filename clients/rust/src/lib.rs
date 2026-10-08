//! Official Rust client for the [GigaVector](https://github.com/jaywyawhare/GigaVector)
//! REST API.
//!
//! The client is dependency-free (standard library only) and maintains a
//! bounded pool of keep-alive TCP connections, so repeated requests reuse
//! sockets instead of reconnecting on every call — connection pooling at the
//! protocol level.
//!
//! ```no_run
//! use gigavector::{Client, Distance};
//!
//! let client = Client::new("http://localhost:8080").unwrap();
//! client.add_vector(&[0.1, 0.2, 0.3, 0.4], &[("tag", "red")]).unwrap();
//! let hits = client.search(&[0.1, 0.2, 0.3, 0.4], 5, Distance::Euclidean).unwrap();
//! for h in hits {
//!     println!("distance={} metadata={:?}", h.distance, h.metadata);
//! }
//! ```

mod http;
mod json;

pub use json::Json;

use std::collections::BTreeMap;
use std::io::Write;
use std::time::Duration;

use http::ConnectionPool;

/// Errors returned by the client.
#[derive(Debug)]
pub enum Error {
    /// Malformed base URL.
    BadUrl(String),
    /// Network/IO failure.
    Io(String),
    /// Response could not be parsed as the expected shape.
    Parse(String),
    /// HTTP-level protocol violation.
    Protocol(String),
    /// Non-2xx HTTP status; carries the status code and server message.
    Http(u16, String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::BadUrl(s) => write!(f, "bad url: {}", s),
            Error::Io(s) => write!(f, "io error: {}", s),
            Error::Parse(s) => write!(f, "parse error: {}", s),
            Error::Protocol(s) => write!(f, "protocol error: {}", s),
            Error::Http(code, msg) => write!(f, "http {}: {}", code, msg),
        }
    }
}

impl std::error::Error for Error {}

/// Distance metric for searches. Serialized to the wire string the server expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distance {
    Euclidean,
    Cosine,
    DotProduct,
    Manhattan,
}

impl Distance {
    pub fn as_str(self) -> &'static str {
        match self {
            Distance::Euclidean => "euclidean",
            Distance::Cosine => "cosine",
            Distance::DotProduct => "dot_product",
            Distance::Manhattan => "manhattan",
        }
    }
}

/// One search hit.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub distance: f32,
    pub data: Option<Vec<f32>>,
    pub metadata: BTreeMap<String, String>,
}

/// Database statistics (`GET /stats`).
#[derive(Debug, Clone)]
pub struct Stats {
    pub vector_count: u64,
    pub dimension: u64,
}

/// A pooled GigaVector REST client.
pub struct Client {
    host: String,
    port: u16,
    api_key: Option<String>,
    pool: ConnectionPool,
}

impl Client {
    /// Create a client for `base_url` (e.g. `"http://localhost:8080"`), with a
    /// default idle-connection cap of 8 and no socket timeout.
    pub fn new(base_url: &str) -> Result<Client, Error> {
        Client::builder(base_url).build()
    }

    /// Start configuring a client.
    pub fn builder(base_url: &str) -> ClientBuilder {
        ClientBuilder {
            base_url: base_url.to_string(),
            api_key: None,
            max_idle: 8,
            timeout: None,
        }
    }

    /// Number of currently pooled idle connections.
    pub fn pooled_connections(&self) -> usize {
        self.pool.idle_count()
    }

    fn request(&self, method: &str, path: &str, body: Option<Json>) -> Result<Json, Error> {
        let payload = body.map(|b| b.to_string()).unwrap_or_default();
        let bytes = payload.into_bytes();

        // Try once on a pooled connection; if that fails (stale socket), retry
        // once on a fresh connection.
        let mut last_err: Option<Error> = None;
        for attempt in 0..2 {
            let mut stream = match self.pool.checkout() {
                Ok(s) => s,
                Err(e) => {
                    last_err = Some(e);
                    continue;
                }
            };

            let req =
                http::build_request(method, &self.host, self.port, path, self.api_key.as_deref(), &bytes);

            if let Err(e) = stream.write_all(&req).and_then(|_| stream.flush()) {
                last_err = Some(Error::Io(e.to_string()));
                // Do not return this (possibly dead) connection to the pool.
                let _ = attempt;
                continue;
            }

            let resp = match http::read_response(&mut stream) {
                Ok(r) => r,
                Err(e) => {
                    last_err = Some(e);
                    continue;
                }
            };

            if resp.keep_alive {
                self.pool.checkin(stream);
            }

            let parsed = if resp.body.trim().is_empty() {
                Json::Null
            } else {
                Json::parse(&resp.body).map_err(Error::Parse)?
            };

            if !(200..300).contains(&resp.status) {
                let msg = parsed
                    .get("message")
                    .and_then(|m| m.as_str())
                    .or_else(|| parsed.get("error").and_then(|m| m.as_str()))
                    .unwrap_or("request failed")
                    .to_string();
                return Err(Error::Http(resp.status, msg));
            }
            return Ok(parsed);
        }
        Err(last_err.unwrap_or_else(|| Error::Protocol("request failed".into())))
    }

    /// `GET /health` — returns the raw JSON (status + vector_count).
    pub fn health(&self) -> Result<Json, Error> {
        self.request("GET", "/health", None)
    }

    /// `GET /stats`.
    pub fn stats(&self) -> Result<Stats, Error> {
        let v = self.request("GET", "/stats", None)?;
        Ok(Stats {
            vector_count: v.get("vector_count").and_then(|n| n.as_f64()).unwrap_or(0.0) as u64,
            dimension: v.get("dimension").and_then(|n| n.as_f64()).unwrap_or(0.0) as u64,
        })
    }

    /// `POST /vectors` — insert a single vector with optional metadata.
    /// Returns the assigned vector index.
    pub fn add_vector(&self, data: &[f32], metadata: &[(&str, &str)]) -> Result<u64, Error> {
        let mut obj = BTreeMap::new();
        obj.insert("data".to_string(), floats_to_json(data));
        if !metadata.is_empty() {
            let mut m = BTreeMap::new();
            for (k, val) in metadata {
                m.insert(k.to_string(), Json::Str(val.to_string()));
            }
            obj.insert("metadata".to_string(), Json::Obj(m));
        }
        let resp = self.request("POST", "/vectors", Some(Json::Obj(obj)))?;
        Ok(resp.get("index").and_then(|n| n.as_f64()).unwrap_or(0.0) as u64)
    }

    /// `POST /search` — k-NN search.
    pub fn search(&self, query: &[f32], k: usize, distance: Distance) -> Result<Vec<SearchResult>, Error> {
        let mut obj = BTreeMap::new();
        obj.insert("query".to_string(), floats_to_json(query));
        obj.insert("k".to_string(), Json::Num(k as f64));
        obj.insert("distance".to_string(), Json::Str(distance.as_str().to_string()));
        let resp = self.request("POST", "/search", Some(Json::Obj(obj)))?;
        parse_results(&resp)
    }

    /// `POST /search/range` — all vectors within `radius`.
    pub fn range_search(
        &self,
        query: &[f32],
        radius: f32,
        max_results: usize,
        distance: Distance,
    ) -> Result<Vec<SearchResult>, Error> {
        let mut obj = BTreeMap::new();
        obj.insert("query".to_string(), floats_to_json(query));
        obj.insert("radius".to_string(), Json::Num(radius as f64));
        obj.insert("max_results".to_string(), Json::Num(max_results as f64));
        obj.insert("distance".to_string(), Json::Str(distance.as_str().to_string()));
        let resp = self.request("POST", "/search/range", Some(Json::Obj(obj)))?;
        parse_results(&resp)
    }

    /// `POST /save` — persist the database to `path` on the server.
    pub fn save(&self, path: &str) -> Result<(), Error> {
        let mut obj = BTreeMap::new();
        obj.insert("path".to_string(), Json::Str(path.to_string()));
        self.request("POST", "/save", Some(Json::Obj(obj)))?;
        Ok(())
    }
}

/// Builder for [`Client`].
pub struct ClientBuilder {
    base_url: String,
    api_key: Option<String>,
    max_idle: usize,
    timeout: Option<Duration>,
}

impl ClientBuilder {
    /// Send `X-API-Key` on every request.
    pub fn api_key(mut self, key: &str) -> Self {
        self.api_key = Some(key.to_string());
        self
    }
    /// Maximum idle connections kept in the pool (default 8).
    pub fn max_idle_connections(mut self, n: usize) -> Self {
        self.max_idle = n;
        self
    }
    /// Per-socket read/write timeout.
    pub fn timeout(mut self, d: Duration) -> Self {
        self.timeout = Some(d);
        self
    }

    pub fn build(self) -> Result<Client, Error> {
        let (host, port) = parse_base_url(&self.base_url)?;
        Ok(Client {
            pool: ConnectionPool::new(host.clone(), port, self.max_idle, self.timeout),
            host,
            port,
            api_key: self.api_key,
        })
    }
}

fn floats_to_json(data: &[f32]) -> Json {
    Json::Arr(data.iter().map(|&x| Json::Num(x as f64)).collect())
}

fn parse_results(resp: &Json) -> Result<Vec<SearchResult>, Error> {
    let arr = match resp.get("results").and_then(|r| r.as_array()) {
        Some(a) => a,
        None => return Ok(Vec::new()),
    };
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let distance = item.get("distance").and_then(|d| d.as_f64()).unwrap_or(0.0) as f32;
        let data = item.get("data").and_then(|d| d.as_array()).map(|a| {
            a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect::<Vec<f32>>()
        });
        let mut metadata = BTreeMap::new();
        if let Some(Json::Obj(m)) = item.get("metadata") {
            for (k, v) in m {
                if let Some(s) = v.as_str() {
                    metadata.insert(k.clone(), s.to_string());
                }
            }
        }
        out.push(SearchResult { distance, data, metadata });
    }
    Ok(out)
}

/// Parse `http://host:port` (or `host:port`, `host`) into (host, port).
fn parse_base_url(url: &str) -> Result<(String, u16), Error> {
    let rest = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .unwrap_or(url);
    // Drop any trailing path.
    let authority = rest.split('/').next().unwrap_or(rest);
    if authority.is_empty() {
        return Err(Error::BadUrl(url.to_string()));
    }
    if let Some((host, port)) = authority.rsplit_once(':') {
        let port: u16 = port
            .parse()
            .map_err(|_| Error::BadUrl(format!("invalid port in {}", url)))?;
        Ok((host.to_string(), port))
    } else {
        Ok((authority.to_string(), 8080))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_url_parsing() {
        assert_eq!(parse_base_url("http://localhost:8080").unwrap(), ("localhost".into(), 8080));
        assert_eq!(parse_base_url("https://db.example.com:443/x").unwrap(), ("db.example.com".into(), 443));
        assert_eq!(parse_base_url("127.0.0.1:9000").unwrap(), ("127.0.0.1".into(), 9000));
        assert_eq!(parse_base_url("myhost").unwrap(), ("myhost".into(), 8080));
        assert!(parse_base_url("http://host:notaport").is_err());
    }

    #[test]
    fn distance_strings() {
        assert_eq!(Distance::Euclidean.as_str(), "euclidean");
        assert_eq!(Distance::DotProduct.as_str(), "dot_product");
    }

    #[test]
    fn builds_add_vector_payload() {
        // Verify the JSON body we would send for add_vector.
        let mut obj = BTreeMap::new();
        obj.insert("data".to_string(), floats_to_json(&[1.0, 2.5]));
        let mut m = BTreeMap::new();
        m.insert("tag".to_string(), Json::Str("red".to_string()));
        obj.insert("metadata".to_string(), Json::Obj(m));
        assert_eq!(Json::Obj(obj).to_string(), r#"{"data":[1,2.5],"metadata":{"tag":"red"}}"#);
    }

    #[test]
    fn parses_search_results() {
        let resp = Json::parse(
            r#"{"results":[{"distance":0.25,"data":[1,0],"metadata":{"tag":"red"}},
                          {"distance":1.5,"data":[0,1],"metadata":{}}],"count":2}"#,
        )
        .unwrap();
        let hits = parse_results(&resp).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].distance, 0.25);
        assert_eq!(hits[0].data.as_ref().unwrap(), &vec![1.0, 0.0]);
        assert_eq!(hits[0].metadata.get("tag").unwrap(), "red");
        assert!(hits[1].metadata.is_empty());
    }

    #[test]
    fn empty_results_ok() {
        let resp = Json::parse(r#"{"count":0}"#).unwrap();
        assert!(parse_results(&resp).unwrap().is_empty());
    }
}
