//! Minimal HTTP/1.1 request building and response parsing over the standard
//! library, plus a small keep-alive connection pool.

use std::collections::VecDeque;
use std::io::Read;
use std::net::TcpStream;
use std::sync::Mutex;
use std::time::Duration;

use crate::Error;

/// A parsed HTTP response.
pub struct HttpResponse {
    pub status: u16,
    pub body: String,
    /// Whether the server agreed to keep the connection alive.
    pub keep_alive: bool,
}

/// Build a raw HTTP/1.1 request. `body` is already-serialized bytes (or empty).
pub fn build_request(
    method: &str,
    host: &str,
    port: u16,
    path: &str,
    api_key: Option<&str>,
    body: &[u8],
) -> Vec<u8> {
    let mut req = String::new();
    req.push_str(&format!("{} {} HTTP/1.1\r\n", method, path));
    req.push_str(&format!("Host: {}:{}\r\n", host, port));
    req.push_str("Connection: keep-alive\r\n");
    req.push_str("Accept: application/json\r\n");
    if let Some(key) = api_key {
        req.push_str(&format!("X-API-Key: {}\r\n", key));
    }
    if !body.is_empty() {
        req.push_str("Content-Type: application/json\r\n");
    }
    req.push_str(&format!("Content-Length: {}\r\n", body.len()));
    req.push_str("\r\n");
    let mut out = req.into_bytes();
    out.extend_from_slice(body);
    out
}

/// Parse an HTTP response from a readable stream. Supports Content-Length and
/// chunked transfer encoding.
pub fn read_response(stream: &mut impl Read) -> Result<HttpResponse, Error> {
    let mut buf: Vec<u8> = Vec::with_capacity(1024);
    let mut tmp = [0u8; 4096];

    // Read until we have the full header block.
    let header_end = loop {
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            break pos + 4;
        }
        let n = stream.read(&mut tmp).map_err(|e| Error::Io(e.to_string()))?;
        if n == 0 {
            return Err(Error::Protocol("connection closed before headers".into()));
        }
        buf.extend_from_slice(&tmp[..n]);
    };

    let header_text = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut lines = header_text.split("\r\n");
    let status_line = lines.next().unwrap_or("");
    let status = parse_status(status_line)?;

    let mut content_length: Option<usize> = None;
    let mut chunked = false;
    let mut keep_alive = true; // HTTP/1.1 default
    for line in lines {
        if line.is_empty() {
            continue;
        }
        if let Some((name, value)) = line.split_once(':') {
            let name = name.trim().to_ascii_lowercase();
            let value = value.trim();
            match name.as_str() {
                "content-length" => content_length = value.parse().ok(),
                "transfer-encoding" => {
                    if value.eq_ignore_ascii_case("chunked") {
                        chunked = true;
                    }
                }
                "connection" if value.eq_ignore_ascii_case("close") => {
                    keep_alive = false;
                }
                _ => {}
            }
        }
    }

    let mut body_bytes: Vec<u8> = buf[header_end..].to_vec();

    if chunked {
        read_chunked(stream, &mut body_bytes)?;
        let decoded = decode_chunked(&body_bytes)?;
        return Ok(HttpResponse {
            status,
            body: String::from_utf8_lossy(&decoded).to_string(),
            keep_alive,
        });
    }

    if let Some(want) = content_length {
        while body_bytes.len() < want {
            let n = stream.read(&mut tmp).map_err(|e| Error::Io(e.to_string()))?;
            if n == 0 {
                break;
            }
            body_bytes.extend_from_slice(&tmp[..n]);
        }
        body_bytes.truncate(want);
    }

    Ok(HttpResponse {
        status,
        body: String::from_utf8_lossy(&body_bytes).to_string(),
        keep_alive,
    })
}

fn parse_status(line: &str) -> Result<u16, Error> {
    // "HTTP/1.1 200 OK"
    let mut parts = line.split_whitespace();
    let _version = parts.next();
    let code = parts
        .next()
        .ok_or_else(|| Error::Protocol("missing status code".into()))?;
    code.parse::<u16>()
        .map_err(|_| Error::Protocol(format!("bad status line: {}", line)))
}

fn read_chunked(stream: &mut impl Read, body: &mut Vec<u8>) -> Result<(), Error> {
    // Read until we see the terminating "0\r\n\r\n".
    let mut tmp = [0u8; 4096];
    while find_subslice(body, b"0\r\n\r\n").is_none() {
        let n = stream.read(&mut tmp).map_err(|e| Error::Io(e.to_string()))?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
    }
    Ok(())
}

fn decode_chunked(data: &[u8]) -> Result<Vec<u8>, Error> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let line_end = match find_subslice(&data[i..], b"\r\n") {
            Some(p) => i + p,
            None => break,
        };
        let size_str = String::from_utf8_lossy(&data[i..line_end]);
        let size = usize::from_str_radix(size_str.trim(), 16)
            .map_err(|_| Error::Protocol("bad chunk size".into()))?;
        i = line_end + 2;
        if size == 0 {
            break;
        }
        if i + size > data.len() {
            return Err(Error::Protocol("truncated chunk".into()));
        }
        out.extend_from_slice(&data[i..i + size]);
        i += size + 2; // skip chunk + trailing CRLF
    }
    Ok(out)
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|w| w == needle)
}

/// A bounded pool of idle keep-alive TCP connections to one endpoint.
pub struct ConnectionPool {
    host: String,
    port: u16,
    idle: Mutex<VecDeque<TcpStream>>,
    max_idle: usize,
    timeout: Option<Duration>,
}

impl ConnectionPool {
    pub fn new(host: String, port: u16, max_idle: usize, timeout: Option<Duration>) -> Self {
        ConnectionPool {
            host,
            port,
            idle: Mutex::new(VecDeque::new()),
            max_idle,
            timeout,
        }
    }

    /// Take an idle connection or open a fresh one.
    pub fn checkout(&self) -> Result<TcpStream, Error> {
        if let Some(stream) = self.idle.lock().unwrap().pop_front() {
            return Ok(stream);
        }
        let stream = TcpStream::connect((self.host.as_str(), self.port))
            .map_err(|e| Error::Io(e.to_string()))?;
        stream.set_read_timeout(self.timeout).ok();
        stream.set_write_timeout(self.timeout).ok();
        Ok(stream)
    }

    /// Return a reusable connection to the pool (dropped if the pool is full).
    pub fn checkin(&self, stream: TcpStream) {
        let mut idle = self.idle.lock().unwrap();
        if idle.len() < self.max_idle {
            idle.push_back(stream);
        }
        // else: dropped here, closing the socket.
    }

    /// Number of currently pooled idle connections (for diagnostics/tests).
    pub fn idle_count(&self) -> usize {
        self.idle.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_has_headers_and_body() {
        let req = build_request("POST", "localhost", 8080, "/vectors", Some("k1"), b"{\"a\":1}");
        let text = String::from_utf8(req).unwrap();
        assert!(text.starts_with("POST /vectors HTTP/1.1\r\n"));
        assert!(text.contains("Host: localhost:8080\r\n"));
        assert!(text.contains("Connection: keep-alive\r\n"));
        assert!(text.contains("X-API-Key: k1\r\n"));
        assert!(text.contains("Content-Type: application/json\r\n"));
        assert!(text.contains("Content-Length: 7\r\n"));
        assert!(text.ends_with("\r\n\r\n{\"a\":1}"));
    }

    #[test]
    fn get_request_omits_content_type() {
        let req = build_request("GET", "h", 1, "/health", None, b"");
        let text = String::from_utf8(req).unwrap();
        assert!(!text.contains("Content-Type"));
        assert!(!text.contains("X-API-Key"));
        assert!(text.contains("Content-Length: 0\r\n"));
    }

    #[test]
    fn parses_content_length_response() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 13\r\n\r\n{\"status\":1}X";
        let mut cur = std::io::Cursor::new(&raw[..]);
        let resp = read_response(&mut cur).unwrap();
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body, "{\"status\":1}X".get(..13).unwrap());
        assert!(resp.keep_alive);
    }

    #[test]
    fn parses_chunked_response() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\n{\"a\"\r\n3\r\n:1}\r\n0\r\n\r\n";
        let mut cur = std::io::Cursor::new(&raw[..]);
        let resp = read_response(&mut cur).unwrap();
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body, "{\"a\":1}");
    }

    #[test]
    fn detects_connection_close() {
        let raw = b"HTTP/1.1 500 Err\r\nConnection: close\r\nContent-Length: 2\r\n\r\n{}";
        let mut cur = std::io::Cursor::new(&raw[..]);
        let resp = read_response(&mut cur).unwrap();
        assert_eq!(resp.status, 500);
        assert!(!resp.keep_alive);
    }

    #[test]
    fn pool_reuses_and_bounds() {
        // Use loopback listener to mint real TcpStreams.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let pool = ConnectionPool::new(addr.ip().to_string(), addr.port(), 1, None);

        let c1 = pool.checkout().unwrap();
        assert_eq!(pool.idle_count(), 0);
        pool.checkin(c1);
        assert_eq!(pool.idle_count(), 1);

        // Second checkin exceeds max_idle(1) and is dropped.
        let c2 = pool.checkout().unwrap(); // reuses the pooled one
        assert_eq!(pool.idle_count(), 0);
        let c3 = pool.checkout().unwrap(); // fresh
        pool.checkin(c2);
        pool.checkin(c3);
        assert_eq!(pool.idle_count(), 1);
    }
}
