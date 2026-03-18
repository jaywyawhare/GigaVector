# Security

GigaVector handles API keys for LLM services, vector embeddings, and metadata that may contain sensitive information.

---

## TLS Configuration

GigaVector provides TLS via `GV_TLSContext` (see `gv_tls.h`):

```c
GV_TLSConfig tls_cfg;
gv_tls_config_init(&tls_cfg);
tls_cfg.cert_file = "/path/to/server.pem";
tls_cfg.key_file  = "/path/to/server-key.pem";
tls_cfg.min_version = GV_TLS_1_2;       // GV_TLS_1_3 also available
tls_cfg.verify_client = 1;              // enable mutual TLS
tls_cfg.ca_file = "/path/to/ca.pem";   // CA bundle for client verification

GV_TLSContext *tls = gv_tls_create(&tls_cfg);
```

- **Minimum version**: TLS 1.2 (`GV_TLS_1_2`) by default; TLS 1.3 (`GV_TLS_1_3`) recommended.
- **Mutual TLS**: Set `verify_client = 1` and provide `ca_file`.
- **Cipher suites**: Pass a custom list via `cipher_list`, or leave `NULL` for sane defaults.
- **Certificate monitoring**: `gv_tls_cert_days_remaining()` returns days until the server certificate expires.

All outbound LLM API calls (OpenAI, Anthropic, Google) use HTTPS exclusively.

---

## API Key Authentication

The server accepts an optional `api_key` in `GV_ServerConfig`. When set, every request must include a matching `Authorization` header or it receives a 401.

```c
GV_ServerConfig srv_cfg;
gv_server_config_init(&srv_cfg);
srv_cfg.api_key = "your-secret-key";
```

### Auth Manager (`gv_auth.h`)

For more control, use the `GV_AuthManager` which supports both API key and JWT authentication:

| Auth type | Enum | Notes |
|---|---|---|
| None | `GV_AUTH_NONE` | No authentication required |
| API key | `GV_AUTH_API_KEY` | Keys are SHA-256 hashed at rest, support expiration |
| JWT | `GV_AUTH_JWT` | HMAC-SHA256 (HS256), configurable issuer/audience/clock skew |

```c
GV_AuthConfig auth_cfg;
gv_auth_config_init(&auth_cfg);
auth_cfg.type = GV_AUTH_API_KEY;

GV_AuthManager *auth = gv_auth_create(&auth_cfg);

// Generate a key (writes hex key to key_out, key ID to id_out)
char key_out[65], id_out[33];
gv_auth_generate_api_key(auth, "my service", 0, key_out, id_out);

// Verify an incoming key
GV_Identity identity;
GV_AuthResult result = gv_auth_verify_api_key(auth, incoming_key, &identity);
if (result != GV_AUTH_SUCCESS) { /* reject */ }
gv_auth_free_identity(&identity);
```

`gv_auth_authenticate()` auto-detects whether a credential is an API key or JWT and validates accordingly.

---

## Memory Protection

`gv_llm_destroy()` clears API keys from memory with `secure_memclear()` before freeing, preventing keys from lingering in process memory or swap.

---

## Security Checklist

### Pre-Deployment

- [ ] TLS configured with valid cert/key (`GV_TLSConfig`)
- [ ] `min_version` set to `GV_TLS_1_3` (or at least `GV_TLS_1_2`)
- [ ] Server `api_key` set in `GV_ServerConfig`, or `GV_AuthManager` configured
- [ ] `bind_address` restricted (not `0.0.0.0` in production without a reverse proxy)
- [ ] `max_connections` and `max_request_body_bytes` tuned for your deployment
- [ ] `enable_cors` disabled or `cors_origins` scoped to your domains

### Ongoing

- [ ] Monitor `gv_tls_cert_days_remaining()` and rotate certificates before expiry
- [ ] Rotate API keys periodically; use `gv_auth_revoke_api_key()` for old keys
- [ ] Review server access logs (`enable_logging = 1`)

---

## Security Contacts

- **GitHub**: Open a private security advisory
- **Response Time**: Within 24 hours for critical issues
