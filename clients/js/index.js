// JavaScript/TypeScript client for the GigaVector HTTP (REST) API.
// Uses the global `fetch` (Node >= 18, browsers, Deno, Bun).

export const Distance = Object.freeze({
  Euclidean: "euclidean",
  Cosine: "cosine",
  DotProduct: "dot_product",
  Manhattan: "manhattan",
  Hamming: "hamming",
});

/** Error thrown for non-2xx responses. */
export class GigaVectorError extends Error {
  constructor(status, code, message) {
    super(`gigavector: ${status} ${code ?? ""}: ${message ?? ""}`.trim());
    this.name = "GigaVectorError";
    this.status = status;
    this.code = code;
  }
}

export class GigaVector {
  /**
   * @param {string} baseURL e.g. "http://localhost:8080"
   * @param {{apiKey?: string, fetch?: typeof fetch}} [opts]
   */
  constructor(baseURL, opts = {}) {
    this.baseURL = baseURL.replace(/\/$/, "");
    this.apiKey = opts.apiKey;
    this._fetch = opts.fetch ?? globalThis.fetch;
  }

  async _request(method, path, body) {
    const headers = {};
    let payload;
    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
      payload = JSON.stringify(body);
    }
    if (this.apiKey) headers["X-API-Key"] = this.apiKey;

    const resp = await this._fetch(this.baseURL + path, { method, headers, body: payload });
    const text = await resp.text();
    const data = text ? JSON.parse(text) : null;
    if (!resp.ok) {
      throw new GigaVectorError(resp.status, data?.error, data?.message);
    }
    return data;
  }

  /** @returns {Promise<{status: string, vector_count: number}>} */
  health() {
    return this._request("GET", "/health");
  }

  /**
   * Insert a single vector with optional metadata.
   * @param {number[]} data
   * @param {Record<string,string>} [metadata]
   */
  addVector(data, metadata) {
    const body = { data };
    if (metadata && Object.keys(metadata).length) body.metadata = metadata;
    return this._request("POST", "/vectors", body);
  }

  /**
   * k-NN search.
   * @param {number[]} query
   * @param {number} k
   * @param {string} [distance] one of Distance.*
   * @returns {Promise<Array<{id:number,distance:number,data:number[]}>>}
   */
  async search(query, k, distance = Distance.Euclidean) {
    const res = await this._request("POST", "/search", { query, k, distance });
    return res.results ?? [];
  }

  stats() {
    return this._request("GET", "/stats");
  }
}

export default GigaVector;
