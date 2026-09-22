import { test } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { GigaVector, GigaVectorError, Distance } from "./index.js";

// Spin up a mock HTTP server; no C server required.
function mockServer(handler) {
  return new Promise((resolve) => {
    const srv = http.createServer(handler);
    srv.listen(0, "127.0.0.1", () => resolve(srv));
  });
}
const urlOf = (srv) => `http://127.0.0.1:${srv.address().port}`;

async function readBody(req) {
  let raw = "";
  for await (const chunk of req) raw += chunk;
  return raw ? JSON.parse(raw) : null;
}

test("health", async () => {
  const srv = await mockServer((req, res) => {
    assert.equal(req.url, "/health");
    res.end('{"status":"healthy","vector_count":3}');
  });
  const h = await new GigaVector(urlOf(srv)).health();
  assert.equal(h.status, "healthy");
  assert.equal(h.vector_count, 3);
  srv.close();
});

test("addVector sends data + auth header", async () => {
  const srv = await mockServer(async (req, res) => {
    assert.equal(req.headers["x-api-key"], "secret");
    const body = await readBody(req);
    assert.deepEqual(body.data, [1, 0, 0, 0]);
    assert.equal(body.metadata.tag, "a");
    res.end('{"success":true,"inserted":1,"indices":[7]}');
  });
  const c = new GigaVector(urlOf(srv), { apiKey: "secret" });
  const r = await c.addVector([1, 0, 0, 0], { tag: "a" });
  assert.equal(r.inserted, 1);
  assert.deepEqual(r.indices, [7]);
  srv.close();
});

test("search decodes results", async () => {
  const srv = await mockServer(async (req, res) => {
    const body = await readBody(req);
    assert.equal(body.k, 2);
    assert.equal(body.distance, Distance.Cosine);
    res.end('{"results":[{"id":1,"distance":0.1,"data":[1,0]}],"count":1}');
  });
  const res = await new GigaVector(urlOf(srv)).search([1, 0], 2, Distance.Cosine);
  assert.equal(res.length, 1);
  assert.equal(res[0].id, 1);
  assert.equal(res[0].distance, 0.1);
  srv.close();
});

test("error responses throw GigaVectorError", async () => {
  const srv = await mockServer((req, res) => {
    res.statusCode = 400;
    res.end('{"error":"dimension_mismatch","message":"bad dim"}');
  });
  await assert.rejects(
    () => new GigaVector(urlOf(srv)).addVector([1], null),
    (err) => {
      assert.ok(err instanceof GigaVectorError);
      assert.equal(err.status, 400);
      assert.equal(err.code, "dimension_mismatch");
      return true;
    },
  );
  srv.close();
});
