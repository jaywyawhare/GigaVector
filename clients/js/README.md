# GigaVector JavaScript/TypeScript client

Zero-dependency client for the GigaVector HTTP API (uses global `fetch`; Node ≥ 18, browsers, Deno, Bun). TypeScript types included (`index.d.ts`).

```js
import { GigaVector, Distance } from "gigavector";

const c = new GigaVector("http://localhost:8080", { apiKey: "secret" });
await c.addVector([1, 0, 0, 0], { tag: "a" });
const hits = await c.search([1, 0, 0, 0], 5, Distance.Cosine);
```

Auth is sent as the `X-API-Key` header. Run tests with `npm test` (`node --test`, mock server, no C server needed).
