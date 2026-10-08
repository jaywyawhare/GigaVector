const API = window.location.origin;

// Design tokens from style.css, so canvas drawing matches the CSS theme.
const cssToken = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const T = {
  surface: cssToken("--chart-surface"),
  grid: cssToken("--chart-grid"),
  axis: cssToken("--chart-axis"),
  ink: cssToken("--text"),
  ink2: cssToken("--text-2"),
  ink3: cssToken("--text-3"),
  accent: cssToken("--accent"),
  s1: cssToken("--series-1"),
  s2: cssToken("--series-2"),
  s3: cssToken("--series-3"),
  other: cssToken("--series-other"),
  good: cssToken("--good"),
  warning: cssToken("--warning"),
  critical: cssToken("--critical"),
};
const CHART_FONT = '11px "IBM Plex Sans", sans-serif';

// Size for a canvas: fills its .chart-body when it has one, otherwise the
// parent's width at a fixed fallback height.
function canvasBox(canvas, fallbackH, padX = 0) {
  const p = canvas.parentElement;
  if (p.classList.contains("chart-body")) {
    return [Math.max(120, p.clientWidth), Math.max(120, p.clientHeight)];
  }
  return [p.clientWidth - padX, fallbackH];
}
const CHART_FONT_MONO = '11px "IBM Plex Mono", monospace';


// utils

function jsonHighlight(obj) {
  if (obj == null) return '<span class="json-null">null</span>';
  const s = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
  return s.replace(
    /("(\\u[a-fA-F0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)/g,
    (m) => {
      let cls = "json-num";
      if (/^"/.test(m)) cls = /:$/.test(m) ? "json-key" : "json-str";
      else if (/true|false/.test(m)) cls = "json-bool";
      else if (/null/.test(m)) cls = "json-null";
      return '<span class="' + cls + '">' + m + "</span>";
    },
  );
}

// api wrapper

let activeCollection = "";

async function apiCall(path, opts) {
  try {
    if (!opts) opts = {};
    if (!opts.headers) opts.headers = {};
    if (activeCollection) opts.headers["X-Collection"] = activeCollection;
    const r = await fetch(API + path, opts);
    const text = await r.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
    return { status: r.status, data, ok: r.ok };
  } catch (e) {
    return { status: 0, data: { error: e.message }, ok: false };
  }
}

function formatBytes(b) {
  if (b == null) return "-";
  if (b < 1024) return b + " B";
  if (b < 1048576) return (b / 1024).toFixed(1) + " KB";
  if (b < 1073741824) return (b / 1048576).toFixed(1) + " MB";
  return (b / 1073741824).toFixed(2) + " GB";
}

function formatUptime(s) {
  if (s == null || s < 0) return "";
  if (s < 60) return s + "s";
  if (s < 3600) return Math.floor(s / 60) + "m " + (s % 60) + "s";
  return Math.floor(s / 3600) + "h " + Math.floor((s % 3600) / 60) + "m";
}

function showToast(msg, type) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.className = "toast show " + (type || "");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => (t.className = "toast"), 3000);
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function escapeJsString(s) {
  return String(s).replace(/\\/g, "\\\\").replace(/'/g, "\\'");
}

async function copyText(text) {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(ta);
      if (!ok) throw new Error("copy failed");
    }
    showToast("Copied to clipboard", "success");
  } catch (_) {
    showToast("Could not copy: the clipboard is not available in this browser", "error");
  }
}

document.querySelectorAll(".tabs").forEach((tabs) => {
  tabs.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.tab;
      tabs
        .querySelectorAll(".tab")
        .forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      tabs.parentElement
        .querySelectorAll(".tab-content")
        .forEach((c) => (c.style.display = "none"));
      document.getElementById(target).style.display = "block";
    });
  });
});

let refreshTimer = null;

// ---- Server stats ----------------------------------------------------------
// /api/detailed-stats reports search latency as a histogram (per-bucket counts
// with upper boundaries in microseconds), memory as a breakdown and health as
// an int. Older builds sent plain numbers, so both shapes are accepted.
const HEALTH_NAMES = { 0: "healthy", "-1": "degraded", "-2": "unhealthy" };
function healthName(v) {
  if (typeof v === "number") return HEALTH_NAMES[v] || "unknown";
  return v || "unknown";
}
function parseDetailed(d) {
  d = d || {};
  const lat = d.search_latency;
  return {
    qps: Number(d.queries_per_second) || 0,
    ips: Number(d.inserts_per_second) || 0,
    latNumber: typeof lat === "number" ? lat : null,
    hist: lat && typeof lat === "object" && Array.isArray(lat.buckets) ? lat : null,
    mem:
      d.memory && typeof d.memory === "object"
        ? d.memory
        : { total_bytes: Number(d.memory) || 0 },
    health: healthName(d.health_status),
    deleted: d.deleted_vector_count ?? null,
    deletedRatio: d.deleted_ratio ?? null,
    basic: d.basic_stats || {},
  };
}
function histPercentile(hist, p) {
  if (!hist || !hist.total_samples) return null;
  const target = hist.total_samples * p;
  let acc = 0;
  for (const b of hist.buckets) {
    acc += b.count;
    if (acc >= target) return b.boundary_us / 1000;
  }
  return null;
}
function fmtMs(ms) {
  if (ms == null || !Number.isFinite(ms)) return "-";
  return ms >= 100 ? `${ms.toFixed(0)} ms` : ms >= 10 ? `${ms.toFixed(1)} ms` : `${ms.toFixed(2)} ms`;
}
function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

// ---- Overview state --------------------------------------------------------
const OV_INTERVAL_S = 2.5;
const OV_MAX_SAMPLES = 360; // 15 minutes
const ov = {
  qps: [],
  ips: [],
  lat: [],
  lastHist: null,
  metric: "qps",
  range: 120,
  collectionsLoaded: 0,
};
const OV_METRICS = {
  qps: { title: "Queries per second", unit: "queries/s", fmt: (v) => v.toFixed(1) },
  ips: { title: "Inserts per second", unit: "inserts/s", fmt: (v) => v.toFixed(1) },
  lat: { title: "Search latency", unit: "", fmt: (v) => fmtMs(v) },
};
function pushSample(arr, v) {
  arr.push(v == null || !Number.isFinite(v) ? null : v);
  if (arr.length > OV_MAX_SAMPLES) arr.shift();
}

// Mean latency of the searches that ran during the last poll interval, from
// the change in the cumulative histogram. Null when no searches ran.
function intervalLatency(D) {
  if (D.latNumber != null) return D.latNumber;
  const h = D.hist;
  if (!h) return null;
  const prev = ov.lastHist;
  ov.lastHist = { n: h.total_samples, sum: h.sum_latency_us };
  if (!prev) return h.total_samples ? h.sum_latency_us / h.total_samples / 1000 : null;
  const dn = h.total_samples - prev.n;
  return dn > 0 ? (h.sum_latency_us - prev.sum) / dn / 1000 : null;
}

function setStatus(name) {
  const dot = document.getElementById("statusDot");
  dot.className = "status-dot " + name;
  const label = name.charAt(0).toUpperCase() + name.slice(1);
  setText("statusText", label);
  setText("side-status", label);
  document.getElementById("side-status-dot").className = "status-dot " + name;
}

async function refreshOverview() {
  const [info, stats, health, det] = await Promise.all([
    apiCall("/api/dashboard/info"),
    apiCall("/stats"),
    apiCall("/health"),
    apiCall("/api/detailed-stats"),
  ]);
  const I = info.ok && info.data ? info.data : {};
  const S = stats.ok && stats.data ? stats.data : {};
  const H = health.ok && health.data ? health.data : null;
  const D = det.ok && det.data ? parseDetailed(det.data) : null;

  if (!H) {
    setStatus("unhealthy");
    setText("statusText", "Unreachable");
    setText("side-status", "Unreachable");
    setText("side-uptime", "The server is not responding");
    return;
  }
  const status = D && D.health !== "unknown" ? D.health : healthName(H.status);
  setStatus(status);
  const uptime = H.uptime_seconds ?? S.uptime_seconds;
  setText("side-uptime", uptime != null ? `Up for ${formatUptime(uptime)}` : "");
  setText("side-version", I.version ? `Version ${I.version}` : "");
  setText("ov-subtitle", `${I.index_type || "Unknown"} index, ${I.dimension ?? "?"} dimensions`);

  // KPIs
  const count = I.vector_count ?? H.vector_count ?? S.total_vectors ?? 0;
  setText("kpi-vectors", count.toLocaleString());
  if (D && D.deleted) {
    const pct = D.deletedRatio != null ? ` (${(D.deletedRatio * 100).toFixed(1)}%)` : "";
    setText("kpi-vectors-sub", `${D.deleted.toLocaleString()} deleted${pct}, reclaimed by compaction`);
  } else {
    setText("kpi-vectors-sub", `${I.dimension ?? "?"} dimensions, ${I.index_type || "unknown"} index`);
  }

  const lat = D ? intervalLatency(D) : null;
  pushSample(ov.qps, D ? D.qps : null);
  pushSample(ov.ips, D ? D.ips : null);
  pushSample(ov.lat, lat);

  setText("kpi-qps", D ? D.qps.toFixed(1) : "-");
  setText(
    "kpi-qps-sub",
    S.total_queries != null ? `${S.total_queries.toLocaleString()} queries since start` : "",
  );
  const lastLat = [...ov.lat].reverse().find((v) => v != null);
  setText("kpi-lat", lastLat != null ? fmtMs(lastLat) : "-");
  // Percentiles come from histogram buckets, so report the bucket bound.
  const p95 = D ? histPercentile(D.hist, 0.95) : null;
  setText(
    "kpi-lat-sub",
    p95 != null ? `95% of searches under ${fmtMs(p95)}` : "No searches yet",
  );
  const mem = D ? D.mem : null;
  setText("kpi-mem", mem ? formatBytes(mem.total_bytes) : "-");
  setText(
    "kpi-mem-sub",
    mem && mem.index_bytes != null ? `${formatBytes(mem.index_bytes)} in the index` : "",
  );

  drawSparkline(document.getElementById("spark-qps"), ov.qps);
  drawSparkline(document.getElementById("spark-lat"), ov.lat);
  drawTraffic();
  if (D) {
    renderMemoryBars(mem);
    drawLatencyHistogram(D.hist);
  }

  renderRows(document.getElementById("ov-instance"), [
    ["Status", statusLabel(status)],
    ["Version", I.version || "-"],
    ["Index type", I.index_type || "-"],
    ["Dimension", I.dimension ?? "-"],
    ["Uptime", uptime != null ? formatUptime(uptime) : "-"],
    ["WAL records", D?.basic.total_wal_records != null ? D.basic.total_wal_records.toLocaleString() : "-"],
    ...(H.read_only != null ? [["Read only", H.read_only ? "Yes" : "No"]] : []),
  ]);

  const reqs = S.total_requests ?? 0,
    errs = S.error_count ?? 0;
  renderRows(document.getElementById("ov-requests"), [
    ["Total requests", reqs.toLocaleString()],
    ["Inserts", (S.total_inserts ?? 0).toLocaleString()],
    ["k-NN queries", (S.total_queries ?? 0).toLocaleString()],
    ["Range queries", (S.total_range_queries ?? 0).toLocaleString()],
    ["Errors", `${errs.toLocaleString()} (${reqs ? ((errs / reqs) * 100).toFixed(2) : "0.00"}%)`],
    ["Received", formatBytes(S.total_bytes_received)],
    ["Sent", formatBytes(S.total_bytes_sent)],
  ]);

  if (Date.now() - ov.collectionsLoaded > 30000) loadOverviewCollections();
}

function statusLabel(name) {
  const span = document.createElement("span");
  const dot = document.createElement("span");
  dot.className = `status-dot ${name}`;
  span.append(dot, name.charAt(0).toUpperCase() + name.slice(1));
  return span;
}

function renderRows(el, rows) {
  if (!el) return;
  const dl = document.createElement("dl");
  dl.className = "kv";
  for (const [label, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    if (value instanceof Node) dd.append(value);
    else dd.textContent = String(value);
    dl.append(dt, dd);
  }
  el.replaceChildren(dl);
}

function renderMemoryBars(mem) {
  const el = document.getElementById("ov-memory");
  if (!el) return;
  const total = mem.total_bytes || 0;
  setText("ov-memory-total", formatBytes(total));
  const parts = [
    ["Vector storage", mem.soa_storage_bytes],
    ["Index", mem.index_bytes],
    ["Metadata index", mem.metadata_index_bytes],
    ["Write-ahead log", mem.wal_bytes],
  ].filter(([, v]) => v != null);
  if (!parts.length) {
    el.innerHTML = '<p class="kv-empty">This server does not report a memory breakdown.</p>';
    return;
  }
  el.replaceChildren(
    ...parts.map(([label, v]) => {
      const row = document.createElement("div");
      row.className = "bar-row";
      const head = document.createElement("div");
      head.className = "bar-head";
      const l = document.createElement("span");
      l.textContent = label;
      const val = document.createElement("span");
      val.className = "bar-value";
      val.textContent = `${formatBytes(v)}${total ? `, ${((v / total) * 100).toFixed(0)}%` : ""}`;
      head.append(l, val);
      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill";
      fill.style.width = `${total ? Math.max(0.5, (v / total) * 100) : 0}%`;
      track.append(fill);
      row.append(head, track);
      return row;
    }),
  );
}

async function loadOverviewCollections() {
  ov.collectionsLoaded = Date.now();
  const card = document.getElementById("ov-collections-card");
  const r = await apiCall("/api/collections");
  const list = r.ok && r.data && Array.isArray(r.data.collections) ? r.data.collections : null;
  if (!list) {
    card.hidden = true;
    return;
  }
  card.hidden = false;
  setText("ov-collections-count", `${list.length} total`);
  const tbody = document.getElementById("ov-collections-body");
  if (!list.length) {
    tbody.innerHTML =
      '<tr><td colspan="5" class="table-empty">No collections yet. Create one through the API or the Python client.</td></tr>';
    return;
  }
  tbody.replaceChildren(
    ...list.map((c) => {
      const tr = document.createElement("tr");
      const cells = [
        c.name,
        c.index_type ?? "-",
        c.dimension ?? "-",
        (c.vector_count ?? 0).toLocaleString(),
        c.memory_bytes != null ? formatBytes(c.memory_bytes) : "-",
      ];
      cells.forEach((v, i) => {
        const td = document.createElement("td");
        td.textContent = v;
        if (i >= 2) td.className = "num";
        tr.append(td);
      });
      return tr;
    }),
  );
}

function drawTraffic() {
  const m = OV_METRICS[ov.metric];
  const data = ov[ov.metric].slice(-ov.range);
  setText("traffic-title", m.title);
  const vals = data.filter((v) => v != null);
  const meta = document.getElementById("traffic-meta");
  if (vals.length) {
    const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
    const peak = Math.max(...vals);
    meta.textContent = `Average ${m.fmt(avg)}, peak ${m.fmt(peak)}`;
  } else {
    meta.textContent = ov.metric === "lat" ? "No searches in this window" : "Collecting samples";
  }
  drawLineChart(document.getElementById("traffic-chart"), data, {
    fill: true,
    spanSec: ov.range * OV_INTERVAL_S,
    intervalSec: OV_INTERVAL_S,
    fmt: m.fmt,
    unit: m.unit,
  });
}

function drawLatencyHistogram(hist) {
  const canvas = document.getElementById("latency-hist");
  if (!canvas) return;
  if (!hist || !hist.total_samples) {
    setText("latency-hist-meta", "No searches yet");
    drawBars(canvas, [], []);
    return;
  }
  let last = 0;
  hist.buckets.forEach((b, i) => {
    if (b.count) last = i;
  });
  const buckets = hist.buckets.slice(0, Math.min(hist.buckets.length, last + 2));
  const labels = buckets.map((b) => fmtMs(b.boundary_us / 1000).replace(" ms", ""));
  setText(
    "latency-hist-meta",
    `${hist.total_samples.toLocaleString()} searches, mean ${fmtMs(hist.sum_latency_us / hist.total_samples / 1000)}`,
  );
  drawBars(canvas, labels, buckets.map((b) => b.count), {
    tip: (i) => {
      const lo = i ? fmtMs(buckets[i - 1].boundary_us / 1000) : "0 ms";
      const pct = ((buckets[i].count / hist.total_samples) * 100).toFixed(1);
      return [`${lo} to ${fmtMs(buckets[i].boundary_us / 1000)}`, `${buckets[i].count.toLocaleString()} searches, ${pct}%`];
    },
    xTitle: "Upper bound (ms)",
  });
}

document.querySelectorAll("[data-ov-metric]").forEach((b) =>
  b.addEventListener("click", () => {
    ov.metric = b.dataset.ovMetric;
    document
      .querySelectorAll("[data-ov-metric]")
      .forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    drawTraffic();
  }),
);
document.querySelectorAll("[data-ov-range]").forEach((b) =>
  b.addEventListener("click", () => {
    ov.range = Number(b.dataset.ovRange);
    document
      .querySelectorAll("[data-ov-range]")
      .forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    drawTraffic();
  }),
);

// Redraw a canvas whenever its .chart-body changes size (view shown, window
// resized, table view opened).
const chartRedraw = {
  "traffic-chart": drawTraffic,
  "latency-hist": () => {},
  "scatter-canvas": () => vizData && drawScatterWithHighlights(),
};
if (window.ResizeObserver) {
  const pending = new Set();
  const ro = new ResizeObserver((entries) => {
    for (const e of entries) {
      const c = e.target.querySelector("canvas");
      const fn = chartRedraw[c?.id] || c?._redraw;
      if (fn) pending.add(fn);
    }
    requestAnimationFrame(() => {
      for (const fn of pending) fn();
      pending.clear();
    });
  });
  document.querySelectorAll(".chart-body, .kpi-spark").forEach((el) => ro.observe(el));
}

function startRefresh() {
  refreshOverview();
  refreshTimer = setInterval(refreshOverview, OV_INTERVAL_S * 1000);
}
startRefresh();

let currentPoints = [],
  selectedPointIdx = -1;
async function loadPoints() {
  const offset = parseInt(document.getElementById("points-offset").value) || 0;
  const limit = parseInt(document.getElementById("points-limit").value) || 50;
  const r = await apiCall(
    "/vectors/scroll?offset=" + offset + "&limit=" + limit,
  );
  if (!r.ok) return;
  currentPoints = r.data.vectors || [];
  document.getElementById("points-count").textContent = `${(r.data.total ?? 0).toLocaleString()} total`;
  selectedPointIdx = -1;
  renderPointsList();
}
function pointsPrev() {
  const el = document.getElementById("points-offset");
  const lim = parseInt(document.getElementById("points-limit").value) || 50;
  el.value = Math.max(0, parseInt(el.value) - lim);
  loadPoints();
}
function pointsNext() {
  const el = document.getElementById("points-offset");
  const lim = parseInt(document.getElementById("points-limit").value) || 50;
  el.value = parseInt(el.value) + lim;
  loadPoints();
}
function renderPointsList() {
  const list = document.getElementById("points-list");
  if (!currentPoints.length) {
    list.innerHTML =
      '<div class="empty-state">No vectors in this range. Lower the offset or add vectors.</div>';
    return;
  }
  list.innerHTML = currentPoints
    .map((p, i) => {
      const d = Array.isArray(p.data)
        ? `[${p.data
            .slice(0, 5)
            .map((v) => (typeof v === "number" ? v.toFixed(4) : v))
            .join(", ")}${p.data.length > 5 ? ", ..." : ""}]`
        : JSON.stringify(p.data).slice(0, 60);
      return `<div class="point-item${
        selectedPointIdx === i ? " selected" : ""
      }" onclick="selectPoint(${i})">
        <div class="point-idx">${p.index}</div>
        <div class="point-data">${d}</div>
        <div class="point-badge">${Array.isArray(p.data) ? `${p.data.length} dims` : ""}</div>
      </div>`;
    })
    .join("");
}

function selectPoint(i) {
  selectedPointIdx = i;
  renderPointsList();
  const p = currentPoints[i];
  document.getElementById("point-detail").innerHTML = `
    <div class="viz-point-id" style="margin-bottom:12px">#${p.index}</div>
    <div class="json-view" style="max-height:300px;margin-bottom:12px">${jsonHighlight(p)}</div>
    <button class="btn btn-sm btn-danger" onclick="deletePointFromBrowser(${p.index})">Delete vector</button>`;
}

async function deletePointFromBrowser(id) {
  if (!confirm(`Delete vector #${id}? This cannot be undone.`)) return;
  const r = await apiCall(`/vectors/${id}`, { method: "DELETE" });
  showToast(
    r.ok ? `Deleted vector #${id}` : `Could not delete vector #${id}`,
    r.ok ? "success" : "error",
  );
  if (r.ok) {
    selectedPointIdx = -1;
    loadPoints();
  }
}

async function addVector() {
  const dataStr = document.getElementById("vec-add-data").value.trim();
  const metaStr = document.getElementById("vec-add-meta").value.trim();
  try {
    const body = { data: JSON.parse(dataStr) };
    if (metaStr) body.metadata = JSON.parse(metaStr);
    const r = await apiCall("/vectors", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    showToast(
      r.ok ? "Vector added" : `Could not add vector: ${r.data?.message || JSON.stringify(r.data)}`,
      r.ok ? "success" : "error",
    );
  } catch (e) {
    showToast(`Values must be valid JSON. ${e.message}`, "error");
  }
}

// scatter plot vis

let vizData = null;
async function runVisualization() {
  const limit = parseInt(document.getElementById("viz-limit").value) || 200;
  const algo = document.getElementById("viz-algo").value;
  const status = document.getElementById("viz-status");
  const info = document.getElementById("scatter-info");
  status.textContent = "Loading vectors";
  const r = await apiCall(`/vectors/scroll?offset=0&limit=${limit}`);
  if (!r.ok || !r.data.vectors || !r.data.vectors.length) {
    vizData = null;
    status.textContent = "";
    info.textContent = r.ok
      ? "This collection has no vectors yet. Add some from the Vectors page, then project again."
      : "Could not load vectors from the server. Check that it is running, then project again.";
    drawScatter();
    showScatterDetail(-1);
    return;
  }
  const vecs = r.data.vectors;
  const raw = vecs.map((v) => (Array.isArray(v.data) ? v.data : []));
  const indices = vecs.map((v) => v.index);
  const metas = vecs.map((v) => v.metadata || {});
  const norms = raw.map((v) => Math.hypot(...v));
  const dim = raw[0].length;
  const pts = algo === "pca" ? pcaProject(raw) : randomProject(raw);
  vizData = { pts, indices, raw, metas, norms };
  scatterHovered = -1;
  scatterSelected = -1;
  showScatterDetail(-1);
  populateColorByOptions(metas);
  status.textContent = `${vecs.length} vectors, ${algo === "pca" ? "PCA" : "random projection"}`;
  info.textContent = `${dim} dimensions flattened to 2. Points that sit close here are not always close in full dimension; select one to compare.`;
  recolorScatter();
}

function pcaProject(data) {
  const n = data.length,
    d = data[0].length;
  const mean = new Float64Array(d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += data[i][j];
  for (let j = 0; j < d; j++) mean[j] /= n;
  const c = data.map((r) => r.map((v, j) => v - mean[j]));
  function powerIter(mat) {
    let v = new Float64Array(d);
    for (let j = 0; j < d; j++) v[j] = Math.random() - 0.5;
    for (let it = 0; it < 80; it++) {
      const nv = new Float64Array(d);
      for (let i = 0; i < n; i++) {
        let dot = 0;
        for (let j = 0; j < d; j++) dot += mat[i][j] * v[j];
        for (let j = 0; j < d; j++) nv[j] += dot * mat[i][j];
      }
      let nm = 0;
      for (let j = 0; j < d; j++) nm += nv[j] * nv[j];
      nm = Math.sqrt(nm) || 1;
      for (let j = 0; j < d; j++) v[j] = nv[j] / nm;
    }
    return v;
  }
  const pc1 = powerIter(c);
  const deflated = c.map((r) => {
    let dot = 0;
    for (let j = 0; j < d; j++) dot += r[j] * pc1[j];
    return r.map((v, j) => v - dot * pc1[j]);
  });
  const pc2 = powerIter(deflated);
  return data.map((_, i) => {
    let x = 0,
      y = 0;
    for (let j = 0; j < d; j++) {
      x += c[i][j] * pc1[j];
      y += c[i][j] * pc2[j];
    }
    return [x, y];
  });
}

function randomProject(data) {
  const d = data[0].length;
  const r1 = [],
    r2 = [];
  for (let j = 0; j < d; j++) {
    r1.push(Math.random() - 0.5);
    r2.push(Math.random() - 0.5);
  }
  let n1 = 0,
    n2 = 0;
  for (let j = 0; j < d; j++) {
    n1 += r1[j] * r1[j];
    n2 += r2[j] * r2[j];
  }
  n1 = Math.sqrt(n1);
  n2 = Math.sqrt(n2);
  for (let j = 0; j < d; j++) {
    r1[j] /= n1;
    r2[j] /= n2;
  }
  return data.map((row) => {
    let x = 0,
      y = 0;
    for (let j = 0; j < d; j++) {
      x += row[j] * r1[j];
      y += row[j] * r2[j];
    }
    return [x, y];
  });
}

// Vector fingerprint: one column per component on a diverging scale (blue for
// negative, gray at zero, red for positive), normalized to the vector's own
// largest magnitude. Lets two vectors be compared by shape at a glance.
const FP_NEG = [57, 135, 229],
  FP_MID = [56, 56, 53],
  FP_POS = [230, 103, 103];
function fingerprintCanvas(vec, w, h) {
  const c = document.createElement("canvas");
  const dpr = window.devicePixelRatio || 1;
  c.width = Math.round(w * dpr);
  c.height = Math.round(h * dpr);
  c.style.width = `${w}px`;
  c.style.height = `${h}px`;
  c.className = "fingerprint";
  c.setAttribute("aria-hidden", "true");
  const ctx = c.getContext("2d");
  let max = 0;
  for (const v of vec) max = Math.max(max, Math.abs(v));
  max = max || 1;
  const n = vec.length,
    cw = c.width / n;
  for (let i = 0; i < n; i++) {
    const t = vec[i] / max,
      pole = t < 0 ? FP_NEG : FP_POS,
      a = Math.min(1, Math.abs(t));
    const rgb = FP_MID.map((m, j) => Math.round(m + (pole[j] - m) * a));
    ctx.fillStyle = `rgb(${rgb})`;
    // Columns overlap by a pixel so wide vectors do not show seams.
    ctx.fillRect(Math.floor(i * cw), 0, Math.ceil(cw) + 1, c.height);
  }
  return c;
}

// k nearest neighbors of point idx by cosine similarity in the original space
// (not the 2D projection), over the loaded sample. Cached per point.
function vizNeighbors(idx, k) {
  const cache = vizData.nbrCache || (vizData.nbrCache = new Map());
  if (cache.has(idx)) return cache.get(idx);
  const { raw, norms } = vizData,
    q = raw[idx],
    qn = norms[idx] || 1;
  const scored = [];
  for (let i = 0; i < raw.length; i++) {
    if (i === idx) continue;
    let dot = 0;
    for (let j = 0; j < q.length; j++) dot += q[j] * raw[i][j];
    scored.push([i, dot / (qn * (norms[i] || 1))]);
  }
  scored.sort((a, b) => b[1] - a[1]);
  const out = scored.slice(0, k);
  cache.set(idx, out);
  return out;
}

// Categorical slots validated (all pairs, normal + CVD, >= 3:1 contrast) against
// the --chart-surface token. A scatter overlaps every pair, so only three hues pass;
// further groups fold into a muted "Other" bucket instead of a generated hue.
const VIZ_PALETTE = [T.s1, T.s2, T.s3];
const VIZ_OTHER_COLOR = T.other;
const VIZ_DEFAULT_COLOR = T.s1;
const VIZ_SURFACE = T.surface;
const VIZ_GRID = T.grid;
const VIZ_AXIS = T.axis;
const VIZ_INK_PRIMARY = T.ink;
const VIZ_INK_SECONDARY = T.ink2;

// Populate the "Color by" dropdown: keep the fixed None / K-means options, then
// append any metadata keys found, preserving the current selection.
function populateColorByOptions(metas) {
  const sel = document.getElementById("viz-color");
  if (!sel) return;
  const prev = sel.value;
  const keys = new Set();
  for (const m of metas) for (const k of Object.keys(m || {})) keys.add(k);
  const sorted = [...keys].sort();
  sel.innerHTML =
    '<option value="">None</option>' +
    '<option value="__cluster__">K-means clusters</option>' +
    sorted
      .map((k) => `<option value="${escapeHtml(k)}">${escapeHtml(k)}</option>`)
      .join("");
  sel.value = prev === "__cluster__" || sorted.includes(prev) ? prev : "";
}

// Deterministic k-means (fixed seed) over the raw vectors. Returns an int label
// per row. Small k and point counts keep this well within interactive budget.
function kmeansLabels(data, k) {
  const n = data.length;
  if (!n) return [];
  const d = data[0].length;
  k = Math.max(2, Math.min(k, n));
  let seed = 1234567;
  const rand = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  const centroids = [];
  const used = new Set();
  while (centroids.length < k) {
    const idx = Math.floor(rand() * n);
    if (used.has(idx)) continue;
    used.add(idx);
    centroids.push(data[idx].slice());
  }
  const labels = new Int32Array(n);
  for (let it = 0; it < 12; it++) {
    for (let i = 0; i < n; i++) {
      let best = 0,
        bestDist = Infinity;
      for (let c = 0; c < k; c++) {
        let dist = 0;
        for (let j = 0; j < d; j++) {
          const diff = data[i][j] - centroids[c][j];
          dist += diff * diff;
        }
        if (dist < bestDist) {
          bestDist = dist;
          best = c;
        }
      }
      labels[i] = best;
    }
    const sums = Array.from({ length: k }, () => new Float64Array(d));
    const counts = new Int32Array(k);
    for (let i = 0; i < n; i++) {
      counts[labels[i]]++;
      const s = sums[labels[i]];
      for (let j = 0; j < d; j++) s[j] += data[i][j];
    }
    for (let c = 0; c < k; c++) {
      if (!counts[c]) continue;
      for (let j = 0; j < d; j++) centroids[c][j] = sums[c][j] / counts[c];
    }
  }
  return labels;
}

// Group each point by k-means cluster or a metadata field. The three largest
// groups get a palette slot (largest first, so a group keeps its color as long
// as it stays in the top three); the rest share "Other". Returns
// { groups: string[], colors: string[], legend: {label,color,count}[] }.
function computeVizGroups() {
  const field = (document.getElementById("viz-color") || {}).value || "";
  const raw = vizData.raw,
    metas = vizData.metas;
  if (!field) {
    return { groups: raw.map(() => ""), colors: raw.map(() => VIZ_DEFAULT_COLOR), legend: [] };
  }
  let groups;
  if (field === "__cluster__") {
    const k = parseInt((document.getElementById("viz-k") || {}).value) || 3;
    groups = Array.from(kmeansLabels(raw, k), (l) => `Cluster ${l + 1}`);
  } else {
    groups = metas.map((m) => (m && m[field] != null ? String(m[field]) : "(none)"));
  }
  const counts = new Map();
  for (const g of groups) counts.set(g, (counts.get(g) || 0) + 1);
  const ranked = [...counts.entries()].sort(
    (a, b) => b[1] - a[1] || a[0].localeCompare(b[0]),
  );
  const slot = new Map();
  ranked.slice(0, VIZ_PALETTE.length).forEach(([g], i) => slot.set(g, VIZ_PALETTE[i]));
  const legend = ranked
    .slice(0, VIZ_PALETTE.length)
    .map(([label, count]) => ({ label, color: slot.get(label), count }));
  const rest = ranked.slice(VIZ_PALETTE.length);
  if (rest.length) {
    legend.push({
      label: `Other (${rest.length} group${rest.length > 1 ? "s" : ""})`,
      color: VIZ_OTHER_COLOR,
      count: rest.reduce((s, [, c]) => s + c, 0),
    });
  }
  return {
    groups,
    colors: groups.map((g) => slot.get(g) || VIZ_OTHER_COLOR),
    legend,
  };
}

// Recompute grouping (k-means is not free) and redraw. Hover redraws reuse the
// cached result instead of re-clustering on every mouse move.
function recolorScatter() {
  if (!vizData) return;
  vizData.coloring = computeVizGroups();
  renderVizLegend();
  renderVizTable();
  drawScatterWithHighlights();
}

function renderVizLegend() {
  const el = document.getElementById("viz-legend");
  if (!el) return;
  el.replaceChildren();
  const legend = (vizData && vizData.coloring && vizData.coloring.legend) || [];
  el.hidden = !legend.length;
  for (const { label, color, count } of legend) {
    const item = document.createElement("span");
    item.className = "viz-legend-item";
    const sw = document.createElement("span");
    sw.className = "viz-legend-swatch";
    sw.style.background = color;
    const name = document.createElement("span");
    name.textContent = label;
    const n = document.createElement("span");
    n.className = "viz-legend-count";
    n.textContent = count;
    item.append(sw, name, n);
    el.append(item);
  }
}

function renderVizTable() {
  const tbody = document.getElementById("viz-table-body");
  if (!tbody || !vizData) return;
  const { pts, indices, coloring } = vizData;
  const rows = pts.map((p, i) => {
    const tr = document.createElement("tr");
    tr.dataset.idx = i;
    const cells = [
      `#${indices[i]}`,
      p[0].toFixed(4),
      p[1].toFixed(4),
      (coloring && coloring.groups[i]) || "-",
    ];
    for (const c of cells) {
      const td = document.createElement("td");
      td.textContent = c;
      tr.append(td);
    }
    if (coloring && coloring.groups[i]) {
      const sw = document.createElement("span");
      sw.className = "viz-legend-swatch";
      sw.style.background = coloring.colors[i];
      tr.lastChild.prepend(sw);
    }
    return tr;
  });
  tbody.replaceChildren(...rows);
}

function drawScatter() {
  const canvas = document.getElementById("scatter-canvas");
  const [w, h] = canvasBox(canvas, 460);
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.fillStyle = VIZ_SURFACE;
  ctx.fillRect(0, 0, w, h);
  if (!vizData) {
    ctx.fillStyle = VIZ_INK_SECONDARY;
    ctx.font = '14px "IBM Plex Sans", sans-serif';
    ctx.textAlign = "center";
    ctx.fillText("Each dot will be one stored vector, placed by similarity.", w / 2, h / 2);
    return;
  }
  if (!vizData.coloring) vizData.coloring = computeVizGroups();
  const { pts, indices } = vizData;
  const padL = 36,
    padR = 20,
    padT = 20,
    padB = 32;
  let mnX = Infinity,
    mxX = -Infinity,
    mnY = Infinity,
    mxY = -Infinity;
  for (const [x, y] of pts) {
    if (x < mnX) mnX = x;
    if (x > mxX) mxX = x;
    if (y < mnY) mnY = y;
    if (y > mxY) mxY = y;
  }
  const rx = mxX - mnX || 1,
    ry = mxY - mnY || 1;
  // 4% breathing room so edge points are not clipped by the frame.
  mnX -= rx * 0.04;
  mxX += rx * 0.04;
  mnY -= ry * 0.04;
  mxY += ry * 0.04;
  const pw = w - padL - padR,
    ph = h - padT - padB;
  const sx = pw / (mxX - mnX),
    sy = ph / (mxY - mnY);
  function toS(x, y) {
    return [(x - mnX) * sx + padL, padT + ph - (y - mnY) * sy];
  }

  // Recessive solid hairline grid; 1px lines snapped to the pixel grid.
  ctx.strokeStyle = VIZ_GRID;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let g = 1; g < 6; g++) {
    const gx = Math.round(padL + (pw * g) / 6) + 0.5;
    ctx.moveTo(gx, padT);
    ctx.lineTo(gx, padT + ph);
    const gy = Math.round(padT + (ph * g) / 4) + 0.5;
    ctx.moveTo(padL, gy);
    ctx.lineTo(padL + pw, gy);
  }
  ctx.stroke();
  ctx.strokeStyle = VIZ_AXIS;
  ctx.beginPath();
  ctx.moveTo(padL + 0.5, padT);
  ctx.lineTo(padL + 0.5, padT + ph + 0.5);
  ctx.lineTo(padL + pw, padT + ph + 0.5);
  ctx.stroke();

  const algo = (document.getElementById("viz-algo") || {}).value;
  const [ax, ay] = algo === "pca" ? ["PC1", "PC2"] : ["Projection 1", "Projection 2"];
  ctx.fillStyle = VIZ_INK_SECONDARY;
  ctx.font = CHART_FONT;
  ctx.textAlign = "right";
  ctx.fillText(ax, padL + pw, h - 10);
  ctx.save();
  ctx.translate(14, padT);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "right";
  ctx.fillText(ay, 0, 0);
  ctx.restore();

  // 8px markers with a 2px surface ring so overlapping points stay separable.
  // "Other" draws first so the named groups sit on top.
  const { colors } = vizData.coloring;
  const order = pts.map((_, i) => i);
  order.sort(
    (a, b) => (colors[b] === VIZ_OTHER_COLOR) - (colors[a] === VIZ_OTHER_COLOR),
  );
  ctx.lineWidth = 2;
  ctx.strokeStyle = VIZ_SURFACE;
  for (const i of order) {
    const [px, py] = toS(pts[i][0], pts[i][1]);
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fillStyle = colors[i];
    ctx.fill();
    ctx.stroke();
  }

  canvas._vizMap = { pts, indices, toS, w, h };
}

let scatterHovered = -1,
  scatterSelected = -1;
const vizTooltip = document.getElementById("viz-tooltip");

// Nearest point within 12px: a 24px hit target around an 8px mark.
function scatterFindNearest(mx, my) {
  if (!vizData || !vizData.pts) return -1;
  const { pts, toS } = document.getElementById("scatter-canvas")._vizMap || {};
  if (!toS) return -1;
  let closest = -1,
    cd = 12;
  for (let i = 0; i < pts.length; i++) {
    const [sx, sy] = toS(pts[i][0], pts[i][1]);
    const d = Math.hypot(mx - sx, my - sy);
    if (d < cd) {
      cd = d;
      closest = i;
    }
  }
  return closest;
}

function placeVizTooltip(clientX, clientY) {
  const r = vizTooltip.getBoundingClientRect();
  let tx = clientX + 14,
    ty = clientY - 10;
  if (tx + r.width > window.innerWidth - 8) tx = clientX - r.width - 10;
  if (ty + r.height > window.innerHeight - 8) ty = clientY - r.height - 10;
  if (ty < 8) ty = 8;
  vizTooltip.style.left = `${tx}px`;
  vizTooltip.style.top = `${ty}px`;
}

// Tooltip body built with textContent: metadata values are user data.
function showPointTooltip(idx, clientX, clientY) {
  const pt = vizData.raw[idx],
    id = vizData.indices[idx];
  const coloring = vizData.coloring || {};
  const group = coloring.groups ? coloring.groups[idx] : "";
  const row = (cls, text) => {
    const d = document.createElement("div");
    d.className = cls;
    d.textContent = text;
    return d;
  };
  const parts = [row("tt-id", `#${id}`)];
  if (group) {
    const g = row("tt-group", group);
    const sw = document.createElement("span");
    sw.className = "viz-legend-swatch";
    sw.style.background = coloring.colors[idx];
    g.prepend(sw);
    parts.push(g);
  }
  parts.push(fingerprintCanvas(pt, 160, 12));
  parts.push(row("tt-dim", `${pt.length} dimensions, norm ${vizData.norms[idx].toFixed(2)}`));
  vizTooltip.replaceChildren(...parts);
  vizTooltip.classList.add("visible");
  placeVizTooltip(clientX, clientY);
}
function hideVizTooltip() {
  vizTooltip.classList.remove("visible");
}

const VIZ_NEIGHBORS = 5;

function showScatterDetail(idx) {
  const body = document.getElementById("scatter-detail-body");
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  };
  if (idx < 0 || !vizData) {
    body.replaceChildren(
      el(
        "p",
        "viz-empty",
        vizData
          ? "Select a point to see its values and its nearest neighbors in full dimension."
          : "Project vectors to start exploring.",
      ),
    );
    return;
  }
  const pt = vizData.raw[idx],
    id = vizData.indices[idx];
  const coloring = vizData.coloring || {};
  const group = coloring.groups ? coloring.groups[idx] : "";
  const panelW = Math.max(160, body.clientWidth - 28);

  const head = el("div", "viz-point-head");
  head.append(el("span", "viz-point-id", `#${id}`));
  if (group) {
    const g = el("span", "viz-point-group", group);
    const sw = el("span", "viz-legend-swatch");
    sw.style.background = coloring.colors[idx];
    g.prepend(sw);
    head.append(g);
  }

  const scale = el("div", "fingerprint-scale");
  scale.append(el("span", null, "negative"), el("span", null, "0"), el("span", null, "positive"));

  const facts = el(
    "p",
    "viz-point-facts",
    `${pt.length} dimensions, norm ${vizData.norms[idx].toFixed(3)}. Projected to (${vizData.pts[idx][0].toFixed(2)}, ${vizData.pts[idx][1].toFixed(2)}).`,
  );

  const nbrTitle = el("h4", "viz-subhead", "Nearest neighbors");
  const nbrNote = el("p", "viz-note", "By cosine similarity in full dimension. Lines show where they landed on the plot.");
  const list = el("ol", "viz-neighbors");
  for (const [j, sim] of vizNeighbors(idx, VIZ_NEIGHBORS)) {
    const li = el("li");
    const btn = el("button", "viz-neighbor");
    btn.type = "button";
    btn.title = `Select #${vizData.indices[j]}`;
    btn.append(
      el("span", "viz-neighbor-id", `#${vizData.indices[j]}`),
      fingerprintCanvas(vizData.raw[j], 96, 10),
      el("span", "viz-neighbor-sim", sim.toFixed(3)),
    );
    btn.addEventListener("click", () => selectScatterPoint(j));
    li.append(btn);
    list.append(li);
  }

  const vecStr = `[${pt.map((v) => v.toFixed(6)).join(", ")}]`;
  const raw = el("details", "viz-raw");
  raw.append(el("summary", null, "Raw values"), el("pre", null, vecStr));
  const copy = el("button", "btn btn-sm btn-outline", "Copy vector");
  copy.type = "button";
  copy.addEventListener("click", () => copyText(vecStr));

  body.replaceChildren(head, fingerprintCanvas(pt, panelW, 28), scale, facts, nbrTitle, nbrNote, list, raw, copy);
}

function drawScatterWithHighlights() {
  drawScatter();
  if (!vizData || !vizData.pts) return;
  const canvas = document.getElementById("scatter-canvas");
  const { pts, toS } = canvas._vizMap || {};
  if (!toS) return;
  const dpr = window.devicePixelRatio || 1;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const { colors } = vizData.coloring;

  // Redraw the emphasized point on top, then ring it: hover in secondary ink,
  // selection in primary ink with a wider gap.
  const ring = (i, r, color, width) => {
    const [sx, sy] = toS(pts[i][0], pts[i][1]);
    ctx.beginPath();
    ctx.arc(sx, sy, 5, 0, Math.PI * 2);
    ctx.fillStyle = colors[i];
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = VIZ_SURFACE;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();
  };
  // Hairlines from the selection to its full-dimension neighbors: long lines
  // mean the projection pulled true neighbors apart.
  if (scatterSelected >= 0 && scatterSelected < pts.length) {
    const [ox, oy] = toS(pts[scatterSelected][0], pts[scatterSelected][1]);
    ctx.strokeStyle = VIZ_INK_SECONDARY;
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const [j] of vizNeighbors(scatterSelected, VIZ_NEIGHBORS)) {
      const [nx, ny] = toS(pts[j][0], pts[j][1]);
      ctx.moveTo(ox, oy);
      ctx.lineTo(nx, ny);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
    for (const [j] of vizNeighbors(scatterSelected, VIZ_NEIGHBORS)) {
      ring(j, 7, VIZ_INK_SECONDARY, 1);
    }
  }
  if (scatterHovered >= 0 && scatterHovered < pts.length && scatterHovered !== scatterSelected) {
    ring(scatterHovered, 8, VIZ_INK_SECONDARY, 1.5);
  }
  if (scatterSelected >= 0 && scatterSelected < pts.length) {
    ring(scatterSelected, 9, VIZ_INK_PRIMARY, 2);
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  const tbody = document.getElementById("viz-table-body");
  if (tbody) {
    for (const tr of tbody.children) {
      tr.classList.toggle("selected", +tr.dataset.idx === scatterSelected);
    }
  }
}

function selectScatterPoint(idx) {
  scatterSelected = idx === scatterSelected ? -1 : idx;
  drawScatterWithHighlights();
  showScatterDetail(scatterSelected);
}

const scatterCanvas = document.getElementById("scatter-canvas");
scatterCanvas.addEventListener("mousemove", function (e) {
  if (!this._vizMap || !vizData) return;
  const rect = this.getBoundingClientRect();
  const prev = scatterHovered;
  scatterHovered = scatterFindNearest(e.clientX - rect.left, e.clientY - rect.top);
  if (scatterHovered !== prev) drawScatterWithHighlights();
  this.style.cursor = scatterHovered >= 0 ? "pointer" : "crosshair";
  if (scatterHovered >= 0) showPointTooltip(scatterHovered, e.clientX, e.clientY);
  else hideVizTooltip();
});
scatterCanvas.addEventListener("mouseleave", function () {
  scatterHovered = -1;
  hideVizTooltip();
  drawScatterWithHighlights();
});
scatterCanvas.addEventListener("click", function (e) {
  if (!this._vizMap || !vizData) return;
  const rect = this.getBoundingClientRect();
  const idx = scatterFindNearest(e.clientX - rect.left, e.clientY - rect.top);
  if (idx >= 0 || scatterSelected >= 0) selectScatterPoint(idx);
});
// Keyboard: arrows step through points left-to-right, Enter/Space selects,
// Escape clears. The tooltip follows the focused point.
scatterCanvas.addEventListener("keydown", function (e) {
  if (!this._vizMap || !vizData || !vizData.pts.length) return;
  const { pts, toS } = this._vizMap;
  const byX = vizData.byX || (vizData.byX = pts.map((_, i) => i).sort((a, b) => pts[a][0] - pts[b][0]));
  const pos = byX.indexOf(scatterHovered);
  let next = scatterHovered;
  if (e.key === "ArrowRight" || e.key === "ArrowDown") next = byX[Math.min(byX.length - 1, pos + 1)];
  else if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = byX[Math.max(0, pos < 0 ? 0 : pos - 1)];
  else if ((e.key === "Enter" || e.key === " ") && scatterHovered >= 0) {
    e.preventDefault();
    selectScatterPoint(scatterHovered);
    return;
  } else if (e.key === "Escape") {
    scatterHovered = -1;
    hideVizTooltip();
    if (scatterSelected >= 0) selectScatterPoint(scatterSelected);
    else drawScatterWithHighlights();
    return;
  } else return;
  e.preventDefault();
  scatterHovered = next;
  drawScatterWithHighlights();
  const rect = this.getBoundingClientRect();
  const [sx, sy] = toS(pts[next][0], pts[next][1]);
  showPointTooltip(next, rect.left + sx, rect.top + sy);
});
scatterCanvas.addEventListener("blur", hideVizTooltip);
document.getElementById("viz-table-body")?.addEventListener("click", (e) => {
  const tr = e.target.closest("tr");
  if (tr) selectScatterPoint(+tr.dataset.idx);
});

// similarlty graph

let graphNodes = [],
  graphEdges = [],
  graphAnim = null;
async function runGraph() {
  const seed = parseInt(document.getElementById("graph-seed").value) || 0;
  const k = parseInt(document.getElementById("graph-k").value) || 5;
  const depth = parseInt(document.getElementById("graph-depth").value) || 2;
  const status = document.getElementById("graph-status");
  status.textContent = "Building graph";
  graphNodes = [];
  graphEdges = [];
  const visited = new Set();
  let frontier = [seed];
  for (let d = 0; d < depth && frontier.length; d++) {
    const next = [];
    for (const nid of frontier) {
      if (visited.has(nid)) continue;
      visited.add(nid);
      const vr = await apiCall(`/vectors/${nid}`);
      if (!vr.ok || !vr.data.data) continue;
      if (!graphNodes.find((n) => n.id === nid))
        graphNodes.push({
          id: nid,
          x: 450 + (Math.random() - 0.5) * 200,
          y: 230 + (Math.random() - 0.5) * 200,
          vx: 0,
          vy: 0,
          data: vr.data.data,
          depth: d,
        });
      const sr = await apiCall("/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: vr.data.data,
          k: k + 1,
          distance: "euclidean",
        }),
      });
      if (sr.ok && sr.data.results) {
        let cnt = 0;
        for (const hit of sr.data.results) {
          if (cnt >= k || !hit.data) continue;
          let tid = -1;
          for (const ex of graphNodes) {
            if (JSON.stringify(ex.data) === JSON.stringify(hit.data)) {
              tid = ex.id;
              break;
            }
          }
          if (tid === -1) {
            tid = 2000 + graphNodes.length;
            graphNodes.push({
              id: tid,
              x: 450 + (Math.random() - 0.5) * 300,
              y: 230 + (Math.random() - 0.5) * 300,
              vx: 0,
              vy: 0,
              data: hit.data,
              depth: d + 1,
            });
            next.push(tid);
          }
          if (
            !graphEdges.find(
              (e) =>
                (e.from === nid && e.to === tid) ||
                (e.from === tid && e.to === nid),
            )
          )
            graphEdges.push({ from: nid, to: tid, dist: hit.distance });
          cnt++;
        }
      }
    }
    frontier = next;
  }
  status.textContent = `${graphNodes.length} nodes, ${graphEdges.length} edges`;
  document.getElementById("graph-info").innerHTML =
    `<b>${graphNodes.length}</b> nodes, <b>${graphEdges.length}</b> edges from seed <b>#${seed}</b>`;
  if (graphAnim) cancelAnimationFrame(graphAnim);
  simGraph();
}

let graphHovered = -1,
  graphSelected = -1;

function graphFindNearest(mx, my) {
  let closest = -1,
    cd = 16;
  for (let i = 0; i < graphNodes.length; i++) {
    const d = Math.hypot(mx - graphNodes[i].x, my - graphNodes[i].y);
    if (d < cd) {
      cd = d;
      closest = i;
    }
  }
  return closest;
}

function getNodeNeighbors(nodeIdx) {
  if (nodeIdx < 0) return [];
  const node = graphNodes[nodeIdx];
  const neighbors = [];
  for (const e of graphEdges) {
    if (e.from === node.id) {
      const nb = graphNodes.find((n) => n.id === e.to);
      if (nb) neighbors.push({ id: nb.id, dist: e.dist });
    } else if (e.to === node.id) {
      const nb = graphNodes.find((n) => n.id === e.from);
      if (nb) neighbors.push({ id: nb.id, dist: e.dist });
    }
  }
  return neighbors.sort((a, b) => (a.dist || 0) - (b.dist || 0));
}

function showGraphDetail(idx) {
  const body = document.getElementById("graph-detail-body");
  if (idx < 0) {
    body.innerHTML =
      '<p class="viz-empty">Select a node to see its values and neighbors.</p>';
    return;
  }
  const node = graphNodes[idx];
  const neighbors = getNodeNeighbors(idx);
  const vecStr = `[${node.data.map((v) => v.toFixed(6)).join(", ")}]`;
  let html = `
    <div class="viz-point-head"><span class="viz-point-id">#${node.id}</span></div>
    <div class="detail-label">Hops from start</div>
    <div class="detail-value dim">${node.depth}</div>
    <div class="detail-label">Dimension</div>
    <div class="detail-value dim">${node.data.length}</div>
    <div class="detail-label">Neighbors (${neighbors.length})</div>`;
  if (neighbors.length) {
    html += '<ol class="viz-neighbors">';
    for (const nb of neighbors) {
      html += `<li class="viz-neighbor" style="cursor:default;grid-template-columns:1fr auto">
        <span class="viz-neighbor-id">#${nb.id}</span>
        <span class="viz-neighbor-sim">${nb.dist != null ? nb.dist.toFixed(4) : ""}</span>
      </li>`;
    }
    html += "</ol>";
  }
  html += `
    <details class="viz-raw"><summary>Raw values</summary><pre>${vecStr}</pre></details>
    <button class="btn btn-sm btn-outline" onclick="copyText('${escapeJsString(vecStr)}')">Copy vector</button>`;
  body.innerHTML = html;
}

// Force simulation tuning constants
const SIM_REPULSION = 4000; // node-node charge repulsion strength
const SIM_GE_REPULSION = 3000; // graph explorer repulsion (fewer, denser nodes)
const SIM_SPRING_LEN = 80; // spring rest length in px
const SIM_SPRING_K = 0.04; // spring stiffness
const SIM_GRAVITY = 0.001; // pull toward canvas center
const SIM_DAMPING = 0.85; // velocity damping per tick
const SIM_MAX_ITERS = 250; // similarity graph tick budget
const SIM_GE_MAX_ITERS = 200; // graph explorer tick budget
const SIM_NODE_MARGIN = 24; // keep nodes this far from canvas edges (px)

function simGraph() {
  const canvas = document.getElementById("graph-canvas");
  const [w, h] = canvasBox(canvas, 460),
    dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  let iter = 0;
  function drawGraphFrame() {
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = T.surface;
    ctx.fillRect(0, 0, w, h);

    const hovNode = graphHovered >= 0 ? graphNodes[graphHovered] : null;
    const selNode = graphSelected >= 0 ? graphNodes[graphSelected] : null;
    const highlightIds = new Set();
    if (hovNode) {
      highlightIds.add(hovNode.id);
      for (const nb of getNodeNeighbors(graphHovered)) highlightIds.add(nb.id);
    }
    if (selNode) {
      highlightIds.add(selNode.id);
      for (const nb of getNodeNeighbors(graphSelected)) highlightIds.add(nb.id);
    }

    // Edges
    for (const e of graphEdges) {
      const a = graphNodes.find((n) => n.id === e.from),
        b = graphNodes.find((n) => n.id === e.to);
      if (!a || !b) continue;
      const isHL =
        (hovNode && (e.from === hovNode.id || e.to === hovNode.id)) ||
        (selNode && (e.from === selNode.id || e.to === selNode.id));
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = isHL ? T.ink2 : T.axis;
      ctx.lineWidth = isHL ? 1.5 : 1;
      ctx.globalAlpha = hovNode || selNode ? (isHL ? 1 : 0.25) : 1;
      ctx.stroke();
      if (e.dist != null) {
        ctx.fillStyle = isHL ? T.ink2 : T.ink3;
        ctx.font = '10px "IBM Plex Sans", sans-serif';
        ctx.textAlign = "center";
        ctx.fillText(e.dist.toFixed(2), (a.x + b.x) / 2, (a.y + b.y) / 2 - 5);
      }
    }
    ctx.globalAlpha = 1;

    // Nodes
    for (let i = 0; i < graphNodes.length; i++) {
      const n = graphNodes[i];
      const isHov = i === graphHovered,
        isSel = i === graphSelected;
      const inHL = highlightIds.has(n.id);
      ctx.globalAlpha = hovNode || selNode ? (inHL ? 1 : 0.2) : 1;
      const r = n.depth === 0 ? 7 : 5;
      // Selected outer ring
      if (isSel) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, r + 4, 0, Math.PI * 2);
        ctx.strokeStyle = T.ink;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      // Hover ring
      if (isHov && !isSel) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, r + 4, 0, Math.PI * 2);
        ctx.strokeStyle = T.ink2;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
      ctx.fillStyle = n.depth === 0 ? T.s1 : isSel ? T.ink : T.other;
      ctx.fill();
      ctx.strokeStyle = T.surface;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = T.ink2;
      ctx.font = CHART_FONT_MONO;
      ctx.textAlign = "center";
      ctx.fillText(`#${n.id}`, n.x, n.y - 12);
    }
    ctx.globalAlpha = 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  function tick() {
    const alpha = Math.max(0.001, 0.3 * Math.pow(0.99, iter));
    for (let i = 0; i < graphNodes.length; i++)
      for (let j = i + 1; j < graphNodes.length; j++) {
        const dx = graphNodes[j].x - graphNodes[i].x,
          dy = graphNodes[j].y - graphNodes[i].y;
        const dist = Math.hypot(dx, dy) || 1,
          f = (SIM_REPULSION / (dist * dist)) * alpha;
        graphNodes[i].vx -= (dx / dist) * f;
        graphNodes[i].vy -= (dy / dist) * f;
        graphNodes[j].vx += (dx / dist) * f;
        graphNodes[j].vy += (dy / dist) * f;
      }
    for (const e of graphEdges) {
      const a = graphNodes.find((n) => n.id === e.from),
        b = graphNodes.find((n) => n.id === e.to);
      if (!a || !b) continue;
      const dx = b.x - a.x,
        dy = b.y - a.y,
        dist = Math.hypot(dx, dy) || 1,
        f = (dist - SIM_SPRING_LEN) * SIM_SPRING_K * alpha;
      a.vx += (dx / dist) * f;
      a.vy += (dy / dist) * f;
      b.vx -= (dx / dist) * f;
      b.vy -= (dy / dist) * f;
    }
    for (const n of graphNodes) {
      n.vx += (w / 2 - n.x) * SIM_GRAVITY * alpha;
      n.vy += (h / 2 - n.y) * SIM_GRAVITY * alpha;
      n.vx *= SIM_DAMPING;
      n.vy *= SIM_DAMPING;
      n.x += n.vx;
      n.y += n.vy;
      n.x = Math.max(SIM_NODE_MARGIN, Math.min(w - SIM_NODE_MARGIN, n.x));
      n.y = Math.max(SIM_NODE_MARGIN, Math.min(h - SIM_NODE_MARGIN, n.y));
    }
    drawGraphFrame();
    iter++;
    if (iter < SIM_MAX_ITERS) graphAnim = requestAnimationFrame(tick);
  }
  tick();

  canvas.addEventListener("mousemove", function (e) {
    const rect = this.getBoundingClientRect();
    const mx = e.clientX - rect.left,
      my = e.clientY - rect.top;
    const prev = graphHovered;
    graphHovered = graphFindNearest(mx, my);
    if (graphHovered !== prev) drawGraphFrame();
    if (graphHovered >= 0) {
      const node = graphNodes[graphHovered];
      const neighbors = getNodeNeighbors(graphHovered);
      const nodePreview = `[${node.data
        .slice(0, 3)
        .map((v) => v.toFixed(3))
        .join(", ")}${node.data.length > 3 ? ", ..." : ""}]`;
      showVizTooltip(
        e,
        `<div class="tt-id">Node #${node.id}</div>
         <div class="tt-dim">${node.data.length}D \u00b7 depth ${node.depth} \u00b7 ${neighbors.length} neighbors</div>
         <div class="tt-data">${nodePreview}</div>`,
      );
      document.getElementById("graph-info").innerHTML =
        `Node <b>#${node.id}</b> | depth ${node.depth} | ${neighbors.length} neighbors`;
    } else {
      hideVizTooltip();
    }
  });
  canvas.addEventListener("mouseleave", function () {
    graphHovered = -1;
    hideVizTooltip();
    drawGraphFrame();
  });
  canvas.addEventListener("click", function (e) {
    const rect = this.getBoundingClientRect();
    const mx = e.clientX - rect.left,
      my = e.clientY - rect.top;
    const idx = graphFindNearest(mx, my);
    graphSelected = idx === graphSelected ? -1 : idx;
    drawGraphFrame();
    showGraphDetail(graphSelected);
  });
}

// search

function renderSearchResults(r, tbodyId, metaId, tableId, emptyId) {
  const tbody = document.getElementById(tbodyId),
    meta = document.getElementById(metaId);
  const tbl = document.getElementById(tableId),
    empty = document.getElementById(emptyId);
  if (r.ok && r.data && r.data.results) {
    meta.textContent = `${r.data.results.length} results${
      r.data.latency_ms ? ` in ${r.data.latency_ms} ms` : ""
    }`;
    tbody.innerHTML = "";
    r.data.results.forEach((h, i) => {
      const dp = h.data
        ? `${JSON.stringify(h.data).slice(0, 50)}${
            JSON.stringify(h.data).length > 50 ? "..." : ""
          }`
        : "-";
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${i + 1}</td><td>${h.index ?? "-"}</td><td class="mono">${
        h.distance != null ? h.distance.toFixed(6) : "-"
      }</td><td class="mono">${dp}</td>`;
      tbody.appendChild(tr);
    });
    tbl.style.display = "table";
    empty.style.display = "none";
  } else {
    meta.textContent = "";
    tbody.innerHTML = "";
    tbl.style.display = "none";
    empty.textContent = `Search failed: ${r.data?.message || JSON.stringify(r.data)}`;
    empty.style.display = "block";
  }
}

async function doSearch() {
  try {
    const q = JSON.parse(document.getElementById("search-query").value.trim());
    const body = {
      query: q,
      k: parseInt(document.getElementById("search-k").value),
      distance: document.getElementById("search-dist").value,
    };
    const fk = document.getElementById("filter-key").value.trim(),
      fv = document.getElementById("filter-value").value.trim();
    if (fk && fv) body.filter = { key: fk, value: fv };
    const os = document.getElementById("search-oversampling").value.trim();
    if (os) body.oversampling_factor = parseFloat(os);
    const r = await apiCall("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    renderSearchResults(
      r,
      "search-tbody",
      "search-meta",
      "search-results",
      "search-empty",
    );
  } catch (e) {
    document.getElementById("search-meta").textContent = "";
    document.getElementById("search-empty").textContent =
      `Invalid JSON: ${e.message}`;
    document.getElementById("search-empty").style.display = "block";
    document.getElementById("search-results").style.display = "none";
  }
}
async function doRangeSearch() {
  try {
    const q = JSON.parse(document.getElementById("range-query").value.trim());
    const r = await apiCall("/search/range", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: q,
        radius: parseFloat(document.getElementById("range-radius").value),
        max_results: parseInt(document.getElementById("range-max").value),
        distance: document.getElementById("range-dist").value,
      }),
    });
    renderSearchResults(
      r,
      "range-tbody",
      "range-meta",
      "range-results",
      "range-empty",
    );
  } catch (e) {
    document.getElementById("range-meta").textContent = "";
    document.getElementById("range-empty").textContent =
      `Invalid JSON: ${e.message}`;
    document.getElementById("range-empty").style.display = "block";
    document.getElementById("range-results").style.display = "none";
  }
}

// console UI section

async function consoleSend() {
  const method = document.getElementById("con-method").value,
    url = document.getElementById("con-url").value;
  const bodyStr = document.getElementById("con-body").value.trim();
  const opts = { method };
  if (method !== "GET" && bodyStr) {
    try {
      JSON.parse(bodyStr);
    } catch (e) {
      document.getElementById("con-meta").innerHTML =
        '<span class="status-err">INVALID JSON</span>';
      document.getElementById("con-result").innerHTML =
        `<span class="json-null">${escapeHtml(String(e.message))}</span>`;
      return;
    }
    opts.headers = { "Content-Type": "application/json" };
    opts.body = bodyStr;
  }
  const t0 = performance.now();
  const r = await apiCall(url, opts);
  const ms = (performance.now() - t0).toFixed(1);
  const statusCls =
    r.status >= 200 && r.status < 400 ? "status-ok" : "status-err";
  document.getElementById("con-meta").innerHTML =
    `<span class="${statusCls}">${r.status}</span><span>${ms}ms</span>`;
  document.getElementById("con-result").innerHTML = jsonHighlight(r.data);
}

const QUICK_ACTIONS = {
  "/compact": ["Compaction finished", "Compaction failed"],
  "/stats": ["Stats refreshed", "Could not load stats"],
  "/health": ["Health check passed", "Health check failed"],
};
async function quickReq(method, url) {
  const [ok, fail] = QUICK_ACTIONS[url] || [`${method} ${url} succeeded`, `${method} ${url} failed`];
  const r = await apiCall(url, { method });
  showToast(
    r.ok ? ok : `${fail}: ${r.data?.message || r.data?.error || "server error"}`,
    r.ok ? "success" : "error",
  );
  if (r.ok) refreshOverview();
}

async function quickBackup() {
  const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const path = `/tmp/gigavector_backup_${ts}.gvb`;
  const r = await apiCall("/api/backups", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  const ok = r.ok && (!r.data || r.data.success !== false);
  showToast(
    ok ? `Backup saved to ${path}` : `Backup failed: ${r.data?.message || "server error"}`,
    ok ? "success" : "error",
  );
}
document.getElementById("con-url").addEventListener("keydown", (e) => {
  if (e.key === "Enter") consoleSend();
});

// chart primitives

function prepCanvas(canvas, w, h) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function fmtAgo(sec) {
  if (sec < 1) return "now";
  if (sec < 60) return `${Math.round(sec)} s ago`;
  return `${Math.round(sec / 60)} min ago`;
}

// Hover for canvas charts: the chart stores a hitTest(x) -> {index, x, lines}
// on canvas._hit; the shared handler redraws with the hover index and shows
// the tooltip. Installed once per canvas.
function ensureChartHover(canvas) {
  if (canvas._hoverBound) return;
  canvas._hoverBound = true;
  canvas.addEventListener("mousemove", (e) => {
    if (!canvas._hit) return;
    const r = canvas.getBoundingClientRect();
    const hit = canvas._hit(e.clientX - r.left, e.clientY - r.top);
    const idx = hit ? hit.index : -1;
    if (idx !== canvas._hoverIdx) {
      canvas._hoverIdx = idx;
      canvas._redraw && canvas._redraw();
    }
    if (!hit) return hideVizTooltip();
    const parts = hit.lines.map((t, i) => {
      const d = document.createElement("div");
      d.className = i === 0 ? "tt-id" : "tt-dim";
      d.textContent = t;
      return d;
    });
    vizTooltip.replaceChildren(...parts);
    vizTooltip.classList.add("visible");
    placeVizTooltip(r.left + hit.x, e.clientY);
  });
  canvas.addEventListener("mouseleave", () => {
    canvas._hoverIdx = -1;
    hideVizTooltip();
    canvas._redraw && canvas._redraw();
  });
}

function drawLineChart(canvas, data, opts = {}) {
  canvas._redraw = () => drawLineChart(canvas, data, opts);
  ensureChartHover(canvas);
  const [w, h] = canvasBox(canvas, opts.height || 160, 32);
  const ctx = prepCanvas(canvas, w, h);
  const fmt = opts.fmt || ((v) => v.toFixed(1));
  const pad = { t: 8, r: 8, b: opts.spanSec ? 22 : 8, l: 44 },
    pw = w - pad.l - pad.r,
    ph = h - pad.t - pad.b;
  ctx.fillStyle = T.surface;
  ctx.fillRect(0, 0, w, h);
  const vals = data.filter((v) => v != null);
  if (!vals.length) {
    ctx.fillStyle = T.ink3;
    ctx.font = CHART_FONT;
    ctx.textAlign = "center";
    ctx.fillText(opts.empty || "Waiting for samples", w / 2, h / 2);
    canvas._hit = null;
    return;
  }
  // Magnitudes: anchor at zero and round the top so noise does not read as
  // a swing.
  let mn = Math.min(0, ...vals),
    mx = niceCeil(Math.max(...vals) * 1.1);
  const tickDigits = mx - mn >= 10 ? 0 : mx - mn >= 1 ? 1 : 2;
  ctx.font = CHART_FONT;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = Math.round(pad.t + (ph * i) / 4) + 0.5;
    ctx.strokeStyle = i === 4 ? T.axis : T.grid;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(pad.l + pw, y);
    ctx.stroke();
    ctx.fillStyle = T.ink3;
    ctx.textAlign = "right";
    ctx.fillText((mx - ((mx - mn) * i) / 4).toFixed(tickDigits), pad.l - 8, y + 4);
  }
  // The window is always the full span; short histories sit at the right.
  const slots = opts.spanSec && opts.intervalSec ? Math.round(opts.spanSec / opts.intervalSec) : data.length;
  const offset = Math.max(0, slots - data.length);
  const X = (i) => pad.l + (pw * (i + offset)) / Math.max(1, slots - 1);
  const Y = (v) => pad.t + ph - ((v - mn) / (mx - mn)) * ph;
  if (opts.spanSec) {
    ctx.fillStyle = T.ink3;
    ctx.textAlign = "left";
    ctx.fillText(fmtAgo(opts.spanSec), pad.l, h - 6);
    ctx.textAlign = "right";
    ctx.fillText("now", pad.l + pw, h - 6);
  }
  const color = opts.color || T.s1;
  // Area then line, breaking at gaps (null samples).
  const runs = [];
  let run = [];
  data.forEach((v, i) => {
    if (v == null) {
      if (run.length) runs.push(run);
      run = [];
    } else run.push(i);
  });
  if (run.length) runs.push(run);
  for (const r of runs) {
    if (opts.fill && r.length > 1) {
      ctx.beginPath();
      ctx.moveTo(X(r[0]), Y(data[r[0]]));
      for (const i of r) ctx.lineTo(X(i), Y(data[i]));
      ctx.lineTo(X(r[r.length - 1]), pad.t + ph);
      ctx.lineTo(X(r[0]), pad.t + ph);
      ctx.closePath();
      ctx.fillStyle = color + "1f";
      ctx.fill();
    }
    ctx.beginPath();
    ctx.moveTo(X(r[0]), Y(data[r[0]]));
    for (const i of r) ctx.lineTo(X(i), Y(data[i]));
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.stroke();
    if (r.length === 1) {
      ctx.beginPath();
      ctx.arc(X(r[0]), Y(data[r[0]]), 2.5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }
  }
  // Crosshair for the hovered sample.
  const hi = canvas._hoverIdx;
  if (hi != null && hi >= 0 && hi < data.length) {
    const x = Math.round(X(hi)) + 0.5;
    ctx.strokeStyle = T.ink3;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, pad.t);
    ctx.lineTo(x, pad.t + ph);
    ctx.stroke();
    if (data[hi] != null) {
      ctx.beginPath();
      ctx.arc(X(hi), Y(data[hi]), 4, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = T.surface;
      ctx.stroke();
    }
  }
  canvas._hit = (x) => {
    if (x < pad.l - 4 || x > pad.l + pw + 4) return null;
    const step = pw / Math.max(1, slots - 1);
    const i = Math.round((x - pad.l) / step) - offset;
    if (i < 0 || i >= data.length) return null;
    const ago = opts.intervalSec ? (data.length - 1 - i) * opts.intervalSec : null;
    const v = data[i];
    return {
      index: i,
      x: X(i),
      lines: [
        v == null ? "No data" : `${fmt(v)}${opts.unit ? " " + opts.unit : ""}`,
        ago != null ? fmtAgo(ago) : `Sample ${i + 1}`,
      ],
    };
  };
}

function drawSparkline(canvas, data) {
  if (!canvas) return;
  canvas._redraw = () => drawSparkline(canvas, data);
  const p = canvas.parentElement;
  const w = Math.max(40, p.clientWidth),
    h = Math.max(20, p.clientHeight);
  const ctx = prepCanvas(canvas, w, h);
  const pts = data.slice(-48);
  const vals = pts.filter((v) => v != null);
  if (vals.length < 2) return;
  const mx = Math.max(...vals) || 1,
    mn = Math.min(0, ...vals);
  const X = (i) => 1 + ((w - 4) * i) / Math.max(1, pts.length - 1);
  const Y = (v) => h - 2 - ((v - mn) / (mx - mn || 1)) * (h - 5);
  ctx.beginPath();
  let started = false;
  pts.forEach((v, i) => {
    if (v == null) {
      started = false;
      return;
    }
    if (!started) ctx.moveTo(X(i), Y(v));
    else ctx.lineTo(X(i), Y(v));
    started = true;
  });
  ctx.strokeStyle = T.s1;
  ctx.lineWidth = 1.5;
  ctx.lineJoin = "round";
  ctx.stroke();
  const li = pts.length - 1;
  if (pts[li] != null) {
    ctx.beginPath();
    ctx.arc(X(li), Y(pts[li]), 2.5, 0, Math.PI * 2);
    ctx.fillStyle = T.s1;
    ctx.fill();
  }
}

function drawBars(canvas, labels, values, opts = {}) {
  canvas._redraw = () => drawBars(canvas, labels, values, opts);
  ensureChartHover(canvas);
  const [w, h] = canvasBox(canvas, opts.height || 180, 32);
  const ctx = prepCanvas(canvas, w, h);
  ctx.fillStyle = T.surface;
  ctx.fillRect(0, 0, w, h);
  if (!values.length) {
    ctx.fillStyle = T.ink3;
    ctx.font = CHART_FONT;
    ctx.textAlign = "center";
    ctx.fillText(opts.empty || "No data yet", w / 2, h / 2);
    canvas._hit = null;
    return;
  }
  const pad = { t: 8, r: 8, b: opts.xTitle ? 38 : 22, l: 44 },
    pw = w - pad.l - pad.r,
    ph = h - pad.t - pad.b;
  const mx = niceCeil(Math.max(...values) * 1.1);
  ctx.font = CHART_FONT;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = Math.round(pad.t + (ph * i) / 4) + 0.5;
    ctx.strokeStyle = i === 4 ? T.axis : T.grid;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(pad.l + pw, y);
    ctx.stroke();
    ctx.fillStyle = T.ink3;
    ctx.textAlign = "right";
    const tv = mx - (mx * i) / 4;
    ctx.fillText(tv >= 1000 ? `${(tv / 1000).toFixed(1)}k` : tv.toFixed(mx >= 10 ? 0 : 1), pad.l - 8, y + 4);
  }
  const slot = pw / values.length,
    bw = Math.max(2, slot - 2);
  const hi = canvas._hoverIdx;
  const every = Math.ceil(values.length / Math.max(1, Math.floor(pw / 44)));
  values.forEach((v, i) => {
    const x = pad.l + i * slot + 1,
      bh = (v / mx) * ph,
      y = pad.t + ph - bh;
    ctx.fillStyle = hi === i ? T.ink2 : T.s1;
    if (bh > 0) {
      const r = Math.min(2, bw / 2, bh);
      ctx.beginPath();
      ctx.moveTo(x, pad.t + ph);
      ctx.lineTo(x, y + r);
      ctx.quadraticCurveTo(x, y, x + r, y);
      ctx.lineTo(x + bw - r, y);
      ctx.quadraticCurveTo(x + bw, y, x + bw, y + r);
      ctx.lineTo(x + bw, pad.t + ph);
      ctx.closePath();
      ctx.fill();
    }
    if (i % every === 0) {
      ctx.fillStyle = T.ink3;
      ctx.textAlign = "center";
      ctx.fillText(labels[i], x + bw / 2, pad.t + ph + 15);
    }
  });
  if (opts.xTitle) {
    ctx.fillStyle = T.ink3;
    ctx.textAlign = "center";
    ctx.fillText(opts.xTitle, pad.l + pw / 2, h - 4);
  }
  canvas._hit = (x) => {
    const i = Math.floor((x - pad.l) / slot);
    if (i < 0 || i >= values.length) return null;
    return {
      index: i,
      x: pad.l + i * slot + slot / 2,
      lines: opts.tip ? opts.tip(i) : [labels[i], String(values[i])],
    };
  };
}

function niceCeil(v) {
  if (!(v > 0)) return 1;
  const p = 10 ** Math.floor(Math.log10(v)),
    n = v / p;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * p;
}

function drawGauge(canvas, value, max, opts = {}) {
  const [w, h] = canvasBox(canvas, opts.height || 160, 32);
  const ctx = prepCanvas(canvas, w, h);
  ctx.fillStyle = T.surface;
  ctx.fillRect(0, 0, w, h);
  const cx = w / 2,
    cy = h * 0.62,
    r = Math.min(w * 0.3, h * 0.45);
  const pct = Math.min(value / (max || 1), 1);
  ctx.lineWidth = 10;
  ctx.lineCap = "butt";
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, 2 * Math.PI);
  ctx.strokeStyle = T.grid;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, Math.PI + Math.PI * pct);
  ctx.strokeStyle = pct > 0.85 ? T.critical : pct > 0.6 ? T.warning : T.good;
  ctx.stroke();
  ctx.fillStyle = T.ink;
  ctx.font = '500 20px "IBM Plex Sans", sans-serif';
  ctx.textAlign = "center";
  ctx.fillText(`${(pct * 100).toFixed(0)}%`, cx, cy + 4);
  if (opts.label) {
    ctx.fillStyle = T.ink3;
    ctx.font = CHART_FONT;
    ctx.fillText(opts.label, cx, cy + 24);
  }
}

// monitoring

const monRing = { qps: [], ips: [], latency: [], mem: [], lastHist: null };
let monTimer = null;

async function refreshMonitoring() {
  const r = await apiCall("/api/detailed-stats");
  if (!r.ok) return;
  const D = parseDetailed(r.data);
  let lat = D.latNumber;
  if (lat == null && D.hist) {
    const prev = monRing.lastHist;
    monRing.lastHist = { n: D.hist.total_samples, sum: D.hist.sum_latency_us };
    const dn = prev ? D.hist.total_samples - prev.n : 0;
    lat = dn > 0 ? (D.hist.sum_latency_us - prev.sum) / dn / 1000 : null;
  }
  const push = (arr, v) => {
    arr.push(v == null || !Number.isFinite(v) ? null : v);
    if (arr.length > 60) arr.shift();
  };
  push(monRing.qps, D.qps);
  push(monRing.ips, D.ips);
  push(monRing.latency, lat);
  push(monRing.mem, D.mem.total_bytes / 1048576);
  setText("mon-qps", D.qps.toFixed(1));
  setText("mon-ips", D.ips.toFixed(1));
  const lastLat = [...monRing.latency].reverse().find((v) => v != null);
  setText("mon-lat", lastLat != null ? fmtMs(lastLat) : "-");
  setText("mon-mem", formatBytes(D.mem.total_bytes));
  const common = { fill: true, spanSec: 60, intervalSec: 1 };
  drawLineChart(document.getElementById("mon-qps-chart"), monRing.qps, {
    ...common,
    unit: "queries/s",
  });
  drawLineChart(document.getElementById("mon-latency-chart"), monRing.latency, {
    ...common,
    fmt: fmtMs,
    empty: "No searches in the last minute",
  });
  drawLineChart(document.getElementById("mon-mem-chart"), monRing.mem, {
    ...common,
    fmt: (v) => `${v.toFixed(1)} MB`,
  });
  document.getElementById("mon-detail-json").innerHTML = jsonHighlight(r.data);
}

// SQL console

async function runSQL() {
  const query = document.getElementById("sql-input").value.trim();
  if (!query) return;
  document.getElementById("sql-explain").style.display = "none";
  const t0 = performance.now();
  const r = await apiCall("/api/sql/execute", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  const ms = (performance.now() - t0).toFixed(1);
  document.getElementById("sql-meta").textContent = r.ok
    ? `${r.data.row_count} rows in ${ms}ms`
    : `Error: ${r.data.message || JSON.stringify(r.data)}`;
  if (r.ok && r.data.columns) {
    const thead = document.getElementById("sql-thead");
    const cols = r.data.columns || [];
    thead.innerHTML = `<tr>${cols
      .map((c) => `<th>${escapeHtml(c)}</th>`)
      .join("")}</tr>`;
    const tbody = document.getElementById("sql-tbody");
    tbody.innerHTML = "";
    (r.data.rows || []).forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = cols
        .map((c) => {
          const v = row ? row[c] : null;
          if (typeof v === "number") return `<td class="mono">${v}</td>`;
          return `<td class="mono">${escapeHtml(
            v == null ? "-" : String(v),
          )}</td>`;
        })
        .join("");
      tbody.appendChild(tr);
    });
    document.getElementById("sql-result").style.display = "block";
  } else {
    document.getElementById("sql-result").style.display = "none";
  }
}

async function explainSQL() {
  const query = document.getElementById("sql-input").value.trim();
  if (!query) return;
  document.getElementById("sql-result").style.display = "none";
  const r = await apiCall("/api/sql/explain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  const el = document.getElementById("sql-explain");
  el.style.display = "block";
  el.innerHTML = r.ok
    ? jsonHighlight(r.data)
    : `<span class="json-null">Error: ${r.data.message || ""}</span>`;
  document.getElementById("sql-meta").textContent = r.ok
    ? "Query plan"
    : "Error";
}

document.getElementById("sql-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runSQL();
});

//  backup & restore section

async function createBackup() {
  const path = document.getElementById("bk-path").value.trim();
  const body = path ? { path } : {};
  const r = await apiCall("/api/backups", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const el = document.getElementById("bk-create-result");
  el.style.display = "block";
  el.innerHTML = jsonHighlight(r.data);
  const ok = r.ok && (!r.data || r.data.success !== false);
  showToast(
    ok ? "Backup created" : "Backup failed. See the response for details.",
    ok ? "success" : "error",
  );
}

async function restoreBackup() {
  const path = document.getElementById("bk-restore-path").value.trim();

  if (!path) {
    showToast("Enter the backup file path first", "error");
    return;
  }
  if (!confirm("This will replace the current database. Continue?")) return;
  const r = await apiCall("/api/backups/restore", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  const el = document.getElementById("bk-restore-result");
  el.style.display = "block";
  el.innerHTML = jsonHighlight(r.data);
  const ok = r.ok && (!r.data || r.data.success !== false);
  showToast(
    ok ? "Restored from backup" : "Restore failed. See the response for details.",
    ok ? "success" : "error",
  );
}

async function readBackupHeader() {
  const path = document.getElementById("bk-header-path").value.trim();
  if (!path) {
    showToast("Enter the backup file path first", "error");
    return;
  }
  const r = await apiCall(
    `/api/backups/header?path=${encodeURIComponent(path)}`,
  );
  const el = document.getElementById("bk-header-result");
  el.style.display = "block";
  el.innerHTML = jsonHighlight(r.data);
}

let importData = null,
  importCols = [];

const dropZone = document.getElementById("import-drop");
const fileInput = document.getElementById("import-file");

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () =>
  dropZone.classList.remove("dragover"),
);
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files.length) handleImportFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) handleImportFile(fileInput.files[0]);
});

function handleImportFile(file) {
  const reader = new FileReader();
  reader.onload = function (e) {
    const text = e.target.result;
    try {
      if (file.name.endsWith(".json")) {
        const parsed = JSON.parse(text);
        importData = Array.isArray(parsed) ? parsed : [parsed];
      } else if (file.name.endsWith(".jsonl")) {
        importData = text
          .trim()
          .split("\n")
          .map((l) => JSON.parse(l));
      } else {
        const lines = text.trim().split("\n");
        const header = lines[0]
          .split(",")
          .map((h) => h.trim().replace(/^"|"$/g, ""));
        importCols = header;
        importData = lines.slice(1).map((line) => {
          const vals = line
            .split(",")
            .map((v) => v.trim().replace(/^"|"$/g, ""));
          const obj = {};
          header.forEach((h, i) => (obj[h] = vals[i]));
          return obj;
        });
      }
      document.getElementById("import-drop-text").textContent =
        `${file.name}: ${importData.length.toLocaleString()} records`;
      showImportPreview();
    } catch (err) {
      showToast(`Could not read the file: ${err.message}`, "error");
    }
  };
  reader.readAsText(file);
}

function showImportPreview() {
  document.getElementById("import-preview").style.display = "block";
  const sample = importData[0] || {};
  const keys = Object.keys(sample);
  const mappings = document.getElementById("import-mappings");
  mappings.innerHTML =
    '<div class="section-desc">Choose which column holds the vector values and which become metadata.</div>';
  const dataOpts = keys
    .map((k) => `<option value="${k}">${k}</option>`)
    .join("");
  mappings.innerHTML += `<div class="mapping-row"><span style="min-width:110px;color:var(--text-2)">Vector values</span>
    <select id="import-map-data"><option value="__auto__">Auto-detect</option>
    ${dataOpts}
    </select></div>
    <div class="mapping-row"><span style="min-width:110px;color:var(--text-2)">Metadata</span>
    <select id="import-map-meta"><option value="__none__">None</option><option value="__all__">All remaining</option>
    ${dataOpts}
    </select></div>`;
}

async function runImport() {
  if (!importData || !importData.length) {
    showToast("Choose a file to import first", "error");
    return;
  }
  const dataCol = document.getElementById("import-map-data").value;
  const metaCol = document.getElementById("import-map-meta").value;
  const bar = document.getElementById("import-progress-bar");
  bar.style.display = "block";
  const fill = document.getElementById("import-progress-fill");
  const status = document.getElementById("import-status");
  const batchSize = 50;
  let inserted = 0,
    errors = 0;
  for (let i = 0; i < importData.length; i += batchSize) {
    const batch = importData.slice(i, i + batchSize).map((row) => {
      let data;
      if (dataCol === "__auto__") {
        const vals = Object.values(row);
        data = Array.isArray(vals[0])
          ? vals[0]
          : vals.map(Number).filter((v) => !isNaN(v));
      } else {
        const v = row[dataCol];
        data = Array.isArray(v)
          ? v
          : typeof v === "string"
            ? JSON.parse(v)
            : [Number(v)];
      }
      let metadata = null;
      if (metaCol === "__all__") {
        metadata = {};
        for (const [k, v] of Object.entries(row)) {
          if (k !== dataCol) metadata[k] = String(v);
        }
      } else if (metaCol !== "__none__") {
        metadata = { [metaCol]: String(row[metaCol]) };
      }
      return { data, metadata };
    });
    const r = await apiCall("/api/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ vectors: batch }),
    });
    if (r.ok) {
      inserted += r.data.inserted;
      errors += r.data.errors;
    }
    fill.style.width = `${Math.min(100, ((i + batchSize) / importData.length) * 100).toFixed(0)}%`;
    status.textContent = `${inserted.toLocaleString()} imported, ${errors.toLocaleString()} failed`;
  }
  fill.style.width = "100%";
  const el = document.getElementById("import-result");
  el.style.display = "block";
  el.innerHTML = jsonHighlight({ inserted, errors, total: importData.length });
  showToast(`Imported ${inserted.toLocaleString()} vectors`, "success");
}

function resetImport() {
  importData = null;
  importCols = [];
  document.getElementById("import-drop-text").textContent =
    "Drop a file here or click to choose one";
  document.getElementById("import-preview").style.display = "none";
  document.getElementById("import-result").style.display = "none";
  document.getElementById("import-progress-bar").style.display = "none";
  fileInput.value = "";
}

async function loadNamespaces() {
  // namespaces
  const r = await apiCall("/api/namespaces");
  const tbody = document.getElementById("ns-tbody");
  const empty = document.getElementById("ns-empty");
  if (r.ok && r.data.namespaces) {
    if (!r.data.namespaces.length) {
      tbody.innerHTML = "";
      empty.textContent = "No namespaces yet. Create one above.";
      empty.style.display = "block";
      return;
    }
    empty.style.display = "none";
    tbody.innerHTML = r.data.namespaces
      .map((ns) => {
        const escNs = escapeHtml(ns);
        const jsNs = escapeJsString(ns);
        return `<tr>
          <td>${escNs}</td>
          <td>
            <button class="btn btn-sm btn-outline" onclick="nsInfo('${jsNs}')">Info</button>
            <button class="btn btn-sm btn-danger" onclick="deleteNamespace('${jsNs}')">Delete</button>
          </td>
        </tr>`;
      })
      .join("");
  }
}

async function createNamespace() {
  const name = document.getElementById("ns-name").value.trim();
  if (!name) {
    showToast("Enter a namespace name first", "error");
    return;
  }
  const r = await apiCall("/api/namespaces", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name,
      dimension: parseInt(document.getElementById("ns-dim").value),
      index_type: document.getElementById("ns-index").value,
    }),
  });
  showToast(
    r.ok ? "Namespace created" : `Could not create namespace: ${r.data?.message || "unknown error"}`,
    r.ok ? "success" : "error",
  );
  if (r.ok) loadNamespaces();
}

async function deleteNamespace(name) {
  if (!confirm(`Delete namespace "${name}" and all of its vectors? This cannot be undone.`)) return;
  const r = await apiCall(`/api/namespaces/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  showToast(
    r.ok ? `Deleted namespace "${name}"` : `Could not delete namespace "${name}"`,
    r.ok ? "success" : "error",
  );
  loadNamespaces();
}

async function nsInfo(name) {
  const r = await apiCall(`/api/namespaces/${encodeURIComponent(name)}/info`);
  const el = document.getElementById("ns-info");
  el.style.display = "block";
  el.innerHTML = jsonHighlight(r.data);
}

//  graph UI

let geNodes = [],
  geEdges = [],
  geAnim = null;

async function geAddNode() {
  const label = document.getElementById("ge-label").value.trim() || "Node";
  const r = await apiCall("/api/graph/node", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label }),
  });
  document.getElementById("ge-status").textContent = r.ok
    ? `Node #${r.data.id} added`
    : `Error: ${r.data.message || ""}`;
  if (r.ok) showToast(`Added node #${r.data.id}`, "success");
}

async function geAddEdge() {
  const src = parseInt(document.getElementById("ge-src").value);
  const tgt = parseInt(document.getElementById("ge-tgt").value);
  const label = document.getElementById("ge-elabel").value.trim() || "";
  const r = await apiCall("/api/graph/edge", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source: src, target: tgt, label }),
  });
  document.getElementById("ge-status").textContent = r.ok
    ? `Edge #${r.data.id} added`
    : `Error: ${r.data.message || ""}`;
}

async function geBFS() {
  const start = parseInt(document.getElementById("ge-bfs-start").value) || 0;
  const depth = parseInt(document.getElementById("ge-bfs-depth").value) || 3;
  const r = await apiCall(`/api/graph/bfs?start=${start}&max_depth=${depth}`);
  if (!r.ok) {
    document.getElementById("ge-status").textContent =
      `Error: ${r.data.message || ""}`;
    return;
  }
  geNodes = r.data.nodes.map((n, i) => ({
    ...n,
    x: 450 + (Math.random() - 0.5) * 300,
    y: 230 + (Math.random() - 0.5) * 300,
    vx: 0,
    vy: 0,
  }));
  geEdges = r.data.edges || [];
  document.getElementById("ge-status").textContent =
    `${geNodes.length} nodes, ${geEdges.length} edges`;
  if (geAnim) cancelAnimationFrame(geAnim);
  simGraphExplorer();
}

async function geShortestPath() {
  const from = parseInt(document.getElementById("ge-sp-from").value);
  const to = parseInt(document.getElementById("ge-sp-to").value);
  const r = await apiCall(`/api/graph/shortest-path?from=${from}&to=${to}`);
  if (r.ok && r.data.path !== null) {
    document.getElementById("ge-status").textContent =
      `Path: ${r.data.node_ids.join(" -> ")} (weight: ${r.data.total_weight.toFixed(2)})`;
    showToast(`Shortest path has ${r.data.node_ids.length} nodes`, "success");
  } else {
    document.getElementById("ge-status").textContent =
      r.data.message || "No path found";
  }
}

async function geRefresh() {
  const r = await apiCall("/api/graph/bfs?start=0&max_depth=10");
  if (!r.ok) {
    document.getElementById("ge-status").textContent = "No graph data. Add a node first, or check the server.";
    return;
  }
  geNodes = r.data.nodes.map((n) => ({
    ...n,
    x: 450 + (Math.random() - 0.5) * 300,
    y: 230 + (Math.random() - 0.5) * 300,
    vx: 0,
    vy: 0,
  }));
  geEdges = r.data.edges || [];
  document.getElementById("ge-status").textContent =
    `${geNodes.length} nodes, ${geEdges.length} edges`;
  if (geAnim) cancelAnimationFrame(geAnim);
  simGraphExplorer();
}

function simGraphExplorer() {
  const canvas = document.getElementById("ge-canvas");
  const w = canvas.parentElement.clientWidth,
    h = 460,
    dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  let iter = 0;
  const idMap = new Map();
  geNodes.forEach((n, i) => idMap.set(n.id, i));

  function draw() {
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = T.surface;
    ctx.fillRect(0, 0, w, h);
    for (const e of geEdges) {
      const ai = idMap.get(e.source),
        bi = idMap.get(e.target);
      if (ai == null || bi == null) continue;
      const a = geNodes[ai],
        b = geNodes[bi];
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = T.axis;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    for (const n of geNodes) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, 6, 0, Math.PI * 2);
      ctx.fillStyle = T.s1;
      ctx.fill();
      ctx.strokeStyle = T.surface;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = T.ink2;
      ctx.font = CHART_FONT;
      ctx.textAlign = "center";
      ctx.fillText(n.label || `#${n.id}`, n.x, n.y - 10);
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  function tick() {
    const alpha = Math.max(0.001, 0.3 * Math.pow(0.99, iter));
    for (let i = 0; i < geNodes.length; i++)
      for (let j = i + 1; j < geNodes.length; j++) {
        const dx = geNodes[j].x - geNodes[i].x,
          dy = geNodes[j].y - geNodes[i].y;
        const dist = Math.hypot(dx, dy) || 1,
          f = (SIM_GE_REPULSION / (dist * dist)) * alpha;
        geNodes[i].vx -= (dx / dist) * f;
        geNodes[i].vy -= (dy / dist) * f;
        geNodes[j].vx += (dx / dist) * f;
        geNodes[j].vy += (dy / dist) * f;
      }
    for (const e of geEdges) {
      const ai = idMap.get(e.source),
        bi = idMap.get(e.target);
      if (ai == null || bi == null) continue;
      const a = geNodes[ai],
        b = geNodes[bi];
      const dx = b.x - a.x,
        dy = b.y - a.y,
        dist = Math.hypot(dx, dy) || 1,
        f = (dist - SIM_SPRING_LEN) * SIM_SPRING_K * alpha;
      a.vx += (dx / dist) * f;
      a.vy += (dy / dist) * f;
      b.vx -= (dx / dist) * f;
      b.vy -= (dy / dist) * f;
    }
    for (const n of geNodes) {
      n.vx += (w / 2 - n.x) * SIM_GRAVITY * alpha;
      n.vy += (h / 2 - n.y) * SIM_GRAVITY * alpha;
      n.vx *= SIM_DAMPING;
      n.vy *= SIM_DAMPING;
      n.x += n.vx;
      n.y += n.vy;
      n.x = Math.max(SIM_NODE_MARGIN, Math.min(w - SIM_NODE_MARGIN, n.x));
      n.y = Math.max(SIM_NODE_MARGIN, Math.min(h - SIM_NODE_MARGIN, n.y));
    }
    draw();
    iter++;
    if (iter < SIM_GE_MAX_ITERS) geAnim = requestAnimationFrame(tick);
  }
  tick();
  document.getElementById("ge-info").innerHTML =
    `<b>${geNodes.length}</b> nodes, <b>${geEdges.length}</b> edges`;
}

async function loadCollections() {
  //  collections
  const r = await apiCall("/api/collections");
  const picker = document.getElementById("collectionPicker");
  // keep first option (Default)
  while (picker.options.length > 1) picker.remove(1);
  if (r.ok && r.data.collections) {
    r.data.collections.forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c.name;
      opt.textContent = `${c.name} (${c.vector_count} vectors)`;
      picker.appendChild(opt);
    });
    if (r.data.collections.length > 0)
      document.getElementById("collectionPickerWrap").style.display = "";
  }
}

document
  .getElementById("collectionPicker")
  .addEventListener("change", function () {
    activeCollection = this.value;
  });
loadCollections();

async function loadCluster() {
  // cluster
  const r = await apiCall("/api/cluster/info");
  if (r.ok) {
    document.getElementById("cl-nodes").textContent = r.data.total_nodes;
    document.getElementById("cl-active").textContent = r.data.active_nodes;
    document.getElementById("cl-shards").textContent = r.data.total_shards;
    document.getElementById("cl-vectors").textContent = r.data.total_vectors;
    document.getElementById("cl-health").textContent = r.data.healthy
      ? "Healthy"
      : "Degraded";
    document.getElementById("cl-health").style.color = r.data.healthy
      ? "var(--green)"
      : "var(--red)";
  }
  const sr = await apiCall("/api/cluster/shards");
  if (sr.ok && sr.data.shards && sr.data.shards.length > 0) {
    const tbody = document.getElementById("cluster-shard-tbody");
    tbody.innerHTML = "";
    sr.data.shards.forEach((s) => {
      const tr = document.createElement("tr");
      tr.innerHTML =
        '<td class="mono">' +
        s.shard_id +
        '</td><td class="mono">' +
        s.node_address +
        "</td><td>" +
        s.state +
        '</td><td class="mono">' +
        s.vector_count +
        '</td><td class="mono">' +
        s.replica_count +
        "</td>";
      tbody.appendChild(tr);
    });
    document.getElementById("cluster-shard-table").style.display = "";
    document.getElementById("cluster-shard-empty").style.display = "none";
  }
}

// navigation/ routing

const viewHooks = {
  monitoring: () => {
    refreshMonitoring();
    if (!monTimer) monTimer = setInterval(refreshMonitoring, 1000);
  },
  visualize: () => {
    if (!vizData) runVisualization();
  },
  namespaces: loadNamespaces,
  cluster: loadCluster,
};
const viewLeaveHooks = {
  monitoring: () => {
    if (monTimer) {
      clearInterval(monTimer);
      monTimer = null;
    }
  },
};
let currentView = "overview";

document.querySelectorAll(".sidebar-nav a").forEach((a) => {
  a.addEventListener("click", (e) => {
    e.preventDefault();
    const view = a.dataset.view;
    if (viewLeaveHooks[currentView]) viewLeaveHooks[currentView]();
    currentView = view;
    document
      .querySelectorAll(".sidebar-nav a")
      .forEach((x) => x.classList.remove("active"));
    a.classList.add("active");
    document
      .querySelectorAll(".view")
      .forEach((v) => v.classList.remove("active"));
    document.getElementById("view-" + view).classList.add("active");
    document.getElementById("viewTitle").textContent =
      a.querySelector("span").textContent;
    if (view === "vectors") loadPoints();
    hideVizTooltip();
    if (viewHooks[view]) viewHooks[view]();
  });
});
