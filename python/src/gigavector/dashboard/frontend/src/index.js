const API = window.location.origin;

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
    showToast("Copied!", "success");
  } catch (_) {
    showToast("Clipboard unavailable", "error");
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
async function refreshOverview() {
  const [info, stats, health] = await Promise.all([
    apiCall("/api/dashboard/info"),
    apiCall("/stats"),
    apiCall("/health"),
  ]);
  const dot = document.getElementById("statusDot"),
    txt = document.getElementById("statusText");
  if (health.ok && health.data) {
    const s = health.data.status || "unknown";
    dot.className = "status-dot " + s;
    txt.textContent = s.charAt(0).toUpperCase() + s.slice(1);
    document.getElementById("ov-health-json").innerHTML = jsonHighlight(
      health.data,
    );
    const up = health.data.uptime_seconds;
    document.getElementById("uptimeBadge").textContent =
      up != null ? "Up " + formatUptime(up) : "";
  } else {
    dot.className = "status-dot unhealthy";
    txt.textContent = "Unreachable";
    document.getElementById("ov-health-json").innerHTML =
      '<span class="json-null">Server unreachable</span>';
    document.getElementById("uptimeBadge").textContent = "";
  }
  if (info.ok && info.data) {
    document.getElementById("ov-count").textContent = (
      info.data.vector_count ?? 0
    ).toLocaleString();
    document.getElementById("ov-dim").textContent = info.data.dimension ?? "-";
    document.getElementById("ov-index").textContent =
      info.data.index_type ?? "-";
    document.getElementById("ov-version").textContent =
      info.data.version ?? "-";
    document.getElementById("footer-version").textContent =
      "GigaVector v" + (info.data.version || "?");
    document.getElementById("footer-index").textContent =
      (info.data.index_type || "") + " | dim " + (info.data.dimension || "?");
    document.getElementById("ov-server-info").innerHTML = jsonHighlight(
      info.data,
    );
  }
  if (stats.ok && stats.data) {
    document.getElementById("ov-reqs").textContent = (
      stats.data.total_requests ??
      (stats.data.total_inserts || 0) + (stats.data.total_queries || 0)
    ).toLocaleString();
    document.getElementById("ov-qps").textContent =
      stats.data.queries_per_second ?? "-";
    document.getElementById("ov-errors").textContent = (
      stats.data.error_count ?? 0
    ).toLocaleString();
    document.getElementById("ov-sent").textContent = formatBytes(
      stats.data.total_bytes_sent,
    );
    document.getElementById("ov-recv").textContent =
      stats.data.total_bytes_received != null
        ? "Recv: " + formatBytes(stats.data.total_bytes_received)
        : "";
  }
}
function startRefresh() {
  refreshOverview();
  refreshTimer = setInterval(refreshOverview, 2500);
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
  document.getElementById("points-count").textContent = `${r.data.total} total`;
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
      '<div style="padding:20px;font-size:12px;color:var(--text-muted)">No vectors found.</div>';
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
        <div class="point-badge">${Array.isArray(p.data) ? `${p.data.length}d` : ""}</div>
      </div>`;
    })
    .join("");
}

function selectPoint(i) {
  selectedPointIdx = i;
  renderPointsList();
  const p = currentPoints[i];
  document.getElementById("point-detail").innerHTML = `
    <div style="font-family:var(--mono);font-size:12px;font-weight:600;color:var(--accent);margin-bottom:12px">Point #${p.index}</div>
    <div class="json-view" style="max-height:300px;margin-bottom:12px">${jsonHighlight(p)}</div>
    <button class="btn btn-sm btn-danger" onclick="deletePointFromBrowser(${p.index})">Delete</button>`;
}

async function deletePointFromBrowser(id) {
  if (!confirm(`Delete vector ${id}?`)) return;
  const r = await apiCall(`/vectors/${id}`, { method: "DELETE" });
  showToast(r.ok ? `Deleted #${id}` : "Failed", r.ok ? "success" : "error");
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
      r.ok ? "Vector added" : `Failed: ${JSON.stringify(r.data)}`,
      r.ok ? "success" : "error",
    );
  } catch (e) {
    showToast(`Invalid JSON: ${e.message}`, "error");
  }
}

// scatter plot vis

let vizData = null;
async function runVisualization() {
  const limit = parseInt(document.getElementById("viz-limit").value) || 200;
  const algo = document.getElementById("viz-algo").value;
  const status = document.getElementById("viz-status");
  const info = document.getElementById("scatter-info");
  status.textContent = "Loading vectors...";
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
// the #0c1118 surface. A scatter overlaps every pair, so only three hues pass;
// further groups fold into a muted "Other" bucket instead of a generated hue.
const VIZ_PALETTE = ["#3987e5", "#d95926", "#199e70"];
const VIZ_OTHER_COLOR = "#898781";
const VIZ_DEFAULT_COLOR = "#3987e5";
const VIZ_SURFACE = "#0c1118";
const VIZ_GRID = "#1a2430";
const VIZ_AXIS = "#2c3a4b";
const VIZ_INK_PRIMARY = "#e9eff8";
const VIZ_INK_SECONDARY = "#9fb1c8";

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
  const w = canvas.parentElement.clientWidth,
    h = 460;
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
    ctx.font = '14px "Source Sans 3", sans-serif';
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
  ctx.font = '10px "IBM Plex Mono"';
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
  status.textContent = "Building...";
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
  status.textContent = `${graphNodes.length} nodes | ${graphEdges.length} edges`;
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
      '<div style="color:var(--text-muted);font-size:13px">Click a node on the graph to inspect it here.</div>';
    return;
  }
  const node = graphNodes[idx];
  const neighbors = getNodeNeighbors(idx);
  const vecStr = `[${node.data.map((v) => v.toFixed(6)).join(", ")}]`;
  let html = `
    <div class="detail-label">Node ID</div>
    <div class="detail-value" style="font-size:16px;font-weight:600;color:var(--accent)">#${node.id}</div>
    <div class="detail-label">Depth</div>
    <div class="detail-value dim">${node.depth}</div>
    <div class="detail-label">Dimension</div>
    <div class="detail-value dim">${node.data.length}</div>
    <div class="detail-label">Neighbors (${neighbors.length})</div>`;
  if (neighbors.length) {
    html += '<div style="margin-bottom:14px">';
    for (const nb of neighbors) {
      html += `<div style="display:flex;justify-content:space-between;padding:3px 0;font-family:var(--mono);font-size:11px;border-bottom:1px solid var(--border-light)">
        <span style="color:var(--text)">#${nb.id}</span>
        <span style="color:var(--text-muted)">${nb.dist != null ? nb.dist.toFixed(4) : ""}</span>
      </div>`;
    }
    html += "</div>";
  }
  html += `
    <div class="detail-label">Vector Data</div>
    <div class="detail-value" style="font-size:11px;line-height:1.6;max-height:180px;overflow-y:auto;background:var(--bg-input);padding:10px;border-radius:6px">${vecStr}</div>
    <button class="btn btn-sm btn-outline" style="margin-top:4px" onclick="copyText('${escapeJsString(vecStr)}')">Copy Vector</button>`;
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
  const w = canvas.parentElement.clientWidth,
    h = 460,
    dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  let iter = 0;
  function drawGraphFrame() {
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#0c1118";
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
      ctx.strokeStyle = isHL ? "#2ee6c8" : "#d5d4cc";
      ctx.lineWidth = isHL ? 2 : 1;
      ctx.globalAlpha = hovNode || selNode ? (isHL ? 1 : 0.25) : 1;
      ctx.stroke();
      if (e.dist != null) {
        ctx.fillStyle = isHL ? "#2ee6c8" : "#999990";
        ctx.font = '9px "IBM Plex Mono"';
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
        ctx.arc(n.x, n.y, r + 6, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(46,230,200,0.22)";
        ctx.lineWidth = 3;
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(n.x, n.y, r + 3, 0, Math.PI * 2);
        ctx.strokeStyle = "#2ee6c8";
        ctx.lineWidth = 2.5;
        ctx.stroke();
      }
      // Hover ring
      if (isHov && !isSel) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, r + 4, 0, Math.PI * 2);
        ctx.strokeStyle = "#2ee6c8";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
      ctx.fillStyle = n.depth === 0 ? "#2ee6c8" : isSel ? "#2ee6c8" : "#555";
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = "#a0a0a0";
      ctx.font = '10px "IBM Plex Mono"';
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
      r.data.latency_ms ? ` in ${r.data.latency_ms}ms` : ""
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
    empty.textContent = `Error: ${JSON.stringify(r.data)}`;
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

async function quickReq(method, url) {
  const el = document.getElementById("ov-action-result");
  if (el) el.innerHTML = '<span class="json-null">Loading...</span>';
  const r = await apiCall(url, { method });
  if (el) el.innerHTML = jsonHighlight(r.data);
  showToast(r.ok ? `${method} ${url} OK` : "Error", r.ok ? "success" : "error");
}

async function quickBackup() {
  const el = document.getElementById("ov-action-result");
  if (el) el.innerHTML = '<span class="json-null">Backing up...</span>';
  const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const path = `/tmp/gigavector_backup_${ts}.gvb`;
  const r = await apiCall("/api/backups", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  if (el) el.innerHTML = jsonHighlight(r.data);
  const ok = r.ok && (!r.data || r.data.success !== false);
  showToast(
    ok ? `Backup saved to ${path}` : "Backup failed",
    ok ? "success" : "error",
  );
}
document.getElementById("con-url").addEventListener("keydown", (e) => {
  if (e.key === "Enter") consoleSend();
});

// char primitives

function drawLineChart(canvas, data, opts = {}) {
  const w = canvas.parentElement.clientWidth - 32,
    h = opts.height || 160,
    dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const pad = { t: 10, r: 10, b: 24, l: 50 },
    pw = w - pad.l - pad.r,
    ph = h - pad.t - pad.b;
  if (!data.length) {
    ctx.fillStyle = "#999";
    ctx.font = "12px sans-serif";
    ctx.fillText("No data", w / 2 - 20, h / 2);
    return;
  }
  let mn = Infinity,
    mx = -Infinity;
  for (const v of data) {
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  if (mn === mx) {
    mn -= 1;
    mx += 1;
  }
  ctx.fillStyle = "#0c1118";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#1e2b3b";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (ph * i) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(pad.l + pw, y);
    ctx.stroke();
    ctx.fillStyle = "#999";
    ctx.font = '10px "IBM Plex Mono"';
    ctx.textAlign = "right";
    ctx.fillText((mx - ((mx - mn) * i) / 4).toFixed(1), pad.l - 4, y + 3);
  }
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t + ph - ((data[0] - mn) / (mx - mn)) * ph);
  for (let i = 1; i < data.length; i++) {
    const x = pad.l + (pw * i) / (data.length - 1),
      y = pad.t + ph - ((data[i] - mn) / (mx - mn)) * ph;
    ctx.lineTo(x, y);
  }
  ctx.strokeStyle = opts.color || "#2ee6c8";
  ctx.lineWidth = 2;
  ctx.stroke();
  if (opts.fill) {
    ctx.lineTo(pad.l + pw, pad.t + ph);
    ctx.lineTo(pad.l, pad.t + ph);
    ctx.closePath();
    ctx.fillStyle = opts.fillColor || "rgba(46,230,200,0.10)";
    ctx.fill();
  }
  if (opts.label) {
    ctx.fillStyle = "#999";
    ctx.font = '10px "IBM Plex Mono"';
    ctx.textAlign = "center";
    ctx.fillText(opts.label, w / 2, h - 4);
  }
}

function drawBarChart(canvas, labels, values, opts = {}) {
  const w = canvas.parentElement.clientWidth - 32,
    h = opts.height || 180,
    dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const pad = { t: 14, r: 10, b: 36, l: 50 },
    pw = w - pad.l - pad.r,
    ph = h - pad.t - pad.b;
  const mx = Math.max(...values) * 1.15 || 1;
  ctx.fillStyle = "#0c1118";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#1e2b3b";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (ph * i) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(pad.l + pw, y);
    ctx.stroke();
    ctx.fillStyle = "#999";
    ctx.font = '10px "IBM Plex Mono"';
    ctx.textAlign = "right";
    ctx.fillText(
      (mx - (mx * i) / 4).toFixed(opts.decimals ?? 1),
      pad.l - 4,
      y + 3,
    );
  }
  const colors = opts.colors || ["#2ee6c8", "#555", "#0369a1", "#16a34a"];
  const gap = pw / (labels.length * 2 + 1),
    bw = gap * 1.5;
  for (let i = 0; i < labels.length; i++) {
    const x = pad.l + gap + i * (bw + gap),
      bh = (values[i] / mx) * ph;
    ctx.fillStyle = colors[i % colors.length];
    ctx.fillRect(x, pad.t + ph - bh, bw, bh);
    ctx.fillStyle = "#e5e5e5";
    ctx.font = 'bold 11px "IBM Plex Mono"';
    ctx.textAlign = "center";
    ctx.fillText(
      values[i].toFixed(opts.decimals ?? 1),
      x + bw / 2,
      pad.t + ph - bh - 4,
    );
    ctx.fillStyle = "#a0a0a0";
    ctx.font = '10px "IBM Plex Mono"';
    ctx.fillText(labels[i], x + bw / 2, h - 8);
  }
}

function drawGauge(canvas, value, max, opts = {}) {
  const w = canvas.parentElement.clientWidth - 32,
    h = opts.height || 160,
    dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.fillStyle = "#0c1118";
  ctx.fillRect(0, 0, w, h);
  const cx = w / 2,
    cy = h * 0.6,
    r = Math.min(w, h) * 0.38;
  const pct = Math.min(value / (max || 1), 1);
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, 2 * Math.PI);
  ctx.strokeStyle = "#1e2b3b";
  ctx.lineWidth = 12;
  ctx.lineCap = "round";
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, Math.PI + Math.PI * pct);
  ctx.strokeStyle = pct > 0.85 ? "#dc2626" : pct > 0.6 ? "#eab308" : "#16a34a";
  ctx.lineWidth = 12;
  ctx.lineCap = "round";
  ctx.stroke();
  ctx.fillStyle = "#e5e5e5";
  ctx.font = 'bold 20px "IBM Plex Mono"';
  ctx.textAlign = "center";
  ctx.fillText(`${(pct * 100).toFixed(0)}%`, cx, cy + 6);
  if (opts.label) {
    ctx.fillStyle = "#999";
    ctx.font = '10px "IBM Plex Mono"';
    ctx.fillText(opts.label, cx, cy + 22);
  }
}

// monitoring

const monRing = { qps: [], latency: [], mem: [] };
let monTimer = null;

async function refreshMonitoring() {
  const r = await apiCall("/api/detailed-stats");
  if (!r.ok) return;
  const d = r.data;
  const qps = d.queries_per_second || 0;
  const lat = d.search_latency || 0;
  const mem = d.memory || 0;
  monRing.qps.push(qps);
  if (monRing.qps.length > 60) monRing.qps.shift();
  monRing.latency.push(lat);
  if (monRing.latency.length > 60) monRing.latency.shift();
  monRing.mem.push(mem);
  if (monRing.mem.length > 60) monRing.mem.shift();
  document.getElementById("mon-qps").textContent = qps.toFixed(1);
  document.getElementById("mon-vecs").textContent = (
    d.basic_stats?.total_vectors || 0
  ).toLocaleString();
  document.getElementById("mon-mem").textContent = formatBytes(mem);
  document.getElementById("mon-health").textContent = d.health_status || "ok";
  drawLineChart(document.getElementById("mon-qps-chart"), monRing.qps, {
    color: "#2ee6c8",
    fill: true,
    label: "Last 60 samples",
  });
  drawLineChart(document.getElementById("mon-latency-chart"), monRing.latency, {
    color: "#0369a1",
    fill: true,
    label: "Search latency (ms)",
  });
  drawGauge(document.getElementById("mon-mem-chart"), mem, 1073741824, {
    label: `${formatBytes(mem)} / 1 GB`,
  });
  document.getElementById("mon-detail-json").innerHTML = jsonHighlight(d);
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
  showToast(ok ? "Backup created" : "Backup failed", ok ? "success" : "error");
}

async function restoreBackup() {
  const path = document.getElementById("bk-restore-path").value.trim();

  if (!path) {
    showToast("Enter backup path", "error");
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
    ok ? "Restore complete" : "Restore failed",
    ok ? "success" : "error",
  );
}

async function readBackupHeader() {
  const path = document.getElementById("bk-header-path").value.trim();
  if (!path) {
    showToast("Enter backup path", "error");
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
      dropZone.innerHTML = `<b>${file.name}</b> - ${importData.length} records`;
      showImportPreview();
    } catch (err) {
      showToast(`Parse error: ${err.message}`, "error");
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
    '<div class="section-desc">Map file columns to vector fields:</div>';
  const dataOpts = keys
    .map((k) => `<option value="${k}">${k}</option>`)
    .join("");
  mappings.innerHTML += `<div class="mapping-row"><span style="min-width:100px;font-weight:600">Vector Data</span>
    <select id="import-map-data"><option value="__auto__">Auto-detect</option>
    ${dataOpts}
    </select></div>
    <div class="mapping-row"><span style="min-width:100px;font-weight:600">Metadata</span>
    <select id="import-map-meta"><option value="__none__">None</option><option value="__all__">All remaining</option>
    ${dataOpts}
    </select></div>`;
}

async function runImport() {
  if (!importData || !importData.length) {
    showToast("No data to import", "error");
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
    status.textContent = `${inserted} inserted, ${errors} errors`;
  }
  fill.style.width = "100%";
  const el = document.getElementById("import-result");
  el.style.display = "block";
  el.innerHTML = jsonHighlight({ inserted, errors, total: importData.length });
  showToast(`Import complete: ${inserted} vectors`, "success");
}

function resetImport() {
  importData = null;
  importCols = [];
  dropZone.innerHTML =
    'Drop a CSV or JSON file here, or click to browse<br><div style="font-size:12px;color:var(--text-muted);margin-top:8px">Supported: .csv, .json, .jsonl</div>';
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
      empty.textContent = "No namespaces yet.";
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
    showToast("Enter namespace name", "error");
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
    r.ok ? "Namespace created" : `Error: ${r.data.message || ""}`,
    r.ok ? "success" : "error",
  );
  if (r.ok) loadNamespaces();
}

async function deleteNamespace(name) {
  if (!confirm(`Delete namespace "${name}"?`)) return;
  const r = await apiCall(`/api/namespaces/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  showToast(r.ok ? "Deleted" : "Error", r.ok ? "success" : "error");
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
  if (r.ok) showToast(`Node #${r.data.id} created`, "success");
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
    showToast(`Path found: ${r.data.node_ids.length} nodes`, "success");
  } else {
    document.getElementById("ge-status").textContent =
      r.data.message || "No path found";
  }
}

async function geRefresh() {
  const r = await apiCall("/api/graph/bfs?start=0&max_depth=10");
  if (!r.ok) {
    document.getElementById("ge-status").textContent = "Graph empty or error";
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
    ctx.fillStyle = "#0c1118";
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
      ctx.strokeStyle = "#333";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
    for (const n of geNodes) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, 6, 0, Math.PI * 2);
      ctx.fillStyle = "#2ee6c8";
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = "#a0a0a0";
      ctx.font = '10px "IBM Plex Mono"';
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
    if (viewHooks[view]) viewHooks[view]();
  });
});
