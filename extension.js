const vscode = require("vscode");
const path = require("path");
const { spawn } = require("child_process");

/** @param {string} s */
function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * @param {vscode.Uri | undefined} resource
 * @returns {Promise<string | undefined>}
 */
async function getMsPythonInterpreterPath(resource) {
  try {
    const extension = vscode.extensions.getExtension("ms-python.python");
    if (!extension) {
      return undefined;
    }
    /** @type {any} */
    const api = extension.isActive
      ? extension.exports
      : await extension.activate();
    const details = api?.settings?.getExecutionDetails?.(resource);
    if (!details) {
      return undefined;
    }
    if (Array.isArray(details.execCommand) && details.execCommand.length > 0) {
      return details.execCommand[0];
    }
    if (
      typeof details.pythonPath === "string" &&
      details.pythonPath.length > 0
    ) {
      return details.pythonPath;
    }
  } catch {
    // ignore — fall back to python3
  }
  return undefined;
}

/**
 * @param {vscode.Uri | undefined} resource
 * @returns {Promise<string>}
 */
async function resolvePythonExecutable(resource) {
  const cfg = vscode.workspace.getConfiguration("torchCheckpointInspector");
  const override = cfg.get("pythonPath", "");
  if (typeof override === "string" && override.trim().length > 0) {
    return override.trim();
  }
  const fromPythonExt = await getMsPythonInterpreterPath(resource);
  if (fromPythonExt) {
    return fromPythonExt;
  }
  return process.platform === "win32" ? "python" : "python3";
}

/**
 * @param {string} pythonPath
 * @param {string} scriptPath
 * @param {string} filePath
 * @param {boolean} unsafe
 * @returns {Promise<{ code: number | null, json: any, stderr: string, parseError?: string, rawOut?: string }>}
 */
function runInspector(pythonPath, scriptPath, filePath, unsafe) {
  const args = [scriptPath, filePath];
  if (unsafe) {
    args.push("--unsafe");
  }
  return new Promise((resolve, reject) => {
    const chunks = [];
    const errChunks = [];
    const proc = spawn(pythonPath, args, {
      stdio: ["ignore", "pipe", "pipe"],
    });
    proc.stdout.on("data", (d) => chunks.push(d));
    proc.stderr.on("data", (d) => errChunks.push(d));
    proc.on("error", (err) => reject(err));
    proc.on("close", (code) => {
      const out = Buffer.concat(chunks).toString("utf8").trim();
      const stderr = Buffer.concat(errChunks).toString("utf8");
      try {
        const json = JSON.parse(out);
        resolve({ code, json, stderr });
      } catch (e) {
        resolve({
          code,
          json: null,
          stderr,
          rawOut: out,
          parseError: String(e),
        });
      }
    });
  });
}

/** @param {string | undefined} dtype */
function dtypeByteSize(dtype) {
  const s = String(dtype || "")
    .toLowerCase()
    .replace(/^torch\./, "");
  const map = {
    float64: 8,
    double: 8,
    float32: 4,
    float: 4,
    bfloat16: 2,
    float16: 2,
    half: 2,
    int64: 8,
    long: 8,
    int32: 4,
    int: 4,
    int16: 2,
    short: 2,
    int8: 1,
    char: 1,
    uint8: 1,
    byte: 1,
    bool: 1,
    complex64: 8,
    complex128: 16,
  };
  return map[s] ?? null;
}

/** @param {number} n */
function humanizeInt(n) {
  if (!Number.isFinite(n)) {
    return String(n);
  }
  const abs = Math.abs(n);
  if (abs < 10000) {
    return String(Math.round(n));
  }
  if (abs < 1e6) {
    return `${(n / 1e3).toFixed(1)}k`.replace(/\.0k$/, "k");
  }
  if (abs < 1e9) {
    return `${(n / 1e6).toFixed(2)}M`.replace(/\.00M$/, "M");
  }
  return `${(n / 1e9).toFixed(2)}G`.replace(/\.00G$/, "G");
}

/** @param {number} n */
function humanizeBytes(n) {
  if (!Number.isFinite(n) || n < 0) {
    return "—";
  }
  if (n < 1024) {
    return `${Math.round(n)} B`;
  }
  if (n < 1024 * 1024) {
    return `${(n / 1024).toFixed(1)} KB`;
  }
  if (n < 1024 ** 3) {
    return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

/**
 * @typedef {{ kindCounts: Record<string, number>, totalNumel: number, dtypeNumel: Record<string, number>, estimatedBytes: number, unknownDtypeNumel: number }} WalkerAcc
 */

/** @returns {WalkerAcc} */
function emptyWalkerAcc() {
  return {
    kindCounts: {},
    totalNumel: 0,
    dtypeNumel: {},
    estimatedBytes: 0,
    unknownDtypeNumel: 0,
  };
}

/** @param {any} node @param {WalkerAcc} acc */
function walkCheckpointStats(node, acc) {
  if (node === null || node === undefined) {
    return;
  }
  if (typeof node !== "object") {
    return;
  }
  const k = node.kind;
  if (!k) {
    return;
  }

  if (k === "Tensor") {
    acc.kindCounts.Tensor = (acc.kindCounts.Tensor || 0) + 1;
    const n = Number(node.numel) || 0;
    acc.totalNumel += n;
    const dt = String(node.dtype ?? "");
    acc.dtypeNumel[dt] = (acc.dtypeNumel[dt] || 0) + n;
    const bpe = dtypeByteSize(dt);
    if (bpe != null) {
      acc.estimatedBytes += n * bpe;
    } else if (n > 0) {
      acc.unknownDtypeNumel += n;
    }
    return;
  }

  if (k === "Dict") {
    acc.kindCounts.Dict = (acc.kindCounts.Dict || 0) + 1;
    const ch = node.children || {};
    for (const key of Object.keys(ch)) {
      walkCheckpointStats(ch[key], acc);
    }
    return;
  }

  if (k === "List") {
    acc.kindCounts.List = (acc.kindCounts.List || 0) + 1;
    for (const it of node.items || []) {
      walkCheckpointStats(it, acc);
    }
    return;
  }

  if (k === "Module") {
    acc.kindCounts.Module = (acc.kindCounts.Module || 0) + 1;
    if (node.state_dict !== undefined) {
      walkCheckpointStats(node.state_dict, acc);
    }
    return;
  }

  if (
    k === "ndarray" ||
    k === "DataFrame" ||
    k === "Series" ||
    k === "Index" ||
    k === "Truncated" ||
    k === "Bytes" ||
    k === "Other" ||
    k === "Error"
  ) {
    acc.kindCounts[k] = (acc.kindCounts[k] || 0) + 1;
  }
}

/** @param {any} root */
function collectRootCheckpointHints(root) {
  if (!root || typeof root !== "object" || root.kind !== "Dict") {
    return [];
  }
  const keys = new Set(Object.keys(root.children || {}));
  /** @type {Array<[string, string]>} */
  const pairs = [
    ["state_dict", "state_dict (weights)"],
    ["model_state_dict", "model_state_dict"],
    ["optimizer_state_dict", "optimizer_state_dict"],
    ["scheduler_state_dict", "scheduler_state_dict"],
    ["lr_scheduler", "lr_scheduler"],
    ["epoch", "epoch"],
    ["global_step", "global_step"],
    ["iteration", "iteration"],
    ["model", "model"],
    ["ema_state_dict", "ema_state_dict"],
    ["scaler", "AMP scaler"],
    ["amp_scaler", "AMP scaler"],
  ];
  const found = [];
  for (const [key, label] of pairs) {
    if (keys.has(key)) {
      found.push(label);
    }
  }
  return found;
}

/** @param {any} j */
function renderOverviewSection(j) {
  const acc = emptyWalkerAcc();
  walkCheckpointStats(j.root, acc);
  const hints = collectRootCheckpointHints(j.root);
  const tensorCount = acc.kindCounts.Tensor || 0;
  const dtypeEntries = Object.entries(acc.dtypeNumel).sort(
    (a, b) => b[1] - a[1],
  );
  const topDtypes = dtypeEntries.slice(0, 5);

  const stat = (label, value) =>
    `<div class="stat"><span class="stat-label">${escapeHtml(label)}</span><span class="stat-value">${value}</span></div>`;

  let html =
    '<section class="overview card" aria-label="Structure overview"><h3 class="section-title">Overview</h3><div class="stat-grid">';

  if (j.file_size_bytes != null && Number.isFinite(j.file_size_bytes)) {
    html += stat("File size", escapeHtml(humanizeBytes(j.file_size_bytes)));
  }
  html += stat("Tensor leaves", escapeHtml(String(tensorCount)));
  html += stat("Total elements", escapeHtml(humanizeInt(acc.totalNumel)));
  const memPlain = humanizeBytes(acc.estimatedBytes);
  const memVal =
    acc.unknownDtypeNumel > 0
      ? `${escapeHtml(memPlain)} <span class="muted">(some dtypes unknown)</span>`
      : escapeHtml(memPlain);
  html += stat("Approx. tensor memory", memVal);

  html += "</div>";

  if (topDtypes.length > 0) {
    html +=
      '<div class="dtype-breakdown"><span class="muted">Dtypes by elements</span><ul class="dtype-list">';
    for (const [dt, nel] of topDtypes) {
      html += `<li><code>${escapeHtml(dt)}</code> <span class="muted">${escapeHtml(humanizeInt(nel))}</span></li>`;
    }
    html += "</ul></div>";
  }

  if (hints.length > 0) {
    html += `<div class="hints"><span class="muted">Looks like</span> <span class="hint-chips">${hints
      .map((h) => `<span class="chip chip-hint">${escapeHtml(h)}</span>`)
      .join(" ")}</div>`;
  }

  html += "</section>";
  return html;
}

/**
 * @param {number} min
 * @param {number} mean
 * @param {number} max
 */
function tensorStatBarHtml(min, mean, max) {
  if (
    !Number.isFinite(min) ||
    !Number.isFinite(mean) ||
    !Number.isFinite(max)
  ) {
    return "";
  }
  if (min > max) {
    return "";
  }
  if (max === min) {
    return '<div class="stat-bar stat-bar-flat" role="img" aria-label="min equals max"></div>';
  }
  const pMean = ((mean - min) / (max - min)) * 100;
  const clamped = Math.max(0, Math.min(100, pMean));
  return `<div class="stat-bar" style="--p-mean:${clamped}%" role="img" aria-label="min to max with mean marker"></div>`;
}

/**
 * @param {string} cls
 * @param {string} key
 * @param {string} valueHtml
 * @param {string} [title]
 */
function chip(cls, key, valueHtml, title) {
  const t = title ? ` title="${escapeHtml(title)}"` : "";
  return `<span class="chip ${cls}"${t}><span class="chip-k">${escapeHtml(key)}</span> ${valueHtml}</span>`;
}

/** @param {any} node @param {number} depth */
function renderTreeNode(node, depth) {
  const d = depth | 0;
  const openAttr = d < 1 ? " open" : "";

  if (node === null || node === undefined) {
    return `<span class="primitive">${escapeHtml(String(node))}</span>`;
  }
  if (typeof node !== "object") {
    return `<span class="primitive">${escapeHtml(JSON.stringify(node))}</span>`;
  }

  switch (node.kind) {
    case "Tensor": {
      const shapeStr = `[${(node.shape || []).join(", ")}]`;
      const numel = Number(node.numel) || 0;
      let row1 =
        chip("chip-shape", "shape", escapeHtml(shapeStr), "Tensor shape") +
        chip("chip-dtype", "dtype", escapeHtml(String(node.dtype)), "dtype") +
        chip(
          "chip-device",
          "device",
          escapeHtml(String(node.device)),
          "device",
        ) +
        chip(
          "chip-numel",
          "numel",
          escapeHtml(`${humanizeInt(numel)} (${numel})`),
          "element count",
        );

      let callouts = "";
      if (node.stats_skipped) {
        callouts += `<div class="callout muted">${escapeHtml(String(node.stats_skipped))}</div>`;
      }
      if (node.stats_error) {
        callouts += `<div class="callout warn-inline">${escapeHtml(String(node.stats_error))}</div>`;
      }

      let statsRow = "";
      const hasMin = node.min !== undefined;
      const hasMax = node.max !== undefined;
      const hasMean = node.mean !== undefined;
      if (hasMin || hasMax || hasMean) {
        const bits = [];
        if (hasMin) {
          bits.push(
            chip("chip-stat", "min", escapeHtml(String(node.min)), "min"),
          );
        }
        if (hasMean) {
          bits.push(
            chip("chip-stat", "mean", escapeHtml(String(node.mean)), "mean"),
          );
        }
        if (hasMax) {
          bits.push(
            chip("chip-stat", "max", escapeHtml(String(node.max)), "max"),
          );
        }
        statsRow = `<div class="chip-row">${bits.join("")}</div>`;
        if (
          hasMin &&
          hasMax &&
          hasMean &&
          typeof node.min === "number" &&
          typeof node.max === "number" &&
          typeof node.mean === "number"
        ) {
          statsRow += tensorStatBarHtml(node.min, node.mean, node.max);
        }
      }

      return `<div class="tensor-block"><div class="chip-row">${row1}</div>${statsRow}${callouts}</div>`;
    }
    case "ndarray": {
      const shapeStr = `[${(node.shape || []).join(", ")}]`;
      const n = Number(node.numel) || 0;
      const row =
        `<span class="kind-tag">NumPy</span>` +
        chip("chip-shape", "shape", escapeHtml(shapeStr)) +
        chip("chip-dtype", "dtype", escapeHtml(String(node.dtype))) +
        chip("chip-numel", "size", escapeHtml(String(n)));
      return `<div class="tensor-block ndarray-block"><div class="chip-row">${row}</div></div>`;
    }
    case "DataFrame": {
      const shape = (node.shape || []).join(" × ");
      const cols = (node.columns || []).join(", ");
      const trunc = node.truncated_columns ? " …" : "";
      const dt = node.dtypes ? escapeHtml(JSON.stringify(node.dtypes)) : "";
      return `<div class="pandas card-inner">
  <div class="pandas-head"><span class="kind-tag">DataFrame</span> <span class="muted">${escapeHtml(shape)} columns${escapeHtml(trunc)}</span></div>
  <pre class="cols cols-scroll"><code>${escapeHtml(cols)}</code></pre>
  ${node.truncated_dtypes ? '<span class="badge">dtypes truncated</span>' : ""}
  <pre class="dtypes-pre"><code>${dt}</code></pre>
</div>`;
    }
    case "Series": {
      return `<div class="pandas card-inner"><span class="kind-tag">Series</span> <span class="muted">name</span> <code>${escapeHtml(String(node.name))}</code> <span class="muted">len</span> ${node.length} <span class="muted">dtype</span> <code>${escapeHtml(String(node.dtype))}</code></div>`;
    }
    case "Index": {
      return `<div class="pandas card-inner"><span class="kind-tag">Index</span> <span class="muted">len</span> ${node.length} <span class="muted">dtype</span> <code>${escapeHtml(String(node.dtype))}</code> <pre class="cols cols-scroll"><code>${escapeHtml(String(node.head || ""))}</code></pre></div>`;
    }
    case "Dict": {
      const keys = Object.keys(node.children || {});
      const total = node.total_keys != null ? node.total_keys : keys.length;
      const keysTrunc = node.truncated_keys
        ? `<span class="badge">showing ${keys.length} of ${total} keys</span>`
        : "";
      const children = node.children || {};
      let inner = "";
      for (const k of keys) {
        inner += `<details class="node"${openAttr}>
  <summary><span class="kind-tag sm">Dict</span> <code class="key-code">${escapeHtml(k)}</code></summary>
  <div class="indent">${renderTreeNode(children[k], d + 1)}</div>
</details>`;
      }
      return `<div class="dict"><div class="container-head"><span class="kind-tag">Dict</span> <span class="muted">${total} keys</span> ${keysTrunc}</div>${inner || ' <em class="muted">empty</em>'}</div>`;
    }
    case "List": {
      const items = node.items || [];
      const trunc = node.truncated
        ? `<span class="badge">showing ${items.length} of ${node.length}</span>`
        : "";
      let inner = "";
      items.forEach((item, i) => {
        inner += `<details class="node"${openAttr}>
  <summary><span class="kind-tag sm">List</span> <code class="key-code">[${i}]</code></summary>
  <div class="indent">${renderTreeNode(item, d + 1)}</div>
</details>`;
      });
      return `<div class="list"><div class="container-head"><span class="kind-tag">List</span> <span class="muted">len ${node.length}</span> ${trunc}</div>${inner}</div>`;
    }
    case "Module": {
      const head = `<div class="module-head"><span class="kind-tag">Module</span> <code class="type-code">${escapeHtml(node.type || "")}</code></div>`;
      if (node.state_dict_error) {
        return `${head}<div class="warn">${escapeHtml(node.state_dict_error)}</div>`;
      }
      if (node.state_dict !== undefined) {
        return `${head}<div class="indent">${renderTreeNode(node.state_dict, d + 1)}</div>`;
      }
      return head;
    }
    case "Truncated":
      return `<div class="warn">${escapeHtml(node.reason || "")}: ${escapeHtml(node.repr || "")}</div>`;
    case "Bytes":
      return `<div class="leaf-row"><span class="kind-tag">Bytes</span> <span class="muted">len</span> ${node.len}</div>`;
    case "Other":
      return `<div class="leaf-row"><span class="kind-tag">Other</span> <code>${escapeHtml(node.type || "")}</code> — <span class="repr">${escapeHtml(node.repr || "")}</span></div>`;
    case "Error":
      return `<div class="err">${escapeHtml(node.message || "")}</div>`;
    default:
      return `<pre class="fallback">${escapeHtml(JSON.stringify(node, null, 2))}</pre>`;
  }
}

/**
 * @param {vscode.Webview} webview
 * @param {string} title
 * @param {string} filePath
 * @param {string} pythonUsed
 * @param {boolean} unsafeSetting
 * @param {any} payload
 */
function buildWebviewHtml(
  webview,
  title,
  filePath,
  pythonUsed,
  unsafeSetting,
  payload,
) {
  const csp = [
    `default-src 'none'`,
    `style-src ${webview.cspSource} 'unsafe-inline'`,
    `script-src ${webview.cspSource} 'unsafe-inline'`,
  ].join("; ");

  let body = "";
  if (payload.parseError || !payload.json) {
    body = `<div class="err">
  <h2>Could not parse inspector output</h2>
  <p><strong>Parse error:</strong> ${escapeHtml(payload.parseError || "")}</p>
  <p><strong>Exit code:</strong> ${payload.code ?? "n/a"}</p>
  <h3>stdout</h3>
  <pre>${escapeHtml((payload.rawOut || "").slice(0, 8000))}</pre>
  <h3>stderr</h3>
  <pre>${escapeHtml((payload.stderr || "").slice(0, 8000))}</pre>
</div>`;
  } else if (!payload.json.ok) {
    body = `<div class="err">
  <h2>Inspection failed</h2>
  <p>${escapeHtml(payload.json.error || "Unknown error")}</p>
  <pre>${escapeHtml((payload.json.traceback || payload.stderr || "").slice(0, 12000))}</pre>
</div>`;
  } else {
    const j = payload.json;
    const baseName = escapeHtml(j.file_name || path.basename(filePath));
    const unsafeBanner =
      j.load_mode === "unsafe_meta"
        ? `<div class="warn unsafe"><strong>Unsafe load path</strong> — file was loaded with <code>weights_only=False</code>. Only use this for trusted checkpoints.</div>`
        : "";
    body = `${unsafeBanner}
<header class="hero card">
  <div class="hero-title">${baseName}</div>
  <div class="hero-path" title="${escapeHtml(filePath)}">${escapeHtml(filePath)}</div>
  <div class="hero-badges">
    <span class="pill pill-load">${escapeHtml(j.load_mode || "")}</span>
    <span class="pill pill-torch">PyTorch ${escapeHtml(j.torch_version || "?")}</span>
  </div>
</header>
<details class="env card">
  <summary class="env-summary">Environment</summary>
  <div class="env-body">
    <div><span class="env-k">Interpreter</span> <code class="env-code">${escapeHtml(pythonUsed)}</code></div>
    <div><span class="env-k">allowUnsafeLoad</span> <code class="env-code">${unsafeSetting ? "true" : "false"}</code></div>
  </div>
</details>
${renderOverviewSection(j)}
<div class="toolbar card toolbar-row">
  <button type="button" id="btn-refresh">Refresh</button>
  <button type="button" id="btn-expand" class="secondary">Expand all</button>
  <button type="button" id="btn-collapse" class="secondary">Collapse all</button>
  <label class="filter-label"><span class="filter-text">Filter</span>
    <input type="search" id="filter-tree" placeholder="Key or substring…" autocomplete="off" />
  </label>
</div>
<section class="tree card" aria-label="Checkpoint structure">
  <h3 class="section-title">Structure</h3>
  <div class="tree-inner">${renderTreeNode(j.root, 0)}</div>
</section>`;
  }

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="${csp}">
  <title>${escapeHtml(title)}</title>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: var(--vscode-font-family, system-ui, sans-serif);
      font-size: var(--vscode-font-size, 13px);
      color: var(--vscode-foreground);
      background: var(--vscode-editor-background);
      margin: 0;
      padding: 1rem;
      overflow: auto;
      line-height: 1.45;
    }
    h2 { margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 600; }
    h3 { margin: 0 0 0.35rem 0; font-size: 0.85rem; font-weight: 600; color: var(--vscode-descriptionForeground); }
    .section-title { margin: 0 0 0.5rem 0; letter-spacing: 0.02em; text-transform: uppercase; font-size: 0.72rem; }
    pre {
      margin: 0.25rem 0;
      padding: 0.5rem;
      background: var(--vscode-textCodeBlock-background);
      border: 1px solid var(--vscode-panel-border);
      overflow: auto;
      font-size: 12px;
    }
    .card {
      background: var(--vscode-sideBar-background, var(--vscode-editorWidget-background));
      border: 1px solid var(--vscode-panel-border);
      border-radius: 6px;
      padding: 0.65rem 0.75rem;
      margin-bottom: 0.65rem;
    }
    .hero { padding: 0.75rem 0.85rem; }
    .hero-title { font-size: 1.15rem; font-weight: 600; }
    .hero-path {
      font-size: 11px;
      color: var(--vscode-descriptionForeground);
      margin-top: 0.25rem;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .hero-badges { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.5rem; }
    .pill {
      font-size: 11px;
      padding: 0.15rem 0.45rem;
      border-radius: 999px;
      border: 1px solid var(--vscode-panel-border);
      background: var(--vscode-textCodeBlock-background);
      font-family: var(--vscode-editor-font-family, monospace);
    }
    .pill-load { border-color: var(--vscode-focusBorder); }
    details.env > summary {
      cursor: pointer;
      user-select: none;
      font-weight: 600;
      color: var(--vscode-descriptionForeground);
    }
    details.env > summary:focus-visible {
      outline: 1px solid var(--vscode-focusBorder);
      outline-offset: 2px;
    }
    .env-body { margin-top: 0.5rem; line-height: 1.6; font-size: 12px; }
    .env-k { color: var(--vscode-descriptionForeground); margin-right: 0.35rem; }
    .env-code { font-size: 11px; word-break: break-all; }
    .stat-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
      gap: 0.5rem 0.75rem;
    }
    .stat { display: flex; flex-direction: column; gap: 0.1rem; }
    .stat-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--vscode-descriptionForeground); }
    .stat-value { font-weight: 600; font-size: 13px; }
    .dtype-breakdown { margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid var(--vscode-panel-border); }
    .dtype-list { margin: 0.25rem 0 0 1rem; padding: 0; font-size: 12px; }
    .hints { margin-top: 0.5rem; display: flex; flex-wrap: wrap; align-items: center; gap: 0.35rem; }
    .hint-chips { display: inline-flex; flex-wrap: wrap; gap: 0.25rem; }
    .muted { color: var(--vscode-descriptionForeground); }
    .toolbar-row {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.5rem;
    }
    .toolbar-row button,
    .toolbar-row .secondary {
      padding: 0.35rem 0.65rem;
      border-radius: 4px;
    }
    button {
      background: var(--vscode-button-background);
      color: var(--vscode-button-foreground);
      border: none;
      cursor: pointer;
      font-family: inherit;
      font-size: inherit;
    }
    button:hover { background: var(--vscode-button-hoverBackground); }
    button:focus-visible,
    #filter-tree:focus-visible {
      outline: 1px solid var(--vscode-focusBorder);
      outline-offset: 2px;
    }
    button.secondary {
      background: var(--vscode-button-secondaryBackground);
      color: var(--vscode-button-secondaryForeground);
    }
    button.secondary:hover { background: var(--vscode-button-secondaryHoverBackground); }
    .filter-label { display: flex; align-items: center; gap: 0.35rem; flex: 1 1 160px; min-width: 140px; }
    .filter-text { font-size: 12px; color: var(--vscode-descriptionForeground); white-space: nowrap; }
    #filter-tree {
      flex: 1;
      min-width: 0;
      padding: 0.3rem 0.45rem;
      font: inherit;
      color: var(--vscode-input-foreground);
      background: var(--vscode-input-background);
      border: 1px solid var(--vscode-input-border, var(--vscode-panel-border));
      border-radius: 4px;
    }
    .tree-inner { margin-top: 0.25rem; }
    .tree .tree-inner details.node { margin: 0.12rem 0; }
    .tree .tree-inner details.node.filter-hidden { display: none; }
    .tree summary {
      cursor: pointer;
      user-select: none;
      list-style-position: outside;
    }
    .tree summary:focus-visible {
      outline: 1px solid var(--vscode-focusBorder);
      outline-offset: 1px;
    }
    .indent {
      margin-left: 0.35rem;
      padding-left: 0.65rem;
      border-left: 2px solid color-mix(in srgb, var(--vscode-panel-border) 85%, transparent);
    }
    .kind-tag {
      display: inline-block;
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      padding: 0.08rem 0.35rem;
      border-radius: 3px;
      background: var(--vscode-badge-background);
      color: var(--vscode-badge-foreground);
      margin-right: 0.25rem;
      vertical-align: middle;
    }
    .kind-tag.sm { font-size: 9px; padding: 0.05rem 0.28rem; }
    .key-code, .type-code { font-size: 12px; }
    .container-head { margin-bottom: 0.35rem; display: flex; flex-wrap: wrap; align-items: center; gap: 0.35rem; }
    .chip-row { display: flex; flex-wrap: wrap; gap: 0.35rem; align-items: center; margin: 0.15rem 0; }
    .chip {
      display: inline-flex;
      align-items: baseline;
      gap: 0.25rem;
      font-family: var(--vscode-editor-font-family, monospace);
      font-size: 11px;
      padding: 0.2rem 0.45rem;
      border-radius: 4px;
      border: 1px solid var(--vscode-panel-border);
      background: var(--vscode-textCodeBlock-background);
    }
    .chip-k { color: var(--vscode-descriptionForeground); font-size: 10px; text-transform: uppercase; letter-spacing: 0.03em; }
    .chip-hint { text-transform: none; letter-spacing: normal; font-family: inherit; font-size: 11px; }
    .tensor-block {
      font-family: var(--vscode-editor-font-family, monospace);
      font-size: 12px;
      padding: 0.35rem 0.45rem;
      border: 1px solid var(--vscode-panel-border);
      border-radius: 4px;
      background: color-mix(in srgb, var(--vscode-textCodeBlock-background) 70%, var(--vscode-sideBar-background));
    }
    .stat-bar {
      height: 6px;
      border-radius: 3px;
      margin-top: 0.35rem;
      background: linear-gradient(
        90deg,
        var(--vscode-charts-blue, var(--vscode-terminal-ansiBlue)) 0%,
        var(--vscode-charts-orange, var(--vscode-terminal-ansiYellow)) var(--p-mean),
        var(--vscode-charts-purple, var(--vscode-terminal-ansiMagenta)) 100%
      );
      opacity: 0.92;
    }
    .stat-bar-flat {
      background: var(--vscode-disabledForeground);
      opacity: 0.35;
    }
    .callout { margin-top: 0.35rem; font-size: 11px; }
    .warn-inline { color: var(--vscode-errorForeground); }
    .pandas.card-inner { padding: 0.25rem 0; }
    .pandas-head { margin-bottom: 0.35rem; }
    .cols-scroll {
      max-height: 10rem;
      margin: 0.25rem 0;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .dtypes-pre {
      max-height: 6rem;
      margin: 0.25rem 0 0 0;
      font-size: 11px;
    }
    .module-head { margin-bottom: 0.35rem; }
    .leaf-row { font-size: 12px; }
    .badge { font-size: 11px; color: var(--vscode-descriptionForeground); }
    .warn {
      background: color-mix(in srgb, var(--vscode-inputValidation-warningBackground) 60%, var(--vscode-editor-background));
      border: 1px solid var(--vscode-inputValidation-warningBorder);
      padding: 0.5rem 0.75rem;
      margin-bottom: 0.65rem;
      border-radius: 6px;
    }
    .warn.unsafe { border-color: var(--vscode-inputValidation-errorBorder); }
    .err { color: var(--vscode-errorForeground); }
    .primitive { font-family: var(--vscode-editor-font-family, monospace); }
    .repr { word-break: break-all; }
    .fallback { max-height: 40vh; }
  </style>
</head>
<body>
  ${body}
  <script>
    (function () {
      const vscode = acquireVsCodeApi();
      const btn = document.getElementById('btn-refresh');
      if (btn) {
        btn.addEventListener('click', function () {
          vscode.postMessage({ type: 'refresh' });
        });
      }
      const expand = document.getElementById('btn-expand');
      const collapse = document.getElementById('btn-collapse');
      const filterInput = document.getElementById('filter-tree');
      function allTreeDetails() {
        return Array.prototype.slice.call(document.querySelectorAll('.tree .tree-inner details.node'));
      }
      if (expand) {
        expand.addEventListener('click', function () {
          allTreeDetails().forEach(function (d) {
            d.open = true;
          });
        });
      }
      if (collapse) {
        collapse.addEventListener('click', function () {
          allTreeDetails().forEach(function (d) {
            d.open = false;
          });
        });
      }
      function subtreeMatches(d, q) {
        if (!q) {
          return true;
        }
        return (d.textContent || '').toLowerCase().indexOf(q) !== -1;
      }
      function applyTreeFilter(q) {
        const query = (q || '').trim().toLowerCase();
        allTreeDetails().forEach(function (d) {
          if (!query) {
            d.classList.remove('filter-hidden');
            return;
          }
          d.classList.toggle('filter-hidden', !subtreeMatches(d, query));
        });
      }
      if (filterInput) {
        var t = null;
        filterInput.addEventListener('input', function () {
          if (t) {
            clearTimeout(t);
          }
          var v = filterInput.value;
          t = setTimeout(function () {
            applyTreeFilter(v);
            t = null;
          }, 160);
        });
      }
    })();
  </script>
</body>
</html>`;
}

class PtDocument {
  /** @param {vscode.Uri} uri */
  constructor(uri) {
    this.uri = uri;
  }
  dispose() {}
}

class PtEditorProvider {
  /** @param {vscode.ExtensionContext} context */
  constructor(context) {
    this.context = context;
  }

  /** @returns {string} */
  getScriptPath() {
    return path.join(
      this.context.extensionPath,
      "python",
      "inspect_checkpoint.py",
    );
  }

  /**
   * @param {vscode.Uri} uri
   * @param {vscode.WebviewPanel} webviewPanel
   */
  async refreshWebview(uri, webviewPanel) {
    const scriptPath = this.getScriptPath();
    const fsPath = uri.fsPath;
    const resource = vscode.workspace.getWorkspaceFolder(uri)?.uri;
    const pythonPath = await resolvePythonExecutable(resource);
    const unsafe = !!vscode.workspace
      .getConfiguration("torchCheckpointInspector")
      .get("allowUnsafeLoad");
    const fileName = path.basename(fsPath);

    let result;
    try {
      result = await runInspector(pythonPath, scriptPath, fsPath, unsafe);
    } catch (err) {
      result = {
        code: null,
        json: null,
        stderr: String(err),
        parseError: "Failed to spawn Python",
        rawOut: "",
      };
    }

    const title = `${fileName} — Torch checkpoint inspector`;
    webviewPanel.webview.html = buildWebviewHtml(
      webviewPanel.webview,
      title,
      fsPath,
      pythonPath,
      unsafe,
      result,
    );
  }

  /**
   * @param {vscode.Uri} uri
   * @param {vscode.CustomDocumentOpenContext} _openContext
   * @param {vscode.CancellationToken} _token
   * @returns {Promise<PtDocument>}
   */
  async openCustomDocument(uri, _openContext, _token) {
    return new PtDocument(uri);
  }

  /**
   * @param {PtDocument} document
   * @param {vscode.WebviewPanel} webviewPanel
   * @param {vscode.CancellationToken} _token
   */
  async resolveCustomEditor(document, webviewPanel, _token) {
    webviewPanel.webview.options = {
      enableScripts: true,
      localResourceRoots: [],
    };

    const sub = webviewPanel.webview.onDidReceiveMessage((msg) => {
      if (msg && msg.type === "refresh") {
        void this.refreshWebview(document.uri, webviewPanel);
      }
    });

    webviewPanel.onDidDispose(() => sub.dispose());

    await this.refreshWebview(document.uri, webviewPanel);
  }
}

/** @param {vscode.Uri | undefined} uri */
function isPtOrPth(uri) {
  if (!uri || uri.scheme !== "file") {
    return false;
  }
  const ext = path.extname(uri.fsPath).toLowerCase();
  return ext === ".pt" || ext === ".pth";
}

/** @param {vscode.ExtensionContext} context */
function activate(context) {
  const provider = new PtEditorProvider(context);
  context.subscriptions.push(
    vscode.window.registerCustomEditorProvider(
      "gelbhart.torchCheckpointInspector",
      provider,
      {
        webviewOptions: { retainContextWhenHidden: true },
        supportsMultipleEditorsPerDocument: false,
      },
    ),
    vscode.commands.registerCommand(
      "torchCheckpointInspector.openWithViewer",
      async (clicked) => {
        /** @type {vscode.Uri | undefined} */
        let uri;
        if (clicked instanceof vscode.Uri) {
          uri = clicked;
        } else {
          const tab = vscode.window.tabGroups.activeTabGroup.activeTab;
          const input = tab?.input;
          if (input instanceof vscode.TabInputText) {
            uri = input.uri;
          } else if (input instanceof vscode.TabInputCustom) {
            uri = input.uri;
          }
          if (!uri) {
            const ed = vscode.window.activeTextEditor;
            uri = ed?.document.uri;
          }
        }
        if (!isPtOrPth(uri)) {
          vscode.window.showWarningMessage(
            "Select or open a .pt or .pth file, then run this command again.",
          );
          return;
        }
        await vscode.commands.executeCommand(
          "vscode.openWith",
          uri,
          "gelbhart.torchCheckpointInspector",
          vscode.ViewColumn.Active,
        );
      },
    ),
  );
}

function deactivate() {}

module.exports = { activate, deactivate };
