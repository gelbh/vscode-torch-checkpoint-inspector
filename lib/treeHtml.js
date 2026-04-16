const { escapeHtml, humanizeInt } = require("./format");

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

module.exports = { renderTreeNode };
