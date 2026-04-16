const {
  escapeHtml,
  dtypeByteSize,
  humanizeInt,
  humanizeBytes,
} = require("./format");

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

module.exports = { renderOverviewSection };
