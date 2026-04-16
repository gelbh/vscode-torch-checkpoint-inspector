const path = require("path");
const { escapeHtml } = require("./format");
const { renderOverviewSection } = require("./checkpointOverview");
const { renderTreeNode } = require("./treeHtml");

/**
 * @param {import("vscode").Webview} webview
 * @param {string} title
 * @param {string} filePath
 * @param {string} pythonUsed
 * @param {boolean} unsafeSetting
 * @param {any} payload
 * @param {{ hasMsPythonExt?: boolean }} [viewOptions]
 */
function buildWebviewHtml(
  webview,
  title,
  filePath,
  pythonUsed,
  unsafeSetting,
  payload,
  viewOptions = {},
) {
  const hasMsPythonExt = !!viewOptions.hasMsPythonExt;
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
    const j = payload.json;
    const hintBlock = j.hint
      ? `<p class="err-hint">${escapeHtml(String(j.hint))}</p>`
      : "";
    const isNoTorch = j.error_code === "NO_TORCH";
    const pyInterpreterBtn = hasMsPythonExt
      ? `<button type="button" id="btn-python-interpreter" class="secondary">Python: Select Interpreter…</button>`
      : "";
    const noTorchActions = `<div class="toolbar-row err-actions card">
  <button type="button" id="btn-refresh">Refresh</button>
  <button type="button" id="btn-open-settings" class="secondary">Extension settings</button>
  <button type="button" id="btn-pick-python" class="secondary">Choose Python executable…</button>
  ${pyInterpreterBtn}
</div>`;
    const genericActions = `<div class="toolbar-row err-actions card">
  <button type="button" id="btn-refresh">Refresh</button>
</div>`;
    const actions = isNoTorch ? noTorchActions : genericActions;
    body = `<div class="err">
  <h2>Inspection failed</h2>
  <p>${escapeHtml(j.error || "Unknown error")}</p>
  ${hintBlock}
  <div class="env-inline card"><span class="env-k">Interpreter</span> <code class="env-code">${escapeHtml(pythonUsed)}</code></div>
  ${actions}
  <pre>${escapeHtml((j.traceback || payload.stderr || "").slice(0, 12000))}</pre>
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
    .err-hint {
      color: var(--vscode-descriptionForeground);
      margin: 0.35rem 0 0.5rem 0;
      line-height: 1.45;
    }
    .env-inline {
      margin: 0.5rem 0;
      padding: 0.5rem 0.65rem;
      font-size: 12px;
    }
    .err-actions { margin: 0.5rem 0; }
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
      function wireClick(id, messageType) {
        const el = document.getElementById(id);
        if (el) {
          el.addEventListener('click', function () {
            vscode.postMessage({ type: messageType });
          });
        }
      }
      wireClick('btn-refresh', 'refresh');
      wireClick('btn-open-settings', 'openExtensionSettings');
      wireClick('btn-pick-python', 'pickPythonExecutable');
      wireClick('btn-python-interpreter', 'openPythonInterpreterPicker');
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

module.exports = { buildWebviewHtml };
