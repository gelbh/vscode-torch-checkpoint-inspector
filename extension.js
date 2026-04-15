const vscode = require('vscode');
const path = require('path');
const { spawn } = require('child_process');

/** @param {string} s */
function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * @param {vscode.Uri | undefined} resource
 * @returns {Promise<string | undefined>}
 */
async function getMsPythonInterpreterPath(resource) {
  try {
    const extension = vscode.extensions.getExtension('ms-python.python');
    if (!extension) {
      return undefined;
    }
    /** @type {any} */
    const api = extension.isActive ? extension.exports : await extension.activate();
    const details = api?.settings?.getExecutionDetails?.(resource);
    if (!details) {
      return undefined;
    }
    if (Array.isArray(details.execCommand) && details.execCommand.length > 0) {
      return details.execCommand[0];
    }
    if (typeof details.pythonPath === 'string' && details.pythonPath.length > 0) {
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
  const cfg = vscode.workspace.getConfiguration('torchCheckpointInspector');
  const override = cfg.get('pythonPath', '');
  if (typeof override === 'string' && override.trim().length > 0) {
    return override.trim();
  }
  const fromPythonExt = await getMsPythonInterpreterPath(resource);
  if (fromPythonExt) {
    return fromPythonExt;
  }
  return process.platform === 'win32' ? 'python' : 'python3';
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
    args.push('--unsafe');
  }
  return new Promise((resolve, reject) => {
    const chunks = [];
    const errChunks = [];
    const proc = spawn(pythonPath, args, {
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    proc.stdout.on('data', (d) => chunks.push(d));
    proc.stderr.on('data', (d) => errChunks.push(d));
    proc.on('error', (err) => reject(err));
    proc.on('close', (code) => {
      const out = Buffer.concat(chunks).toString('utf8').trim();
      const stderr = Buffer.concat(errChunks).toString('utf8');
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

/** @param {any} node */
function renderTreeNode(node) {
  if (node === null || node === undefined) {
    return `<span class="primitive">${escapeHtml(String(node))}</span>`;
  }
  if (typeof node !== 'object') {
    return `<span class="primitive">${escapeHtml(JSON.stringify(node))}</span>`;
  }

  switch (node.kind) {
    case 'Tensor': {
      const parts = [
        `shape=[${(node.shape || []).join(', ')}]`,
        `dtype=${node.dtype}`,
        `device=${node.device}`,
        `numel=${node.numel}`,
      ];
      if (node.stats_skipped) {
        parts.push(String(node.stats_skipped));
      }
      if (node.stats_error) {
        parts.push(`stats_error=${node.stats_error}`);
      }
      if (node.min !== undefined) {
        parts.push(`min=${node.min}`);
      }
      if (node.max !== undefined) {
        parts.push(`max=${node.max}`);
      }
      if (node.mean !== undefined) {
        parts.push(`mean=${node.mean}`);
      }
      return `<div class="tensor">${escapeHtml(parts.join(' · '))}</div>`;
    }
    case 'ndarray': {
      const parts = [
        `shape=[${(node.shape || []).join(', ')}]`,
        `dtype=${node.dtype}`,
        `size=${node.numel}`,
      ];
      return `<div class="tensor"><span class="label">NumPy</span> ${escapeHtml(parts.join(' · '))}</div>`;
    }
    case 'DataFrame': {
      const shape = (node.shape || []).join(' × ');
      const cols = (node.columns || []).join(', ');
      const trunc = node.truncated_columns ? ' …' : '';
      const dt = node.dtypes ? escapeHtml(JSON.stringify(node.dtypes)) : '';
      return `<div class="pandas"><span class="label">DataFrame</span> ${escapeHtml(shape)} columns:${trunc}<br><code class="cols">${escapeHtml(cols)}</code>${node.truncated_dtypes ? ' <span class="badge">dtypes truncated</span>' : ''}<br><small>${dt}</small></div>`;
    }
    case 'Series': {
      return `<div class="pandas"><span class="label">Series</span> name=${escapeHtml(String(node.name))} len=${node.length} dtype=${escapeHtml(String(node.dtype))}</div>`;
    }
    case 'Index': {
      return `<div class="pandas"><span class="label">Index</span> len=${node.length} dtype=${escapeHtml(String(node.dtype))} <code>${escapeHtml(String(node.head || ''))}</code></div>`;
    }
    case 'Dict': {
      const keys = Object.keys(node.children || {});
      const keysTrunc = node.truncated_keys
        ? ` <span class="badge">showing ${keys.length} of ${node.total_keys} keys</span>`
        : '';
      const children = node.children || {};
      let inner = '';
      for (const k of keys) {
        inner += `<details class="node" open>
  <summary><code>${escapeHtml(k)}</code></summary>
  <div class="indent">${renderTreeNode(children[k])}</div>
</details>`;
      }
      return `<div class="dict"><span class="label">Dict</span>${keysTrunc}${inner || ' <em>empty</em>'}</div>`;
    }
    case 'List': {
      const items = node.items || [];
      const trunc = node.truncated
        ? ` <span class="badge">showing ${items.length} of ${node.length}</span>`
        : '';
      let inner = '';
      items.forEach((item, i) => {
        inner += `<details class="node" open>
  <summary>[${i}]</summary>
  <div class="indent">${renderTreeNode(item)}</div>
</details>`;
      });
      return `<div class="list"><span class="label">List</span> (len=${node.length})${trunc}${inner}</div>`;
    }
    case 'Module': {
      const head = `<div class="module"><span class="label">Module</span> <code>${escapeHtml(node.type || '')}</code></div>`;
      if (node.state_dict_error) {
        return `${head}<div class="warn">${escapeHtml(node.state_dict_error)}</div>`;
      }
      if (node.state_dict !== undefined) {
        return `${head}<div class="indent">${renderTreeNode(node.state_dict)}</div>`;
      }
      return head;
    }
    case 'Truncated':
      return `<div class="warn">${escapeHtml(node.reason || '')}: ${escapeHtml(node.repr || '')}</div>`;
    case 'Bytes':
      return `<div><span class="label">Bytes</span> len=${node.len}</div>`;
    case 'Other':
      return `<div><span class="label">Other</span> <code>${escapeHtml(node.type || '')}</code> — <span class="repr">${escapeHtml(node.repr || '')}</span></div>`;
    case 'Error':
      return `<div class="err">${escapeHtml(node.message || '')}</div>`;
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
function buildWebviewHtml(webview, title, filePath, pythonUsed, unsafeSetting, payload) {
  const csp = [
    `default-src 'none'`,
    `style-src ${webview.cspSource} 'unsafe-inline'`,
    `script-src ${webview.cspSource} 'unsafe-inline'`,
  ].join('; ');

  let body = '';
  if (payload.parseError || !payload.json) {
    body = `<div class="err">
  <h2>Could not parse inspector output</h2>
  <p><strong>Parse error:</strong> ${escapeHtml(payload.parseError || '')}</p>
  <p><strong>Exit code:</strong> ${payload.code ?? 'n/a'}</p>
  <h3>stdout</h3>
  <pre>${escapeHtml((payload.rawOut || '').slice(0, 8000))}</pre>
  <h3>stderr</h3>
  <pre>${escapeHtml((payload.stderr || '').slice(0, 8000))}</pre>
</div>`;
  } else if (!payload.json.ok) {
    body = `<div class="err">
  <h2>Inspection failed</h2>
  <p>${escapeHtml(payload.json.error || 'Unknown error')}</p>
  <pre>${escapeHtml((payload.json.traceback || payload.stderr || '').slice(0, 12000))}</pre>
</div>`;
  } else {
    const j = payload.json;
    const unsafeBanner =
      j.load_mode === 'unsafe_meta'
        ? `<div class="warn unsafe"><strong>Unsafe load path</strong> — file was loaded with <code>weights_only=False</code>. Only use this for trusted checkpoints.</div>`
        : '';
    body = `${unsafeBanner}
<div class="meta">
  <div><strong>File</strong> <code>${escapeHtml(filePath)}</code></div>
  <div><strong>Interpreter</strong> <code>${escapeHtml(pythonUsed)}</code></div>
  <div><strong>PyTorch</strong> ${escapeHtml(j.torch_version || '?')} · <strong>Load</strong> ${escapeHtml(j.load_mode || '')}</div>
  <div><strong>Settings</strong> allowUnsafeLoad=${unsafeSetting ? 'true' : 'false'}</div>
</div>
<div class="toolbar">
  <button type="button" id="btn-refresh">Refresh</button>
</div>
<div class="tree">${renderTreeNode(j.root)}</div>`;
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
    }
    h2 { margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 600; }
    h3 { margin: 1rem 0 0.25rem 0; font-size: 0.9rem; }
    pre {
      margin: 0.25rem 0;
      padding: 0.5rem;
      background: var(--vscode-textCodeBlock-background);
      border: 1px solid var(--vscode-panel-border);
      overflow: auto;
      max-height: 40vh;
      font-size: 12px;
    }
    .meta { margin-bottom: 1rem; line-height: 1.6; }
    .meta code { font-size: 12px; }
    .toolbar { margin-bottom: 0.75rem; }
    button {
      background: var(--vscode-button-background);
      color: var(--vscode-button-foreground);
      border: none;
      padding: 0.35rem 0.75rem;
      cursor: pointer;
      font-family: inherit;
      font-size: inherit;
    }
    button:hover { background: var(--vscode-button-hoverBackground); }
    .tree details.node { margin: 0.15rem 0; }
    .tree summary { cursor: pointer; user-select: none; }
    .indent { margin-left: 0.75rem; border-left: 1px solid var(--vscode-panel-border); padding-left: 0.5rem; }
    .label { font-weight: 600; color: var(--vscode-descriptionForeground); margin-right: 0.35rem; }
    .tensor { font-family: var(--vscode-editor-font-family, monospace); font-size: 12px; }
    .badge { font-size: 11px; color: var(--vscode-descriptionForeground); }
    .warn { background: color-mix(in srgb, var(--vscode-inputValidation-warningBackground) 60%, var(--vscode-editor-background)); border: 1px solid var(--vscode-inputValidation-warningBorder); padding: 0.5rem 0.75rem; margin-bottom: 0.75rem; }
    .warn.unsafe { border-color: var(--vscode-inputValidation-errorBorder); }
    .err { color: var(--vscode-errorForeground); }
    .primitive { font-family: var(--vscode-editor-font-family, monospace); }
    .repr { word-break: break-all; }
  </style>
</head>
<body>
  <h2>${escapeHtml(title)}</h2>
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
    return path.join(this.context.extensionPath, 'python', 'inspect_checkpoint.py');
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
    const unsafe = !!vscode.workspace.getConfiguration('torchCheckpointInspector').get('allowUnsafeLoad');
    const fileName = path.basename(fsPath);

    let result;
    try {
      result = await runInspector(pythonPath, scriptPath, fsPath, unsafe);
    } catch (err) {
      result = {
        code: null,
        json: null,
        stderr: String(err),
        parseError: 'Failed to spawn Python',
        rawOut: '',
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
      if (msg && msg.type === 'refresh') {
        void this.refreshWebview(document.uri, webviewPanel);
      }
    });

    webviewPanel.onDidDispose(() => sub.dispose());

    await this.refreshWebview(document.uri, webviewPanel);
  }
}

/** @param {vscode.Uri | undefined} uri */
function isPtOrPth(uri) {
  if (!uri || uri.scheme !== 'file') {
    return false;
  }
  const ext = path.extname(uri.fsPath).toLowerCase();
  return ext === '.pt' || ext === '.pth';
}

/** @param {vscode.ExtensionContext} context */
function activate(context) {
  const provider = new PtEditorProvider(context);
  context.subscriptions.push(
    vscode.window.registerCustomEditorProvider('gelbhart.torchCheckpointInspector', provider, {
      webviewOptions: { retainContextWhenHidden: true },
      supportsMultipleEditorsPerDocument: false,
    }),
    vscode.commands.registerCommand('torchCheckpointInspector.openWithViewer', async (clicked) => {
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
        vscode.window.showWarningMessage('Select or open a .pt or .pth file, then run this command again.');
        return;
      }
      await vscode.commands.executeCommand(
        'vscode.openWith',
        uri,
        'gelbhart.torchCheckpointInspector',
        vscode.ViewColumn.Active,
      );
    }),
  );
}

function deactivate() {}

module.exports = { activate, deactivate };
