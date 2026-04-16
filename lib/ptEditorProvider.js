const vscode = require("vscode");
const path = require("path");
const {
  resolvePythonExecutable,
  runInspector,
  runTorchProbe,
} = require("./python");
const { buildWebviewHtml } = require("./webviewHtml");

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
   * @param {any} msg
   */
  async handleWebviewMessage(uri, webviewPanel, msg) {
    if (!msg || typeof msg.type !== "string") {
      return;
    }
    if (msg.type === "refresh") {
      await this.refreshWebview(uri, webviewPanel);
      return;
    }
    if (msg.type === "openExtensionSettings") {
      await vscode.commands.executeCommand(
        "workbench.action.openSettings",
        "torchCheckpointInspector.pythonPath",
      );
      return;
    }
    if (msg.type === "pickPythonExecutable") {
      const picked = await vscode.window.showOpenDialog({
        canSelectMany: false,
        openLabel: "Use this Python",
        title: "Select Python interpreter",
      });
      const fileUri = picked && picked[0];
      if (!fileUri || fileUri.scheme !== "file") {
        return;
      }
      const chosen = fileUri.fsPath;
      const folder = vscode.workspace.getWorkspaceFolder(uri);
      const target = folder
        ? vscode.ConfigurationTarget.Workspace
        : vscode.ConfigurationTarget.Global;
      await vscode.workspace
        .getConfiguration("torchCheckpointInspector")
        .update("pythonPath", chosen, target);
      await this.refreshWebview(uri, webviewPanel);
      return;
    }
    if (msg.type === "openPythonInterpreterPicker") {
      const pyExt = vscode.extensions.getExtension("ms-python.python");
      if (!pyExt) {
        vscode.window.showWarningMessage(
          "Install the Python extension (ms-python.python) to use the interpreter picker.",
        );
        return;
      }
      if (!pyExt.isActive) {
        await pyExt.activate();
      }
      await vscode.commands.executeCommand("python.setInterpreter");
      vscode.window.showInformationMessage(
        "After selecting an interpreter, click Refresh in the checkpoint viewer (or reopen the file) if you use the extension setting for a custom Python path.",
      );
    }
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
    const cfg = vscode.workspace.getConfiguration("torchCheckpointInspector");
    const unsafe = !!cfg.get("allowUnsafeLoad");
    const probeEnabled = !!cfg.get("probeTorchBeforeInspect");
    const fileName = path.basename(fsPath);

    let result;
    try {
      if (probeEnabled) {
        /** @type {Awaited<ReturnType<typeof runTorchProbe>> | null} */
        let probe = null;
        try {
          probe = await runTorchProbe(pythonPath, scriptPath);
        } catch {
          probe = null;
        }
        if (
          probe &&
          probe.json?.ok === false &&
          probe.json.error_code === "NO_TORCH"
        ) {
          result = probe;
        } else {
          result = await runInspector(pythonPath, scriptPath, fsPath, unsafe);
        }
      } else {
        result = await runInspector(pythonPath, scriptPath, fsPath, unsafe);
      }
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
    const hasMsPythonExt = !!vscode.extensions.getExtension("ms-python.python");
    webviewPanel.webview.html = buildWebviewHtml(
      webviewPanel.webview,
      title,
      fsPath,
      pythonPath,
      unsafe,
      result,
      { hasMsPythonExt },
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
      void this.handleWebviewMessage(document.uri, webviewPanel, msg);
    });

    webviewPanel.onDidDispose(() => sub.dispose());

    await this.refreshWebview(document.uri, webviewPanel);
  }
}

module.exports = { PtDocument, PtEditorProvider };
