const vscode = require("vscode");
const path = require("path");
const { PtEditorProvider } = require("./lib/ptEditorProvider");

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
