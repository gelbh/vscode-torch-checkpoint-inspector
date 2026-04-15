# Torch checkpoint inspector

[![Version](https://img.shields.io/badge/version-0.2.0-blue)](https://github.com/gelbhart/vscode-torch-checkpoint-inspector)
[![License](https://img.shields.io/github/license/gelbhart/vscode-torch-checkpoint-inspector)](https://github.com/gelbhart/vscode-torch-checkpoint-inspector/blob/main/LICENSE)
[![VS Code](https://img.shields.io/badge/VS%20Code-%5E1.85.0-purple)](https://code.visualstudio.com/)

Inspect `.pt` / `.pth` in a custom editor. Needs Python with **PyTorch**; **pandas** / **numpy** if checkpoints embed DataFrames. Interpreter: `torchCheckpointInspector.pythonPath`, else Python extension, else `python3`. Optional `torchCheckpointInspector.allowUnsafeLoad` (trusted files only). **F5** to run the extension.
