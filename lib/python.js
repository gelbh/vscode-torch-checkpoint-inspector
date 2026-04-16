const vscode = require("vscode");
const { spawn } = require("child_process");

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
 * @param {string[]} scriptArgs arguments after scriptPath (e.g. file path, flags)
 * @returns {Promise<{ code: number | null, json: any, stderr: string, parseError?: string, rawOut?: string }>}
 */
function runPythonJson(pythonPath, scriptPath, scriptArgs) {
  const args = [scriptPath, ...scriptArgs];
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

/**
 * @param {string} pythonPath
 * @param {string} scriptPath
 * @param {string} filePath
 * @param {boolean} unsafe
 */
function runInspector(pythonPath, scriptPath, filePath, unsafe) {
  const scriptArgs = [filePath];
  if (unsafe) {
    scriptArgs.push("--unsafe");
  }
  return runPythonJson(pythonPath, scriptPath, scriptArgs);
}

/**
 * @param {string} pythonPath
 * @param {string} scriptPath
 */
function runTorchProbe(pythonPath, scriptPath) {
  return runPythonJson(pythonPath, scriptPath, ["--probe-torch"]);
}

module.exports = {
  getMsPythonInterpreterPath,
  resolvePythonExecutable,
  runPythonJson,
  runInspector,
  runTorchProbe,
};
