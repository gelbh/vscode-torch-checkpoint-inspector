/** @param {string} s */
function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
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

module.exports = {
  escapeHtml,
  dtypeByteSize,
  humanizeInt,
  humanizeBytes,
};
