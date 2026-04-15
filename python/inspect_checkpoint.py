#!/usr/bin/env python3
"""
Inspect a PyTorch checkpoint or pickled object. Prints exactly one JSON object to stdout.
Logs diagnostics to stderr; exits with non-zero on failure.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import traceback
from collections import OrderedDict
from typing import Any

MAX_DICT_KEYS = 200
MAX_LIST_ITEMS = 50
MAX_DEPTH = 20
STATS_NUMEL_CAP = 1_000_000


def _emit(obj: dict[str, Any]) -> None:
    print(json.dumps(obj, default=str, ensure_ascii=False))
    sys.stdout.flush()


def _tensor_summary(t: Any, depth: int) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {
        "kind": "Tensor",
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "numel": int(t.numel()),
    }
    if t.numel() == 0:
        return out
    if getattr(t.device, "type", "") == "meta":
        return out
    if t.numel() > STATS_NUMEL_CAP:
        out["stats_skipped"] = f"numel>{STATS_NUMEL_CAP}"
        return out
    try:
        tc = t.detach().cpu()
        if tc.dtype in (torch.bool,):
            out["min"] = bool(tc.min().item())
            out["max"] = bool(tc.max().item())
        elif tc.is_floating_point() or tc.is_complex():
            out["min"] = float(tc.real.min().item()) if not tc.is_complex() else float(tc.abs().min().item())
            out["max"] = float(tc.real.max().item()) if not tc.is_complex() else float(tc.abs().max().item())
            if tc.is_floating_point() and not tc.is_complex():
                out["mean"] = float(tc.float().mean().item())
        else:
            out["min"] = int(tc.min().item())
            out["max"] = int(tc.max().item())
    except Exception as exc:  # noqa: BLE001
        out["stats_error"] = str(exc)
    return out


def summarize(obj: Any, depth: int = 0) -> Any:
    if depth > MAX_DEPTH:
        return {"kind": "Truncated", "reason": "max_depth", "repr": repr(obj)[:400]}

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return {"kind": "Error", "message": "torch is not installed in this interpreter"}

    if isinstance(obj, torch.Tensor):
        return _tensor_summary(obj, depth)

    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return {
                "kind": "ndarray",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "numel": int(obj.size),
            }
    except ImportError:
        pass

    try:
        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            cols = list(obj.columns)
            dtypes = {str(c): str(dt) for c, dt in list(obj.dtypes.items())[:40]}
            return {
                "kind": "DataFrame",
                "shape": [int(obj.shape[0]), int(obj.shape[1])],
                "columns": cols[:80],
                "truncated_columns": len(cols) > 80,
                "dtypes": dtypes,
                "truncated_dtypes": len(obj.columns) > 40,
            }
        if isinstance(obj, pd.Series):
            return {
                "kind": "Series",
                "name": obj.name,
                "length": len(obj),
                "dtype": str(obj.dtype),
            }
        if isinstance(obj, pd.Index):
            return {
                "kind": "Index",
                "length": len(obj),
                "dtype": str(obj.dtype),
                "head": str(obj[:20]),
            }
    except ImportError:
        pass

    if isinstance(obj, nn.Module):
        try:
            sd = obj.state_dict()
        except Exception as exc:  # noqa: BLE001
            return {
                "kind": "Module",
                "type": repr(type(obj)),
                "state_dict_error": str(exc),
            }
        return {
            "kind": "Module",
            "type": repr(type(obj)),
            "state_dict": summarize(sd, depth + 1),
        }

    if isinstance(obj, (dict, OrderedDict)):
        keys = list(obj.keys())
        total = len(keys)
        truncated = total > MAX_DICT_KEYS
        keys = keys[:MAX_DICT_KEYS]
        children: dict[str, Any] = {}
        for k in keys:
            key_str = str(k)
            children[key_str] = summarize(obj[k], depth + 1)
        return {
            "kind": "Dict",
            "total_keys": total,
            "truncated_keys": truncated,
            "children": children,
        }

    if isinstance(obj, (list, tuple)):
        n = len(obj)
        items = [summarize(obj[i], depth + 1) for i in range(min(n, MAX_LIST_ITEMS))]
        return {
            "kind": "List",
            "length": n,
            "truncated": n > MAX_LIST_ITEMS,
            "items": items,
        }

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if isinstance(obj, bytes):
        return {"kind": "Bytes", "len": len(obj)}

    return {"kind": "Other", "type": repr(type(obj)), "repr": repr(obj)[:300]}


# Only auto-allowlist globals under these prefixes (from PyTorch WeightsUnpickler errors).
_WEIGHTS_ONLY_AUTO_GLOBAL_PREFIXES: tuple[str, ...] = ("pandas.", "numpy.")


def _optional_weights_only_seed_types() -> list[Any]:
    """
    Seed types for torch.serialization.safe_globals before iterative expansion.
    """
    types_list: list[Any] = []
    try:
        import numpy as np

        types_list.append(np.ndarray)
        types_list.append(np.dtype)
    except ImportError:
        pass
    try:
        import pandas as pd

        types_list.append(pd.DataFrame)
        types_list.append(pd.Series)
        types_list.append(pd.Index)
    except ImportError:
        pass
    return types_list


def _parse_unsupported_global_qualname(error_message: str) -> str | None:
    m = re.search(
        r"Unsupported global: GLOBAL (\S+) was not an allowed global",
        error_message,
    )
    return m.group(1) if m else None


def _resolve_dotted_global(qualname: str) -> Any | None:
    """
    Import qualname like 'pandas.core.internals.managers.BlockManager'.
    Tries longest module path first.
    """
    parts = qualname.split(".")
    if len(parts) < 2:
        return None
    for i in range(len(parts) - 1, 0, -1):
        modname, attr_parts = ".".join(parts[:i]), parts[i:]
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        try:
            obj: Any = mod
            for name in attr_parts:
                obj = getattr(obj, name)
        except AttributeError:
            continue
        return obj
    return None


def _maybe_allowlist_from_unpickler_error(error_message: str) -> Any | None:
    qual = _parse_unsupported_global_qualname(error_message)
    if qual is None or not qual.startswith(_WEIGHTS_ONLY_AUTO_GLOBAL_PREFIXES):
        return None
    return _resolve_dotted_global(qual)


def load_checkpoint(path: str, unsafe: bool) -> tuple[Any, str]:
    import torch

    errors: list[str] = []

    def load_meta(weights_only: bool) -> Any:
        return torch.load(path, map_location="meta", weights_only=weights_only)

    # --- weights_only=True: plain first ---
    try:
        return load_meta(True), "safe_meta"
    except Exception as e:  # noqa: BLE001
        errors.append(f"weights_only=True (plain): {e!s}")

    safe_ctx = getattr(torch.serialization, "safe_globals", None)
    if safe_ctx is None:
        errors.append("weights_only=True + safe_globals: not available (upgrade PyTorch)")
    else:
        # Iteratively extend allowlist for pandas/numpy internals (e.g. BlockManager).
        extra: list[Any] = []
        seen_ids: set[int] = set()

        def add_extra(obj: Any) -> None:
            oid = id(obj)
            if oid in seen_ids:
                return
            seen_ids.add(oid)
            extra.append(obj)

        for seed in _optional_weights_only_seed_types():
            add_extra(seed)

        if not extra:
            errors.append(
                "weights_only=True + safe_globals: install pandas/numpy in this interpreter for DataFrame checkpoints"
            )
        else:
            max_rounds = 48
            last_msg = ""
            for round_i in range(max_rounds):
                try:
                    with safe_ctx(extra):
                        return load_meta(True), "safe_meta_allowlist"
                except Exception as e:  # noqa: BLE001
                    last_msg = str(e)
                    nxt = _maybe_allowlist_from_unpickler_error(last_msg)
                    if nxt is None:
                        errors.append(
                            f"weights_only=True + safe_globals (stopped after {len(extra)} types, round {round_i}): {last_msg}"
                        )
                        break
                    if id(nxt) in seen_ids:
                        errors.append(
                            f"weights_only=True + safe_globals: repeated unresolved global after {len(extra)} types: {last_msg}"
                        )
                        break
                    add_extra(nxt)
            else:
                errors.append(
                    f"weights_only=True + safe_globals: exceeded {max_rounds} expansion rounds (last: {last_msg})"
                )

    # --- weights_only=False (full unpickling), only when user opted in ---
    if unsafe:
        try:
            return load_meta(False), "unsafe_meta"
        except Exception as e:  # noqa: BLE001
            errors.append(f"weights_only=False: {e!s}")
        raise RuntimeError("torch.load failed:\n" + "\n".join(errors))

    raise RuntimeError(
        "torch.load failed with weights_only=True (including pandas/numpy safe_globals expansion). "
        "If this is a trusted file, enable torchCheckpointInspector.allowUnsafeLoad.\n"
        + "\n".join(errors)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a .pt / .pth file as JSON.")
    parser.add_argument("path", help="Checkpoint file path")
    parser.add_argument(
        "--unsafe",
        action="store_true",
        help="Allow weights_only=False after safe load fails (trusted files only).",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        _emit(
            {
                "ok": False,
                "error": "PyTorch is not installed in this Python environment.",
                "hint": "Install torch in the interpreter configured for Torch checkpoint inspector.",
            }
        )
        return 1

    path = args.path
    try:
        obj, load_mode = load_checkpoint(path, args.unsafe)
        root = summarize(obj, 0)
        try:
            file_size_bytes = int(os.path.getsize(path))
        except OSError:
            file_size_bytes = None
        _emit(
            {
                "ok": True,
                "path": path,
                "file_name": os.path.basename(path),
                "file_size_bytes": file_size_bytes,
                "torch_version": torch.__version__,
                "load_mode": load_mode,
                "unsafe_requested": bool(args.unsafe),
                "root": root,
            }
        )
        return 0
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        _emit(
            {
                "ok": False,
                "error": str(sys.exc_info()[1]),
                "traceback": tb,
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
