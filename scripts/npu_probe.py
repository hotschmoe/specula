"""Phase 5 step 1 - NPU environment probe.

Reports:
  (1) onnxruntime + QNN EP visibility
  (2) qai-hub device catalogue + our target device presence

Run via:
    .venv\\Scripts\\python.exe scripts\\npu_probe.py

No arguments, exits 0 on full success, 1 on partial, 2 on hard failure.
Writes a delta to results/npu_env_snapshot.txt.
"""

import sys
import traceback
from pathlib import Path

SNAPSHOT = Path(__file__).resolve().parent.parent / "results" / "npu_env_snapshot.txt"


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def probe_ort() -> dict:
    """Report ORT version, available providers, and QNN EP shared libs."""
    section("5.3 onnxruntime + QNN EP")
    out: dict = {}
    import onnxruntime as ort

    out["ort_version"] = ort.__version__
    out["build_info"] = ort.get_build_info() if hasattr(ort, "get_build_info") else "(n/a)"
    out["providers"] = ort.get_available_providers()
    out["device"] = ort.get_device()

    print(f"onnxruntime version  : {out['ort_version']}")
    print(f"onnxruntime device   : {out['device']}")
    print(f"available providers  : {out['providers']}")

    ort_pkg_dir = Path(ort.__file__).parent
    out["ort_pkg_dir"] = str(ort_pkg_dir)
    print(f"package dir          : {ort_pkg_dir}")

    qnn_dlls = sorted(ort_pkg_dir.rglob("QnnHtp*.dll")) + sorted(ort_pkg_dir.rglob("onnxruntime_providers_qnn*.dll"))
    out["qnn_dlls"] = [str(p.relative_to(ort_pkg_dir)) for p in qnn_dlls]
    print(f"qnn-related DLLs in package ({len(qnn_dlls)} found):")
    for p in qnn_dlls:
        print(f"    {p.relative_to(ort_pkg_dir)}")

    out["qnn_ep_visible"] = "QNNExecutionProvider" in out["providers"]
    return out


def probe_qai_hub() -> dict:
    """Report qai-hub version and confirm target device is in the catalogue."""
    section("5.4 qai-hub")
    out: dict = {}
    import qai_hub

    out["qai_hub_version"] = qai_hub.__version__
    print(f"qai-hub version      : {out['qai_hub_version']}")

    try:
        devices = qai_hub.get_devices()
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["auth_state"] = "unknown - get_devices() failed"
        print(f"get_devices() ERROR  : {out['error']}")
        print("  (likely QAI_HUB_API_TOKEN unset or qai-hub not configured)")
        return out

    out["device_count"] = len(devices)
    out["auth_state"] = "ok"
    print(f"device catalogue size: {len(devices)}")

    target_substrs = ["X2 Elite", "X2E", "Snapdragon X2"]
    matches = [d for d in devices if any(s.lower() in d.name.lower() for s in target_substrs)]
    out["x2_matches"] = [
        {"name": d.name, "os": getattr(d, "os", None), "attributes": list(getattr(d, "attributes", []))}
        for d in matches
    ]
    print(f"X2-family devices    : {len(matches)}")
    for d in matches:
        attrs = ",".join(getattr(d, "attributes", []) or [])
        print(f"    name='{d.name}'  os='{getattr(d, 'os', 'n/a')}'  attrs=[{attrs}]")

    exact_target = "Snapdragon X2 Elite CRD"
    out["exact_target_present"] = any(d.name == exact_target for d in devices)
    print(f"exact target '{exact_target}' present: {out['exact_target_present']}")
    return out


def write_snapshot_delta(ort_info: dict, hub_info: dict) -> None:
    section("writing snapshot delta")
    banner = "\n\n## 5.3 + 5.4 probe results (appended by scripts/npu_probe.py)\n"
    lines = [banner]
    lines.append("### 5.3 ORT-QNN\n")
    lines.append(f"- onnxruntime version : {ort_info.get('ort_version')}")
    lines.append(f"- available providers : {ort_info.get('providers')}")
    lines.append(f"- QNN EP visible      : {ort_info.get('qnn_ep_visible')}")
    lines.append(f"- package dir         : {ort_info.get('ort_pkg_dir')}")
    lines.append(f"- bundled QNN DLLs    : {len(ort_info.get('qnn_dlls') or [])} files")
    for p in ort_info.get("qnn_dlls") or []:
        lines.append(f"    {p}")
    lines.append("")
    lines.append("### 5.4 qai-hub\n")
    lines.append(f"- qai-hub version     : {hub_info.get('qai_hub_version')}")
    lines.append(f"- auth state          : {hub_info.get('auth_state')}")
    if "error" in hub_info:
        lines.append(f"- error               : {hub_info['error']}")
    else:
        lines.append(f"- device catalogue    : {hub_info.get('device_count')} devices")
        lines.append(f"- X2-family matches   : {len(hub_info.get('x2_matches') or [])}")
        for m in hub_info.get("x2_matches") or []:
            lines.append(f"    {m['name']}  os={m['os']}  attrs={m['attributes']}")
        lines.append(f"- 'Snapdragon X2 Elite CRD' present: {hub_info.get('exact_target_present')}")
    lines.append("")
    SNAPSHOT.parent.mkdir(exist_ok=True)
    with SNAPSHOT.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"appended to {SNAPSHOT}")


def main() -> int:
    try:
        ort_info = probe_ort()
    except Exception:  # noqa: BLE001
        section("5.3 FAILED")
        traceback.print_exc()
        return 2

    try:
        hub_info = probe_qai_hub()
    except Exception:  # noqa: BLE001
        section("5.4 FAILED")
        traceback.print_exc()
        hub_info = {"error": "exception during probe"}

    write_snapshot_delta(ort_info, hub_info)

    status = "ok"
    if not ort_info.get("qnn_ep_visible"):
        status = "partial: QNN EP not in available providers"
    if hub_info.get("error"):
        status = f"partial: qai-hub error ({hub_info['error']})"

    section(f"STATUS: {status}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
