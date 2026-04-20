"""
Patch llama.cpp's KleidiAI SMCU detection so Windows/Oryon-v2 doesn't
fault on default-dispatch to the SME Q8_0 kernel.

Background (see docs/SME_investigation.md for the full story):

    detect_num_smcus() in ggml-cpu/kleidiai/kleidiai.cpp has a
    #else return 1; branch that fires on non-Linux, non-Apple
    platforms. On Windows that causes KleidiAI to conclude one
    Streaming Mode Compute Unit is present, OR CPU_FEATURE_SME
    into ctx.features, and dispatch the SME Q8_0 kernel. That
    kernel faults on the Qualcomm Oryon v2 in the Snapdragon
    X2 Elite Extreme (base SME works per scripts/sme_probe/,
    but the specific instructions KleidiAI's kernel uses do not)
    with STATUS_ILLEGAL_INSTRUCTION at default env settings.

    This patch replaces the fallback `return 1;` with `return 0;`
    so Windows builds pick non-SME kernels by default. The user
    can still force SME on via `GGML_KLEIDIAI_SME=<n>` for
    experiments; with that env set the old faulting path is
    reachable again.

Usage:
    python patch_kleidiai_detect.py <path-to-llama.cpp-root>

    e.g. python patch_kleidiai_detect.py C:\\...\\specula\\llama.cpp

Idempotent: re-running is a no-op on already-patched files.
"""
import os
import sys

TARGET = "ggml/src/ggml-cpu/kleidiai/kleidiai.cpp"

MARKER = "// specula: return 0 on Windows fallthrough"

ORIGINAL = """    return 1;

#else
    return 1;
#endif
}
"""

REPLACEMENT = """    return 1;

#else
    // specula: return 0 on Windows fallthrough (see
    // docs/SME_investigation.md). Upstream returns 1 here, which
    // makes KleidiAI dispatch its SME kernel by default on Windows
    // and fault on Oryon v2 X2 Elite Extreme. Base SME is usable
    // on this CPU+OS (see scripts/sme_probe/) — the fault is in a
    // specific SME instruction KleidiAI's kernel uses. Forcing this
    // to 0 falls back to the SVE/I8MM/DOTPROD path, which is
    // correct. Users can still opt into SME via GGML_KLEIDIAI_SME=N.
    return 0;
#endif
}
"""


def main(root: str) -> int:
    path = os.path.join(root, TARGET)
    if not os.path.isfile(path):
        print(f"error: expected file not found: {path}", file=sys.stderr)
        return 2

    with open(path, "r", encoding="utf-8") as fh:
        data = fh.read()

    if MARKER in data:
        print("patch_kleidiai_detect: already patched, skipping")
        return 0

    if ORIGINAL not in data:
        print(
            "patch_kleidiai_detect: expected source pattern not found — "
            "KleidiAI upstream may have moved the detect_num_smcus fallthrough. "
            "Patch not applied; re-check docs/SME_investigation.md.",
            file=sys.stderr,
        )
        return 3

    data = data.replace(ORIGINAL, REPLACEMENT, 1)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(data)

    print(f"patch_kleidiai_detect: patched {TARGET}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "usage: python patch_kleidiai_detect.py <path-to-llama.cpp-root>",
            file=sys.stderr,
        )
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
