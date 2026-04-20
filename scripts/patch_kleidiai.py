"""
Patch KleidiAI's .S files so clang-on-Windows-ARM64 can assemble them.

Background:
    On Windows, clang defines both _MSC_VER and __clang__. KleidiAI's armasm
    branch assumes _MSC_VER means "armasm64 will assemble this", which is
    wrong for clang. We relax the outer guard to
        #if defined(_MSC_VER) && !defined(__clang__)
    so clang-on-Windows falls through to the GAS branch, and add an inner
    #if defined(_WIN32) sub-branch that omits ELF-only `.type`/`.size`
    directives (those are illegal in COFF).

Usage:
    python patch_kleidiai.py <path-to-kai-src>

    e.g. python patch_kleidiai.py C:\\...\\llama.cpp\\build-cpu-kleidiai\\_deps\\kleidiai_download-src\\kai

Idempotent: re-running is a no-op on already-patched files.
"""
import os
import re
import sys


GUARD_PATTERN = re.compile(r"^#if defined\(_MSC_VER\)\s*$", re.MULTILINE)

SWITCH_PATTERN = re.compile(
    r"(\s*#if defined\(__APPLE__\)\n"
    r"(?:[^\n]*\n)*?"
    r"\s*#else\n)"
    r"((?:[^\n]*\n)*?)"
    r"(\s*#endif\n)",
    re.MULTILINE,
)


def transform_switch(match):
    apple_start = match.group(1)
    linux_body = match.group(2)
    endif = match.group(3)

    win_lines = []
    for line in linux_body.splitlines(keepends=True):
        if ".type name, %function" in line:
            line = re.sub(r"(#define\s+KAI_ASM_FUNCTION_TYPE\(name\)).*", r"\1", line)
        elif ".size name, .-name" in line:
            line = re.sub(r"(#define\s+KAI_ASM_FUNCTION_END\(name\)).*", r"\1", line)
        win_lines.append(line)
    return (
        apple_start
        + "    #if defined(_WIN32)\n"
        + "".join(win_lines)
        + "    #else\n"
        + linux_body
        + "    #endif\n"
        + endif
    )


def main(root: str) -> int:
    if not os.path.isdir(root):
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2

    patched = 0
    skipped = 0
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not f.endswith(".S"):
                continue
            p = os.path.join(dirpath, f)
            with open(p, "r", encoding="utf-8") as fh:
                data = fh.read()
            orig = data
            data = GUARD_PATTERN.sub(
                "#if defined(_MSC_VER) && !defined(__clang__)", data, count=1
            )
            if "#if defined(_WIN32)" not in data:
                data, _ = SWITCH_PATTERN.subn(transform_switch, data, count=1)
            if data != orig:
                with open(p, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(data)
                patched += 1
            else:
                skipped += 1

    print(f"patched={patched} skipped={skipped}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python patch_kleidiai.py <path-to-kai-src>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
