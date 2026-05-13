# 2026-05-12 overnight perf sprint — handover

**For the next session opening this repo.** Hardware handed over for
~6 hours after the 2026-05-12 sweep + landscape audit. Three tracks
below. **Hard rule: every bench gets a timeout. No unbounded runs.**

Starting state recap (see `current_status.md` session 25):
- llama.cpp at `856c3adac` (mainline HEAD as of session close).
- Four builds clean: `build-cpu`, `build-cpu-kleidiai`, `build-opencl`,
  `build-vulkan` — all stamped `856c3adac`.
- NPU side unchanged (ORT-QNN 1.24.4 ↔ QAIRT 2.42 ↔ same Qualcomm
  w4a16 bundle).
- 48 GB unified LPDDR5X @ 228 GB/s; **44 GB BIOS-allocated to GPU**
  (so even a 35B GGUF fits via `-ngl 99` on Adreno).
- Vulkan `GGML_VK_DISABLE_F16=1` is BROKEN on this HEAD
  (STATUS_ACCESS_VIOLATION). Don't use that env var anywhere.
- The git stash `stash@{0}` in `llama.cpp/` holds the obsolete
  SME-detect patch (now redundant on `856c3adac`). Drop after first
  successful cpu-kleidiai build; `patch_kleidiai_detect.py` produces
  the same end state.

---

## Watchdog discipline — read first

**Every bench command must have a hard timeout.** The stuck-Vulkan
incident on 2026-05-12 evening cost ~5 min of user attention before
manual kill. Don't repeat it.

### Concrete rules

1. **Foreground bench (PowerShell tool):** pass `timeout=N` in
   milliseconds, capped at:
   - 4B-class model: `300000` (5 min)
   - 35B-class model: `900000` (15 min)
   - Quantization / training job: `1800000` (30 min)
2. **Background bench (run_in_background):** same `timeout` cap.
   The harness kills the process at expiry. Don't pass `0` or omit.
3. **For Python subprocesses inside bench scripts:** wrap with
   `subprocess.run(..., timeout=N)` always. Existing
   `scripts/bench_qwen3_4b_all_backends.py` uses `timeout=600` per
   subprocess (10 min) — that's fine for 4B. For 35B raise to 1200
   and re-thread the wall-clock budget.
4. **Liveness check before declaring "stuck":** if `pp_time_s`
   trend is sane (e.g. 30 s in and you have first-token timing),
   keep waiting. If wall-clock is >2× the worst sane expectation
   AND no `--verbose` output for >60s, kill immediately.
5. **Backup kill switch:** if `llama-bench.exe` is sitting at >10 GB
   RAM with <1% GPU/CPU utilization for >60s, kill the process
   tree via `Stop-Process -Force -Id <pid>`. The
   "high-RAM-low-util" pattern is the unambiguous "stuck"
   signature this hardware shows when Vulkan/OpenCL load paths
   misbehave.

### Sane wall-clock budgets (use these to set timeouts)

| run | expected wall | watchdog timeout |
|---|---|---|
| Qwen3-4B llama-bench `-p 512 -n 128 -r 3` | 20-35 s | 5 min (300 s) |
| Qwen3-4B `npu_engine` sidecar bench | 25 s | 5 min |
| Qwen3-4B Genie via `bench_qwen3_4b_all_backends.py` | 150 s | 10 min |
| Qwen3.6-35B-A3B llama-bench `-p 512 -n 128 -r 1` | 120-300 s (CPU); 200-600 s (GPU) | 15 min |
| Qwen3.6-27B llama-bench (estimate, MTP-preserved) | 90-200 s (CPU) | 10 min |
| Speculative decode `--spec-draft-model` 4B target | 60-120 s | 5 min |

---

## Track A — Qwen3.6-27B with MTP

### Goal
Establish a "with-MTP vs without-MTP" TG number on this laptop. MTP
heads were released with Qwen3.6 family; the limiting factor today
is the llama.cpp runtime side (PR #22673 — was draft at session
close; **first thing to do: check if it merged tonight**).

### Step 1 — check MTP merge status (5 min, no compute)
```powershell
cd C:\Users\hotschmoe\Documents\GitHub\specula\llama.cpp
git fetch origin
git log origin/master --oneline --grep="MTP" | Select-Object -First 5
git log origin/master --oneline -- src/llama-mtp* 2>$null
# Also: is PR #22673 merged?
gh pr view 22673 -R ggml-org/llama.cpp --json state,mergedAt
```

If `state` = `MERGED`: rebuild mainline at new HEAD, proceed.
If not: switch the runtime to `origin/gg/spec-mtp-experiments`
**but only for this track** — preserve `master` checkout for tracks B
and C.

### Step 2 — fetch MTP-preserved GGUF (30 min, network)
```powershell
# 27B Q4_K_M MTP-preserved (Adreno 44 GB headroom => fits)
curl.exe -L -C - -o "models\Qwen3.6-27B-MTP-UD-Q4_K_M.gguf" `
  https://huggingface.co/havenoammo/Qwen3.6-27B-MTP-UD-GGUF/resolve/main/Qwen3.6-27B-MTP-UD-Q4_K_M.gguf
```
(Path is approximate — verify with `gh api /repos/havenoammo/Qwen3.6-27B-MTP-UD-GGUF/contents` or `huggingface-cli ls`.)

**Watchdog:** `curl.exe -C -` resumes; if no progress for >60s
abort and retry with `--connect-timeout 30 --max-time 1800`.

### Step 3 — baseline run (15 min)
- Standard non-MTP load: `llama-bench -m Qwen3.6-27B-MTP-UD-Q4_K_M.gguf -p 512 -n 128 -r 1 -o md` on CPU.
- If `--draft-max` accepts an MTP-self-draft via a new flag (check `llama-bench --help | findstr -i mtp`), run again with that flag.

### Step 4 — sweep MTP draft-max
Once a baseline MTP run works:
```
-draft-max 1, 2, 3, 4, 6, 8
```
Record TG t/s and acceptance rate per setting. Best value is the
MTP cap that maximizes TG without driving the verifier ratio off a
cliff.

### Stop conditions for Track A
- MTP merge isn't on mainline AND `gg/spec-mtp-experiments` doesn't
  build cleanly → punt, document, move on. Don't sink >2 hr here.
- MTP-preserved GGUF unavailable from `havenoammo` AND nowhere
  else (search HF for `qwen3.6 mtp gguf`) → punt.
- MTP gives <10% TG win vs no-MTP-self-draft → record and stop;
  not worth more knob-twisting tonight.

### Result format
Add a `# Track A — Qwen3.6-27B MTP` section to a new doc
`docs/2026-05-13_overnight_perf_results.md`. Headline table:

| config | PP (t/s) | TG (t/s) | accept rate | wall (s) | notes |

---

## Track B — Hybrid GPU-prefill + CPU-decode (MXFP4_MOE)

### Goal
**Same GGUF hit by two compute backends to hit OpenCL's PP (210
t/s) AND CPU's TG (31 t/s) on Qwen3.6-35B-A3B in one inference.**
The hypothesis: 48 GB unified memory + 228 GB/s should support both
GPU and CPU touching the weights without copy overhead.

### Reality check first
llama.cpp does NOT have a built-in "phase split" (GPU prefill / CPU
decode) mode. What it has:
- `-ngl N`: layer split. First N layers on GPU, rest on CPU.
  Sequential per-layer dispatch — not parallel.
- `--split-mode {none,layer,row}`: multi-GPU split. We have one
  GPU.
- `--device <BACKEND>` for selecting backend per model run.

So a literal "GPU PP + CPU TG" doesn't exist as a knob. Three real
experiments instead:

#### B1 — `-ngl` sweep on Qwen3.6-35B-A3B MXFP4_MOE (1.5 hr)
Bench at `-ngl 0, 8, 16, 24, 32, 40, 56, 99` on OpenCL build. The
prediction: prefill scales with `-ngl` (GPU faster), decode goes
the other way (CPU faster). Find the crossover. Maybe an
intermediate `-ngl` gives best combined PP+TG.

```powershell
$d = "C:\Users\hotschmoe\Documents\GitHub\specula\llama.cpp"
$m = "C:\Users\hotschmoe\Documents\GitHub\specula\models\Qwen3.6-35B-A3B-MXFP4_MOE.gguf"
foreach ($ngl in 0,8,16,24,32,40,56,99) {
  Write-Output "=== ngl=$ngl ==="
  & "$d\build-opencl\bin\llama-bench.exe" -m $m -p 512 -n 128 -r 1 -o md -ngl $ngl 2>&1 | Select-Object -Last 4
}
```
Wrap in a single PowerShell tool call with `timeout=2700000` (45 min
total budget) since each ngl point is ~3-5 min and we have 8 points.

#### B2 — speculative decode "draft on GPU, target on CPU" (1 hr)
`--spec-draft-model` accepts a separate draft model. Use it as a
proxy for "two backends share work":
- Draft: `Qwen3-0.6B-Q8_0.gguf` on OpenCL (fast PP, low cost)
- Target: `Qwen3.6-35B-A3B-Q4_K_M.gguf` on CPU (correctness)

Per-backend device selection: check `--help` for `--device-draft` /
`--device-target` flags (added in #22XXX recently — look in
`common/arg.cpp` for `device-draft`). If present:
```
llama-speculative --model-draft Qwen3-0.6B-Q8_0.gguf --device-draft OPENCL `
                  --model Qwen3.6-35B-A3B-Q4_K_M.gguf --device-target CPU `
                  --draft-max 8 -p "..." -n 128
```
**This is the closest thing to "two backends in parallel" on
unified memory.** If it works, it's track B's win.

#### B3 — llama-server two-instance setup (skip unless B2 fails)
Run two `llama-server` processes (one OpenCL, one CPU) and a thin
proxy that routes prefill to one and decode to the other. The KV
cache sync is the killer here — probably not worth it overnight.

### Stop conditions for Track B
- `-ngl` sweep shows monotonic curve (no crossover sweet spot) →
  record, move on.
- `--device-draft` / `--device-target` flags don't exist on
  mainline → skip B2.
- Total Track B time > 2.5 hr → stop, write up what's there.

### Result format
`# Track B — GPU+CPU hybrid` section. Headline:

| backend config | -ngl | PP (t/s) | TG (t/s) | wall (s) | win vs single-backend? |

---

## Track C — "Try anything" (4 hr budget if A+B left time)

Order by expected payoff. **One bench at a time with timeout
discipline.**

### C1 — KleidiAI SME env sweep (30 min)
`GGML_KLEIDIAI_SME=N` for N in `{0,1,2,4,8,16}` on `build-cpu-kleidiai`.
The X2 Elite has known-broken default SME dispatch (returns 0 from
detect → falls back to non-SME). Force-enabling at various core
counts may unlock a perf path. Bench Qwen3-4B Q4_K_M only.

```powershell
$d = "C:\Users\hotschmoe\Documents\GitHub\specula\llama.cpp"
$m = "C:\Users\hotschmoe\Documents\GitHub\specula\models\Qwen3-4B-Q4_K_M.gguf"
foreach ($sme in 0,1,2,4,8,16) {
  $env:GGML_KLEIDIAI_SME = "$sme"
  Write-Output "=== SME=$sme ==="
  & "$d\build-cpu-kleidiai\bin\llama-bench.exe" -m $m -p 512 -n 128 -r 1 -o md -t 8 2>&1 | Select-Object -Last 4
  Remove-Item Env:GGML_KLEIDIAI_SME -ErrorAction SilentlyContinue
}
```
**Watchdog:** any SME value > 0 may segfault (see
`docs/SME_investigation.md`). Catch the crash, log it, continue.

### C2 — OpenCL F16xF32 prefill GEMM opt-in (45 min)
PR #22755 added an OPT-IN Adreno xmem F16xF32 GEMM for prefill.
Find the runtime flag (grep `arg.cpp` for `xmem` or `f16xf32`),
enable, bench Qwen3.6-35B-A3B MXFP4_MOE.

Expected: PP > 210 t/s on Adreno OpenCL if the opt-in helps.

### C3 — `-b` / `-ub` batch-size sweep (45 min)
llama-bench default `-b 2048 -ub 512`. Try `-b 4096 -ub 1024` and
`-b 8192 -ub 2048` on OpenCL Q4_0 (Qwen3-4B). Adreno's GEMM kernels
may improve with larger ubatch.

### C4 — Vulkan envvar exploration (45 min, KEEP A WATCHDOG)
With `DISABLE_F16` confirmed broken, try other knobs (in isolation,
short bench, kill if `<1%` util after 30s):
- `GGML_VK_USE_F32=1` (does this exist?)
- `GGML_VK_FORCE_SHADER=...`
- `GGML_VK_VALIDATE=0` (turn off validation)
- `GGML_VK_DEBUG=1` then read the log for clues

Bench: r=1, n=64 only. **30 s watchdog per probe.**

### C5 — quant comparison on Qwen3-4B (45 min)
Convert/download alternative quants and bench:
- Q3_K_M, Q4_K_S, Q5_K_M, Q6_K, Q8_0
- IQ4_XS, IQ3_M (if available)

Best non-NPU TG for 4B at session close was Vulkan 38 t/s.
Q3_K_M may push higher.

### C6 — Concurrency sweep on OpenCL Adreno MoE (30 min)
Existing `scripts/bench_concurrency4_all_backends.py` covers CPU
and OpenCL on Qwen3-4B / Qwen2.5-7B. Run it on Qwen3.6-35B-A3B
MXFP4_MOE to characterize agentic-scale MoE throughput on Adreno
with the new MoE kernels.

### Stop conditions for Track C
- Per-sub-track 30 min hard cap. If a probe is mid-bench at 30
  min, kill it and move on.
- Total track C budget: 4 hours. If you've used 4 hr on tracks A+B,
  skip C entirely.

### Result format
`# Track C — Knob sweeps` section. One subsection per Cx. Best
finding at top.

---

## Do NOT do this overnight

1. **Do not pull llama.cpp HEAD again.** Bench numbers compare to
   the 2026-05-12 baseline at `856c3adac`. Pulling forward makes
   the deltas meaningless.
2. **Do not build the DFlash/PFlash/MTP forks.** Already audited;
   all are CUDA-only or wrong model. See
   `docs/2026-05-12_sweep_and_mtp_landscape.md` §3 table.
3. **Do not try to backport MTP to ORT-QNN.** That's a multi-day
   research project, not an overnight knob.
4. **Do not run `GGML_VK_DISABLE_F16=1` anywhere.** Confirmed
   segfault on `856c3adac`.
5. **Do not let any bench run >15 min without a watchdog.** Even
   if the "expected wall" is uncertain. Better to kill a slow run
   and document than to lose the night to a stuck process.
6. **Do not commit anything mid-track.** Commit once at the end
   with a single tidy message + the full overnight results doc.

---

## End-of-night checklist

Before reporting back to the user in the morning:

- [ ] `docs/2026-05-13_overnight_perf_results.md` written with all
      three tracks (even if a track was "punted, here's why").
- [ ] Per-track CSV(s) in `results/csv/` with `2026-05-13` date
      tag.
- [ ] Best new headline number, if any, called out at the top of
      the results doc.
- [ ] Surprising findings noted with enough detail to reproduce.
- [ ] `current_status.md` session 26 banner prepended (1 paragraph).
- [ ] Single commit with subject like
      `2026-05-13 overnight perf sprint: <one-line headline>`.
- [ ] Raw bench logs under `marked_for_deletion/2026-05-13_overnight/`
      (gitignored).
- [ ] `git push origin master`.

---

## Reference — key file paths

- Models: `models/Qwen3-4B-Q4_K_M.gguf`, `Qwen3-4B-Q4_0.gguf`,
  `Qwen3.6-35B-A3B-{Q4_K_M,MXFP4_MOE}.gguf`,
  `Qwen3-0.6B-Q8_0.gguf` (draft candidate),
  `Qwen3-14B-Q4_K_M.gguf` (SQ1 target — don't touch).
- Builds: `llama.cpp/build-{cpu,cpu-kleidiai,opencl,vulkan}/bin/llama-*.exe`.
- Bench harness: `scripts/bench_qwen3_4b_all_backends.py`,
  `scripts/bench_concurrency4_all_backends.py`,
  `npu_engine/bench_qwen3_4b_ortqnn.py`.
- Venv: `.venv/Scripts/python.exe`.
- QAIRT: `C:\Qualcomm\AIStack\QAIRT\2.45.40.260406`.

Good luck. Aim for one solid headline, not 12 partial knobs.
