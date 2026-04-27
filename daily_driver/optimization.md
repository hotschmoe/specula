# Daily-driver optimization — Qwen3.6-35B-A3B

Living investigation doc for the chosen production-style local model.
Append findings here; don't open a new doc unless the topic is worth
its own life (per `docs/repo_hygiene.md`). The headline tables at the
top are the answer; everything below is the work that produced them.

**Phase: kickoff (2026-04-26)**. Models downloading, no rows landed
yet. The TODO grid below is the test plan.

## What we're optimizing for

The workload (per `README.md` § Primary use case) is a **long-running
coding agent at 120k+ context, concurrency=1, heavy KV reuse**. That
reorders the metrics from how the 4B/7B baseline docs framed them.

In priority order:

1. **TG t/s at long context (32k / 64k / 120k)**, single-stream.
   The agent spends almost no time at short context after warm-up.
   A backend that does 25 t/s at 1k-ctx and 8 t/s at 120k-ctx is
   *worse* than one doing 18 t/s flat. Floor: ≥ 12 t/s at 120k
   (slow but usable); aspiration: ≥ 18 t/s at 120k.
2. **TTFT after a tool call** — the actual UX number for an agentic
   loop. With KV reuse, only the delta tokens (tool result + harness
   framing, typically 200-2000 tokens) need prefilling. So **PP t/s
   on a small delta** matters more than PP t/s on a cold 4k prompt.
3. **Memory headroom at 120k ctx.** Does the chosen config actually
   fit in 48 GB unified RAM? A config that OOMs at 80k is dead. KV
   quant (`-ctk q8_0 -ctv q8_0` minimum) is **non-optional** here.
4. **Quality at 120k ctx.** A model that loses coherence past 32k
   isn't a 120k model, no matter what the spec sheet says. Sanity-
   check with a needle-in-haystack probe before declaring a winner.
5. **J/tok on battery.** Secondary — the rig is mostly AC-tethered
   while the agent runs. Run BAT measurements only on the canonical
   AC winners.

What we are *not* optimizing for:

- **N=4 aggregate throughput.** Concurrency=1 only. The 7B's "Vulkan
  N=4 hits 3× scaling" finding is irrelevant.
- **Cold PP at 4k.** Yes we measure it, but a 4k cold-prompt time of
  10 s only matters once per session start.
- **Maximum benchmark t/s** at trivial context. Pretty numbers at 1k
  ctx are misleading for this use case.

## Backend starting configs (what we measure first)

| backend | build | model | starting flags | source of starting config |
|---|---|---|---|---|
| CPU         | `build-cpu`          | Q4_K_M (lmstudio-community)   | `-t 8 -p 512 -n 128 -r 3` | matches 4B / 7B baseline |
| CPU+KleidiAI| `build-cpu-kleidiai` | Q4_K_M (lmstudio-community)   | `-t 8 -p 512 -n 128 -r 3` | matches baselines |
| GPU OpenCL  | `build-opencl`       | MXFP4_MOE (Unsloth)           | `-ngl 99 -p 512 -n 128 -r 3` | MXFP4 is on Adreno's fast-path list (`Q4_0/Q8_0/MXFP4`); plain Q4_0 doesn't exist for this model |
| GPU Vulkan  | `build-vulkan`       | MXFP4_MOE (Unsloth)           | `-ngl 99 -p 512 -n 128 -r 3`, env `GGML_VK_DISABLE_F16=1 GGML_VK_PREFER_HOST_MEMORY=1` | 7B doc had this combo on Q4_0; whether it carries to MXFP4_MOE is an open question — confirm with knob A/B during matrix sweep |

**Headline AC table — short-ctx baseline (Phase 1 done 2026-04-26)**:

Initial PP512/TG128 sanity sweep at default ctx — comparable to the
4B/7B baselines, *not* representative of the daily-driver workload.

| backend | model | PP512 t/s | TG128 t/s | wall (s) |
|---|---|---:|---:|---:|
| **CPU**        | Q4_K_M    | 161.71      | **36.01**   | 38.5 |
| CPU+KleidiAI   | Q4_K_M    | 140.86      | 33.61       | 39.7 |
| **GPU OpenCL** | MXFP4_MOE | **180.28**  | 17.43       | 59.7 |
| GPU Vulkan     | MXFP4_MOE | 46.05       | 22.73       | 75.1 |

CSV: `results/csv/daily_driver_2026-04-26_full_ac.csv`. CPU wins TG
by a wide margin; OpenCL wins PP. Vulkan is broken with the
DISABLE_F16+PREFER_HOST combo that worked on Q4_0 at 7B — Phase 3
will A/B that. See § Findings for the full analysis.

**Headline AC table — long-context (the one that actually matters, TODO)**:

KV at q8_0 always (per the use-case section). TG measured after the
context is filled to the listed depth, then 128 more tokens are
generated.

| backend | model | TG@4k | TG@32k | TG@120k | TTFT(delta=512) | mem peak | CSV |
|---|---|---:|---:|---:|---:|---:|---|
| CPU         | Q4_K_M    | TODO | TODO | TODO | TODO | TODO | `results/csv/daily_driver_longctx_*.csv` |
| CPU+KleidiAI| Q4_K_M    | TODO | TODO | TODO | TODO | TODO | same |
| GPU OpenCL  | MXFP4_MOE | TODO | TODO | TODO | TODO | TODO | same |
| GPU Vulkan  | MXFP4_MOE | TODO | TODO | TODO | TODO | TODO | same |

## Variable sweep matrix

We sweep one variable at a time, holding everything else at the
backend's starting config. Order is "biggest expected effect first"
so we can stop early if a knob settles the question.

### Tier-1 sweeps (run on every backend at first opportunity)

Note on context size: every Tier-1 measurement runs at **three context
points** — 4k (warm-up reference), 32k (mid-agent-session), 120k
(end-of-agent-session). A knob that helps at 4k and hurts at 120k is
a regression for our workload. Build the runner to default to a 3-ctx
sweep, not a single point.

| variable | values | hypothesis | success criterion |
|---|---|---|---|
| **`-c` (context size)** | 4096, 32768, 131072 | TG t/s drops with longer ctx (KV read per step grows); want the slope, not just an endpoint | report TG@4k, TG@32k, TG@120k for every backend; the **120k number is the headline** |
| **`-ctk`, `-ctv` (KV quant)** | q8_0 (default), q4_0, f16 (reference only) | q8_0 KV halves KV memory at near-zero quality cost; q4_0 quarters it but may dent retrieval | confirm q8_0 is essentially free; quantify q4_0's quality cost via needle-in-haystack at 64k |
| `-t` (threads) | 4, 6, 8, 10, 12 | X2 has 12 P-cores; TG sweet spot may be lower than PP sweet spot, especially at long ctx where KV-read bandwidth dominates | find the knee for PP and TG@120k separately |
| `-b` (logical batch) | 256, 512, 1024, 2048 | larger PP batch = better matmul utilization on GPU; matters most for the *initial* cold prefill | only if PP@120k improves |
| `-ub` (physical ubatch) | 128, 256, 512 | smaller ubatch reduces peak memory, may unlock more `-ngl` at long ctx | check at 120k where headroom is tight |
| `--no-mmap` | on / off | mmap can fault under memory pressure at high resident set (model + 120k KV); disabling forces full load up-front | TG variance ↓ at 120k, PP unchanged |
| `--mlock` | on / off | locks pages; combined with Windows large pages may cut TLB miss rate on a ~30 GB working set | only if --no-mmap shows variance |
| **`--lookup` / `--lookup-cache-dynamic`** | off / on | n-gram lookup decoding: free win on the repetitive JSON tool-call output our use case generates | acceptance > 0.3 on a tool-call-heavy bench prompt = real win |
| **prefix cache** (server-side) | hit-rate measurement, not on/off | every tool-call turn extends prior KV; quantify the cost of a delta-only prefill vs cold | TTFT(delta) << TTFT(cold) — this is the whole agentic-loop UX |

### Tier-2 sweeps (after Tier-1 picks a winning backend)

| variable | values | applies to | hypothesis |
|---|---|---|---|
| Quant variant | Q4_K_M (lmstudio), MXFP4_MOE (unsloth), UD-Q4_K_M, UD-IQ4_XS, Q5_K_M, Q6_K, Q8_0 | all | quality vs throughput trade. Plain Q4_0 doesn't exist for this model — MXFP4_MOE substitutes on the Adreno fast-path. Unsloth's UD- variants are dynamic-quant — measure separately to know if "UD" pays off here |
| `GGML_VK_DISABLE_F16` | 0 / 1 | Vulkan | already known win at 7B; re-confirm at 35B |
| `GGML_VK_PREFER_HOST_MEMORY` | 0 / 1 | Vulkan | already known win at 7B |
| `GGML_OPENCL_USE_ADRENO_KERNELS` | ON / OFF | OpenCL | ON should win on Q4_0; verify |
| `-fa` (flash attention) | on / off | all | if available for the backend, often a TG win on long ctx |
| Native CPU build (`-march=native`, `-DGGML_NATIVE=ON`) | rebuild | CPU | check current `build-cpu` was configured native; ARM64 NEON should be auto, i8mm via KleidiAI build |
| KleidiAI SME2 fence | scripts/patch_kleidiai_detect.py on/off | CPU+KleidiAI | currently SME2 is fenced off (per repo memory); A3B may actually fit SME2's tile better — worth measuring |
| Windows large-page support | Lock Pages in Memory privilege + `--mlock` | CPU | reduces TLB miss cost on 22 GB resident; may also need driver-level toggle |
| `--numa` | distribute / isolate / off | CPU | X2 is single-NUMA-node, so this is mostly inert; document the no-op for completeness |

### Tier-3 sweeps (only if Tier-2 leaves an open question)

| variable | values | applies to | notes |
|---|---|---|---|
| `-sm row` vs `-sm layer` | row / layer | GPU | layer split is default; row may help if expert weights aren't contiguous in VRAM |
| Rope scaling / context extension | YaRN / linear | all | only if we want >native ctx (i.e., model native is <120k and we need to push past) |
| Concurrent streams (`-np N`) | 1, 2 | all | **Demoted from Tier-1.** Use case is concurrency=1, but a tiny N=2 may matter if we ever drive two agents at once. Skip until that's a real ask. |

## Research questions / advanced speed paths

These need investigation before they enter the sweep matrix. Each
gets a short stub here; promote to its own doc only if it produces
a real lever.

### Speculative decoding (the repo's namesake)

Three flavors to try, in priority order:

1. **DFlash block-diffusion draft** — `z-lab/Qwen3.6-35B-A3B-DFlash`
   is a draft model purpose-built for our exact target. See § DFlash
   below for details. Highest expected acceptance rate of the three
   options *because the draft was trained against this target*, but
   needs z-lab's runtime, not llama.cpp.
2. **Small-dense-draft + MoE-target via llama.cpp `--draft`**. Drafts
   already in `models/`: `Qwen3-0.6B-Q8_0.gguf`, `Qwen3-1.7B-Q8_0.gguf`.
   Catch: tokenizer must match. Qwen3 → Qwen3.6 is *probably*
   compatible (same family) but verify before benching — silent
   mismatch kills acceptance rate. This is the easiest path to
   stand up because it lives entirely inside llama.cpp.
3. **Self-speculation with MoE**. Run the target with a top-k=1
   expert draft (cheaper) and verify with top-k=2/4. Requires
   runtime support that llama.cpp may not have yet for MoE —
   check `llama-speculative` and `llama-server -md`.

**Acceptance-rate measurement protocol** (applies to all three):
fixed test prompt, `--draft N` sweep (1, 2, 4, 8). Headline metric is
`tokens/s on target × acceptance rate = effective t/s`.

Open question for MoE targets: does verify cost dominate enough to
nullify the speedup? With ~3B active params per step, draft+verify
might not beat plain decode unless the draft is very cheap (e.g.,
the 0.6B Qwen3 draft). **Start by measuring acceptance rate on a
200-token chat-style continuation with the 0.6B draft via llama.cpp;
if AR < 0.5, the dense-draft path is dead and DFlash becomes the
only way forward.**

### N-gram lookup decoding (`--lookup`, `--lookup-cache-*`) — promoted to Tier-1

**This is the highest-EV cheap knob for our specific use case.**
llama.cpp has a draft-free spec mode: build an n-gram cache from the
prompt + generated text, propose continuations, verify. The agent
workload is **dominated by repetitive structured output** — JSON
tool-call envelopes, file paths, error message templates, code
patterns, the harness's own framing tokens. All recur across turns.
N-gram lookup turns this redundancy into free throughput.

To try:

```bash
llama-server -m model.gguf \
    --lookup-cache-dynamic <path> \
    --draft 8 \
    -c 131072 -ctk q8_0 -ctv q8_0
```

Bench plan (do this *early*, before the broader matrix is filled —
if the win is large, the rest of the matrix should be measured with
lookup on):

1. Capture a real opencode/Hermes session log (5-10 turns of tool
   calls + responses). Save the prompts + model outputs.
2. Replay deterministically (seed=0, temp=0) with `--lookup` on/off.
3. Measure: aggregate t/s, per-turn TTFT, acceptance rate.

Expected outcome: **acceptance rate 0.3-0.5 on tool-call turns** is
realistic; that maps to ~1.4-2× effective t/s for those turns. Novel
prose turns tie at AR ≈ 0. Net win across a session = function of
the tool-call/prose ratio, which for an agentic loop is heavily
tool-call-weighted.

Risk: lookup-cache can blow up RAM if `--lookup-cache-dynamic` is
unbounded across a long session. Cap with `--lookup-cache-static`
sized to ~100 MB and let the dynamic cache evict.

### TurboQuant — KV cache compression

**What it is** (clarified 2026-04-26): KV cache quantization to
**3-bit keys / 2-bit values**, ICLR 2026 (arXiv:2504.19874).
Reference implementation at `0xSero/turboquant` is a vLLM integration
with Triton kernels. Per their README, on Qwen3.5-35B-A3B MoE
(extremely close cousin to our target) on 8× RTX 3090 with vLLM:
prefill +5.7%, decode +3.1%, **30 GB KV freed**, 2× max-token capacity
at long ctx. The throughput gains are modest; the real win is memory
headroom, which buys longer context.

**Catch for this rig**: TurboQuant is **vLLM + Triton + CUDA**.
None of those run on the X2 Elite — no CUDA, no Triton ARM64 build,
llama.cpp is our serving runtime. There's no llama.cpp port yet.
That makes TurboQuant a **research / "watch upstream"** item, not
a near-term knob.

What we *can* do near-term:

- Use llama.cpp's existing KV quant (`-ctk q8_0 -ctv q8_0`,
  optionally `q4_0`) as a coarser version of the same idea. Tier-1
  sweep already covers this.
- Track the 0xSero repo + the ICLR paper's official release for a
  llama.cpp PR. If/when one lands, add `-ctk q3_K -ctv q2_K` (or
  whatever the format names settle on) to the sweep.
- If we ever stand up a CUDA/Triton machine for cloud-side prep
  work (`docs/rent_cloud_compute.md` Scenario A), TurboQuant is a
  bench-worthy comparison there — but it doesn't help the laptop
  serving path.

**Action**: park as research. Re-evaluate when llama.cpp KV-quant
formats grow past q4_0/q8_0 or when `0xSero/turboquant` (or upstream
authors) ship a non-CUDA implementation.

### DFlash — block-diffusion speculative decoding

**What it is** (clarified 2026-04-26): a lightweight block-diffusion
model used as a speculative-decoding draft. Repo `z-lab/dflash`.
Block diffusion enables high-quality parallel drafting (multiple
tokens per draft step rather than autoregressive one-at-a-time),
which raises acceptance rate vs a same-size dense draft.

**Why this is a bigger deal than I had initially scoped**: z-lab has
**already trained and published a DFlash draft for Qwen3.6-35B-A3B**
— `z-lab/Qwen3.6-35B-A3B-DFlash` on HF. That removes the hardest
part of doing spec decoding well (training a draft that's both small
and aligned with the target) and makes DFlash the **highest-priority
spec-decoding path to investigate** for this exact model.

**Catch**: DFlash drafts run on z-lab's own runtime (block-diffusion
inference is non-standard — not a vanilla AR draft). It's **not a
drop-in for llama.cpp's `--draft` flag**, which assumes an AR draft
sharing the same tokenizer. To bench DFlash here we'd need to:

1. Stand up z-lab's runtime (Python + their inference stack) on the
   X2 Elite. ARM64-Windows porting may bite — likely needs WSL or a
   Linux VM, or an x86-Windows fallback.
2. Drive their runtime with the target weights we already have
   (or download z-lab's blessed target+draft pair).
3. Benchmark against vanilla llama.cpp serving for the same prompts.

This is its own ~half-day workstream, not a knob. Still worth the
investment because the published draft already exists for our target.

**Action**: open a sibling investigation doc
(`daily_driver/dflash_investigation.md`, TBD) once we've nailed
down the llama.cpp-only baseline numbers in the matrix above. Don't
start until we have a credible baseline to beat — otherwise we
won't know whether DFlash's gain is real or just covering for an
un-tuned vanilla path.

**Companion question**: does Qwen3.6-35B-A3B-DFlash ship in a format
llama.cpp could load as a draft (e.g., a GGUF-able AR distillation)?
If z-lab also publishes an AR-mode variant of the draft, the
llama.cpp `--draft` path becomes available without their full
runtime. Check the HF repo's file listing on first visit.

### Native CPU build / explicit ISA flags

llama.cpp's `build-cpu` should already pick up ARM64 NEON. The
KleidiAI build adds i8mm. Open questions:

- Does our `build-cpu` actually compile with `-mcpu=native` and
  ARM64-specific FP16 intrinsics? Verify by `objdump -d
  build-cpu/bin/llama-bench.exe | grep -c fmla` and compare against
  a `-DGGML_NATIVE=OFF` rebuild.
- **SVE / SVE2** — does the X2 Elite expose SVE? `llvm-readobj` on
  any compiled binary would show SVE intrinsics if used. The
  Snapdragon X2 cores are Oryon-class; SVE2 support is documented
  in Qualcomm's spec sheet but may need an explicit `-march=armv9-a+sve2`.
- **SME2** — currently fenced off via `scripts/patch_kleidiai_detect.py`
  per repo memory. If SME2 is actually present and we just lacked
  confidence in older runs, re-enable on this model (matmul tiles
  are bigger at 35B than 4B, may show the win that 4B/7B didn't).

### Huge pages / large pages on Windows ARM64

Windows requires **Lock Pages in Memory** privilege + `--mlock` to
use large pages. Effect on a 22 GB resident model: TLB miss rate
should drop, helping memory-bound TG. To enable:

1. `secpol.msc` → User Rights Assignment → Lock pages in memory →
   add user.
2. Reboot.
3. Re-run with `--mlock`.

Effect size: low single-digit percent on Linux x86 historically;
unknown on ARM64 Windows. Worth a one-shot test.

### NUMA

Snapdragon X2 Elite Extreme is a single SoC with all 12 P-cores in
one cluster sharing the L3 / SLC. **No NUMA in the strict sense.**
The `--numa` flag should be a no-op. Run a one-shot "is it actually
a no-op?" measurement with `--numa distribute` vs nothing; if
results differ, we've discovered something interesting; if not,
document the no-op and move on.

### KV cache quantization (also in Tier-1 sweep above)

Promoted here because it's the one knob with a quality dimension.
Run perplexity on a small held-out set (e.g.,
`llama-perplexity -m … -ctk q8_0 -ctv q8_0 -f wikitext.test`) for
each KV-quant combo. Acceptable PPL drift bound: **< 0.5% relative**.
Anything above and we keep f16 for KV.

Memory math at ctx=8192, 35B model:
- KV f16: ~1.2 GB
- KV q8_0: ~0.65 GB
- KV q4_0: ~0.35 GB

So KV quant only matters when we push ctx ≥ 16k. At default 8k the
absolute savings are small; quality risk dominates. **Start with
f16 KV; only revisit if we want long-context modes.**

### Sliding-window / sink attention

If Qwen3.6-A3B uses sliding-window attention natively, llama.cpp's
attention path may or may not respect it. Worth checking
`gguf_dump.py` on the downloaded GGUF for the SWA tokens. If it's
there, ensure the build supports it; if not, document the cap on
ctx.

### llama.cpp commit cadence

The 7B doc landed on llama.cpp `f53577432` (2026-04-26). MoE-specific
kernel improvements are landing fast in upstream. Once the matrix
is filled at the pinned commit, do a single rebase to the latest
`master` and re-run the canonical winning row to detect drift. If
PP or TG moves > 5%, re-bench the matrix.

## Findings (append-only log)

### 2026-04-26 — kickoff

- Directory created. Recipe drafted. Downloads kicked off:
  `Qwen3.6-35B-A3B-Q4_K_M.gguf` (lmstudio-community, 21.17 GB) and
  `Qwen3.6-35B-A3B-MXFP4_MOE.gguf` (Unsloth, 21.71 GB).
- **GPU quant changed: Q4_0 → MXFP4_MOE.** Verified across Unsloth,
  lmstudio-community, mradermacher, bartowski — **no plain Q4_0
  exists for this model in 2026.** The community moved past Q4_0;
  K-quants and MXFP4 dominate. Adreno OpenCL's documented fast-path
  list is `{Q4_0, Q8_0, MXFP4}` so MXFP4 is the natural substitute,
  and Unsloth's `MXFP4_MOE` is even MoE-tuned (per-expert MXFP4).
  Open question: do the 7B Vulkan knobs (`DISABLE_F16+PREFER_HOST`)
  still help on MXFP4_MOE? Will A/B during matrix sweep.
- **CPU GGUF from `lmstudio-community`**, not Unsloth. Their plain
  Q4_K_M is the "comparable to most users" baseline; Unsloth's
  `UD-Q4_K_M` is dynamic-quant variant — measure as a Tier-2 sweep.
- Research-paths clarified:
  - **TurboQuant** = KV-cache 3-bit-key / 2-bit-value compression
    (ICLR 2026, arXiv:2504.19874, ref impl `0xSero/turboquant`).
    vLLM + Triton + CUDA only — not a near-term knob on the laptop;
    park as research, use llama.cpp `-ctk q8_0/q4_0` as the
    near-term substitute.
  - **DFlash** = z-lab block-diffusion speculative-decoding draft.
    A draft is **already published for our exact target**:
    `z-lab/Qwen3.6-35B-A3B-DFlash`. Needs z-lab's runtime, not
    llama.cpp. Promoted to "first non-trivial spec-decode bet"
    once llama.cpp baseline is locked.

### 2026-04-26 — Phase 1: full 4-backend AC matrix, PP512/TG128

CSV: `results/csv/daily_driver_2026-04-26_full_ac.csv`. llama.cpp
build `f53577432`. KV at f16 (default; the long-ctx sweep is where
KV quant matters).

| backend | model | PP512 (t/s) | TG128 (t/s) | wall (s) |
|---|---|---:|---:|---:|
| **CPU**        | Q4_K_M    | 161.71      | **36.01**   | 38.5 |
| CPU+KleidiAI   | Q4_K_M    | 140.86      | 33.61       | 39.7 |
| **GPU OpenCL** | MXFP4_MOE | **180.28**  | 17.43       | 59.7 |
| GPU Vulkan     | MXFP4_MOE | 46.05       | 22.73       | 75.1 |

(GPU Vulkan with default env knobs `GGML_VK_DISABLE_F16=1
GGML_VK_PREFER_HOST_MEMORY=1`.)

**Headline finding: CPU is the surprise winner on TG.** 36.01 t/s on
a 35B model from a laptop CPU — the 3B-active MoE architecture is
delivering exactly what it's engineered for. CPU TG beats every GPU
backend. Comparison against prior baselines:

| model | CPU PP | CPU TG |
|---|---:|---:|
| Qwen3-4B (dense)            | 188.30 | 39.50 |
| Qwen2.5-7B (dense)          | 122.98 | 24.17 |
| **Qwen3.6-35B-A3B (MoE)**   | **161.71** | **36.01** |

35B-A3B CPU TG (36.01) beats the dense 7B (24.17) by **+49%** despite
having 5× the parameter count, because only ~3B params are touched
per decode step. PP also beats 7B (+31%) — batched prefill amortizes
the larger expert pool well.

**Surprises**:

1. **KleidiAI lost on both axes** (-13% PP, -7% TG vs plain CPU).
   This **inverts** the 7B finding (KleidiAI +2% PP / +5% TG on
   dense Q4_K_M). Hypothesis: MoE routing breaks the matmul tile
   shapes the i8mm ukernels are tuned for; per-token expert
   selection produces irregularly-shaped GEMMs that don't fit the
   ukernel sweet spot. **For 35B-A3B, plain CPU > KleidiAI — strip
   KleidiAI from the daily-driver serve config.**

2. **Vulkan PP crashed to 46 t/s** (3.5× slower than CPU, 4× slower
   than OpenCL). Vulkan TG of 22.73 is *also worse than CPU*. The
   `DISABLE_F16+PREFER_HOST` combo that fixed Vulkan PP on Q4_0 at
   7B is the wrong combo for MXFP4_MOE. Either MXFP4 wants the F16
   path on (it's a mixed-precision format by design), or there's
   another knob. **Phase 3 (Vulkan env A/B) is now urgent** —
   running default-Vulkan-env next will tell us if the F16-off
   knob is actively hurting MXFP4.

3. **OpenCL PP best, OpenCL TG worst.** Same shape as the 4B/7B
   findings — Adreno's per-token kernel-launch overhead dominates
   at AR=1 decode. MXFP4_MOE didn't change this.

**Decisions this matrix unblocks**:

- **CPU + Q4_K_M is the new daily-driver default**, not Vulkan + MXFP4.
  At 35B-A3B-Q4_K_M, CPU wins TG (UX-critical) and only loses PP by
  10% to OpenCL. Memory pressure at 120k ctx still favors CPU since
  the host RAM has 48 GB; GPU has Adreno-budget concerns.
- **Vulkan + MXFP4_MOE needs Phase 3 to be salvageable.** If
  default-Vulkan-env (no F16-off knob) still doesn't beat CPU,
  Vulkan is out for this model + quant combo.
- **Strip KleidiAI from the recipe** — it was added as a paranoid
  build option in case it won. It doesn't on this MoE.

### 2026-04-26 — use case clarified: coding agent, 120k ctx, conc=1

User confirmed the daily-driver target is a **long-running coding
agent in a harness** (opencode, Hermes, Pi). That reordered the
optimization priorities significantly:

- **Concurrency=1 only.** Drops the entire N=4 matrix from scope.
  The 4B/7B doc N=4 wins (Vulkan-Q4_0 hitting 3× scaling) are
  irrelevant here. Single-stream TG is the metric.
- **120k+ ctx is the canonical operating point.** The headline
  measurement is TG@120k, not TG@1k. Added a 3-context-point sweep
  (4k / 32k / 120k) to Tier-1.
- **KV quant `q8_0` made default-on**, not a sweep variable. At
  120k ctx, f16 KV is unjustifiable (5-8 GB at f16 with mixed-attn
  arch, more if pure full-attn) when q8_0 halves it for ~zero
  quality cost.
- **N-gram lookup decoding promoted to Tier-1** as the cheapest
  potential win — agentic tool-call output is highly repetitive
  (JSON envelopes, file paths, error templates), which is exactly
  the workload n-gram lookup wins on.
- **TTFT-after-tool-call** added as a primary metric. The agent
  loop's UX is gated by how fast the model responds after a tool
  result lands, which is dominated by PP on the *delta* tokens
  with the rest of the context already in KV cache. Prefix-cache
  hit-rate on llama-server matters here.
- **Quality at long context** added as gate #4. A model that loses
  coherence past 32k isn't a 120k model. Run a needle-in-haystack
  probe before declaring a winner.

(Add new dated subsections below as runs land. Mirror the
`docs/qwen2_5_7b_baseline_all_backends.md` § "Update log" pattern
once we have data.)

## Runner (TODO)

Adapt `scripts/bench_qwen2_5_7b_all_backends.py` into
`scripts/bench_daily_driver.py` once the GGUFs are on disk:

- Drop the NPU-Genie row (out of scope).
- Add a `--quant` flag toggling between Q4_K_M and Q4_0 (so CPU
  and GPU rows pick their respective quants automatically).
- Output to `results/csv/daily_driver_<tag>.csv`.
- Logs to `marked_for_deletion/daily_driver_<tag>/` per
  `docs/repo_hygiene.md`.

PP512 prompt: regen with `gen_pp512_prompt_qwen2_5_7b.py` style,
swapping in the Qwen3.6 tokenizer (download
`Qwen/Qwen3.6-35B-A3B/tokenizer.json` from HF). Output to
`results/daily_driver/pp512_prompt.txt`.

## What this doc is *not*

- Not a quality-eval doc. Perplexity/MMLU/etc. for the chosen quant
  goes in a separate `daily_driver/quality.md` *if and when* we
  decide quant choice is contested. Throughput is the lead question
  here.
- Not a battery-life doc. BAT measurements are a follow-on once AC
  picks a winner. Mirror the 7B doc's BAT pattern when we get there.
- Not a place for backend-build instructions. Those live in
  `docs/qwen3_4b_baseline_methods.md`.
