# Daily-driver optimization — Qwen3.6-35B-A3B

Living investigation doc for the chosen production-style local model.
Append findings here; don't open a new doc unless the topic is worth
its own life (per `docs/repo_hygiene.md`). The headline tables at the
top are the answer; everything below is the work that produced them.

**Phase: kickoff (2026-04-26)**. Models downloading, no rows landed
yet. The TODO grid below is the test plan.

## What we're optimizing for

In priority order:

1. **TG (decode) tokens/sec, single-stream** — the chat-loop UX number.
   Aspirational floor: **20 t/s** at 35B-A3B (≈ what the 4B baseline
   delivered in CPU TG). Anything ≥ 15 t/s is usable; <10 t/s isn't.
2. **PP (prefill) tokens/sec** — code-paste / long-context-loading
   latency. Below ~150 t/s a 4k-token input takes >25 s, which is
   noticeable.
3. **J/tok on battery** — secondary; the `daily_driver` rig is mostly
   AC-tethered, but battery numbers gate "use this on a flight"
   workflows. Run BAT only on the canonical wins from AC.
4. **Time-to-first-token (TTFT)** — derivable from PP, but the
   server-side framing overhead is its own thing; measure once via
   `llama-server` curl timing on the canonical config.

## Backend starting configs (what we measure first)

| backend | build | model | starting flags | source of starting config |
|---|---|---|---|---|
| CPU         | `build-cpu`          | Q4_K_M | `-t 8 -p 512 -n 128 -r 3` | matches 4B / 7B baseline |
| CPU+KleidiAI| `build-cpu-kleidiai` | Q4_K_M | `-t 8 -p 512 -n 128 -r 3` | matches baselines |
| GPU OpenCL  | `build-opencl`       | Q4_0   | `-ngl 99 -p 512 -n 128 -r 3` | 7B doc: Q4_0 +55% PP / +16% TG over Q4_K_M |
| GPU Vulkan  | `build-vulkan`       | Q4_0   | `-ngl 99 -p 512 -n 128 -r 3`, env `GGML_VK_DISABLE_F16=1 GGML_VK_PREFER_HOST_MEMORY=1` | 7B doc: this combo unbroke Vulkan PP and put TG (25.80) above CPU (24.17) |

**Headline AC table (TODO — populated as rows land)**:

| backend | model | PP t/s | TG t/s | TG / generated tok | wall (s) | CSV |
|---|---|---:|---:|---:|---:|---|
| CPU         | Q4_K_M | TODO | TODO | TODO | TODO | `results/csv/daily_driver_2026-04-XX_ac.csv` |
| CPU+KleidiAI| Q4_K_M | TODO | TODO | TODO | TODO | same |
| GPU OpenCL  | Q4_0   | TODO | TODO | TODO | TODO | same |
| GPU Vulkan  | Q4_0   | TODO | TODO | TODO | TODO | same |

## Variable sweep matrix

We sweep one variable at a time, holding everything else at the
backend's starting config. Order is "biggest expected effect first"
so we can stop early if a knob settles the question.

### Tier-1 sweeps (run on every backend at first opportunity)

| variable | values | hypothesis | success criterion |
|---|---|---|---|
| `-t` (threads) | 4, 6, 8, 10, 12 | X2 has 12 P-cores; saturating may help PP, hurt TG (cache thrash) | find the knee for PP and TG separately |
| `-c` (context size) | 2048, 4096, 8192, 16384 | KV cache grows linearly; TG t/s drops with longer ctx (KV read per step) | quantify the slope |
| `-b` (logical batch) | 256, 512, 1024, 2048 | larger PP batch = better matmul utilization on GPU | only if PP improves |
| `-ub` (physical ubatch) | 128, 256, 512 | smaller ubatch reduces peak memory, may allow more `-ngl` on GPU | only if it unlocks more offload |
| `--no-mmap` | on / off | mmap can fault under memory pressure; disabling forces full load up-front | TG variance ↓, PP unchanged |
| `--mlock` | on / off | locks pages, prevents pageout under load | only if --no-mmap shows variance |
| `-ctk`, `-ctv` (KV quant) | f16 / q8_0 / q4_0 | KV quant cuts KV memory 2-4×, may dent quality | KV memory → headroom for longer ctx; quantify TG drop and quality drift |

### Tier-2 sweeps (after Tier-1 picks a winning backend)

| variable | values | applies to | hypothesis |
|---|---|---|---|
| Quant variant | Q4_K_M, Q4_0, Q5_K_M, Q6_K, Q8_0, IQ4_NL/IQ4_XS | all | quality vs throughput trade; Q4 is the comparable-to-others starting point per user direction |
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
| Concurrent streams (`-np N`) | 1, 2, 4 | all | 7B doc shows Vulkan scales 3.0× at N=4; expect MoE to scale differently because expert routing serializes |
| `-sm row` vs `-sm layer` | row / layer | GPU | layer split is default; row may help if expert weights aren't contiguous in VRAM |
| Rope scaling / context extension | YaRN / linear | all | only if we want >native ctx; not a priority for daily-driver |

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

### N-gram lookup decoding (`--lookup`, `--lookup-cache-*`)

llama.cpp has a draft-free spec mode: build an n-gram cache from the
prompt + generated text, propose continuations, verify. **Free win
on repetitive workloads** (code completion, structured-output JSON,
boilerplate generation). Hurts nothing on novel prose because
acceptance just stays at zero.

To try:

```bash
llama-server -m model.gguf --lookup-cache-dynamic <path> --draft 8
```

Measure: with `--lookup` on/off, on (a) a "write a unit test"
prompt and (b) a "continue this story" prompt. The repetitive-code
case should win; the prose case should tie.

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

- Directory created. Recipe drafted. Model downloads not yet run.
- Starting configs locked from the 7B side-quest's canonical wins
  (Q4_0 for GPU, `DISABLE_F16+PREFER_HOST` for Vulkan).
- GGUF source confirmed: **`unsloth/Qwen3.6-35B-A3B-GGUF`** (not
  bartowski; Unsloth is the canonical community quant for this model).
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
