# specula

Speculative decoding research on Windows-on-ARM. Target hardware: Snapdragon X2 Elite
Extreme, 48 GB LPDDR5X unified memory @ 228 GB/s, Adreno X2 GPU, Hexagon NPU.

*Specula (Latin):* a watchtower, a vantage from which you look ahead.
Exactly what a draft model is.

## Goal

Characterize and push the ceiling of speculative decoding on Windows-on-ARM,
then contribute back upstream. Three concentric ambitions:

1. **Baseline the hardware.** Measure stock llama.cpp speculative decoding across
   every available backend (CPU, Vulkan/Adreno, OpenCL/Adreno, Hexagon NPU) with
   a clean sweep matrix. No such characterization exists publicly for X2 Elite
   Extreme.
2. **Land new speculative techniques on llama.cpp for this hardware.** Validate
   EAGLE-3 (PR #18039) on Adreno/Hexagon. Port DFlash (block-diffusion drafting,
   arXiv:2602.06036) which currently exists only for CUDA and MLX.
3. **Novel placement: draft on NPU, verify on GPU.** No other mainstream platform
   exposes an independently-addressable NPU to user-space inference code. Apple
   Silicon cannot do this. This is the architectural advantage Snapdragon has
   that nobody has exploited yet.

## Hardware assumptions

| Component        | Spec                                               |
|------------------|----------------------------------------------------|
| CPU              | Oryon v2, 18 cores (2 prime + 16 performance)      |
| GPU              | Adreno X2                                          |
| NPU              | Hexagon (v79 generation)                           |
| Memory           | 48 GB LPDDR5X unified, 228 GB/s                    |
| OS               | Windows 11 ARM64 (native — no WSL2/Docker for GPU) |

At 228 GB/s, autoregressive decode of any model >2 GB is firmly memory-bandwidth
bound. That is precisely the regime where speculative decoding pays maximum
dividends: one weight-read amortizes across K verified tokens.

## Why native Windows and not WSL2/Docker

- **WSL2 Vulkan** is dzn (Mesa Vulkan→D3D12 translation). Non-conformant,
  slow for compute, unstable under LLM load. Same problem on NVIDIA in WSL.
- **WSL2 OpenCL** — no Qualcomm ICD in WSL.
- **WSL2 Hexagon** — QNN SDK and FastRPC are Windows-native; no passthrough.
- **Docker Desktop on Windows ARM** uses the WSL2 backend, so same constraints.

WSL2 remains useful for: CPU-only baselines, running reference CUDA-only
implementations (z-lab's DFlash Transformers path) on CPU to understand their
behavior, and Python tooling.

## Progression

### Phase 0 — Infrastructure

- [x] Download initial model set (`scripts/download_models.ps1`)
- [ ] Build llama.cpp native ARM64, per preset:
  - [x] `cpu` — reproduces `LOCAL_LLM_NOTES.md` CPU numbers; sanity check the toolchain
  - [ ] `vulkan` — Adreno via Vulkan backend (Phase 1 GPU baseline)
  - [ ] `opencl` — Adreno via OpenCL backend; blocked on OpenCL SDK install
  - [ ] `vulkan-opencl` — both backends in one binary (once OpenCL is unblocked)
  - [ ] `cpu-kleidiai` — Phase 1 SME2 retry (expected to crash today, see Phase 1)
  - [ ] `hexagon` — out-of-band; needs the Qualcomm toolchain docker image
- [ ] Verify each built backend runs a trivial generation
- [ ] Establish sweep harness that writes structured CSV results

Build invocation:
```powershell
.\scripts\build_llama_cpp.ps1 -Preset cpu
.\scripts\build_llama_cpp.ps1 -Preset vulkan          # needs VULKAN_SDK env var (LunarG installer sets it)
.\scripts\build_llama_cpp.ps1 -Preset opencl          # needs OpenCL headers + OpenCL.lib on disk
.\scripts\build_llama_cpp.ps1 -Preset cpu-kleidiai    # applies the clang-on-Windows KleidiAI .S patch
```
Each preset builds into its own `llama.cpp\build-<preset>\` directory so
configurations coexist. Runtime DLLs (`msvcp140`, `vcruntime140`,
`libomp140.aarch64`) are copied next to the binaries automatically.

### Phase 1 — Autoregressive baselines

For each `{model, quant, backend, threads, ngl}`:
measure prompt-processing tok/s and token-generation tok/s at ctx ∈ {512, 2048, 8192}.

This gives us the ceiling. Every speculative-decoding experiment is measured
as a ratio against the corresponding baseline in this table.

**CPU baseline must match the numbers in `gguf_models/LOCAL_LLM_NOTES.md`**
(e.g. Qwen3-4B Q4_K_M @ 18 threads: ~248 t/s PP / ~42 t/s TG; Qwen3.6-35B-A3B
Q5_K_M with FA + q8 KV: ~145 t/s PP / ~29.6 t/s TG). Anything lower is a
build/config regression, not a hardware result.

**SME2 / KleidiAI retry.** Prior attempt (see `LOCAL_LLM_NOTES.md`) built
llama.cpp with `-DGGML_CPU_KLEIDIAI=ON` cleanly, but its SME2 MOPA kernels
tripped `STATUS_ILLEGAL_INSTRUCTION` on any batched matmul — suspected
Windows-on-ARM64 missing the full SME2 ZA-tile user-mode state (basic SME
smstart/smstop works; ZA-tile extensions do not). The 4096-bit Matrix
Engine is sitting idle until this is unlocked. Tasks:

- [ ] Re-test on current Windows 11 ARM64 build; specifically look for a
  post-2026-04 servicing patch mentioning SME2 user-mode / context-switch
- [ ] Confirm whether the ZA-tile trap is kernel-level (no user-mode enable)
  or KleidiAI-level (kernel ok, compile target wrong)
- [ ] Try `-DGGML_CPU_KLEIDIAI=ON` isolated to TG-only paths first
  (the prior crash was PP batched matmul; TG is single-token and may
  exercise different kernels)
- [ ] If it runs, add SME2 on/off as a column in the baseline sweep so
  we can quote the lift directly
- [ ] Escalate upstream (llama.cpp + KleidiAI) with the exact trap signature
  if we cannot make it work

### Phase 2 — Stock speculative decoding

1. **Draft-model spec** (`llama-speculative`): Qwen3-0.6B → Qwen3-8B/14B,
   sweep `--draft-max ∈ {4, 8, 16, 32}`, `--draft-min ∈ {0, 1, 4}`, temperature=0.
   Record acceptance rate + effective tok/s.
2. **Draftless ngram spec**: `--spec-type` ∈ {`ngram-cache`, `ngram-simple`,
   `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`}. Memory-free win on
   structured output. Quick to run.
3. **Mixed-device placement**: draft on CPU, target on Adreno. Draft on
   Hexagon NPU, target on Adreno. This is novel; llama.cpp supports per-model
   `--device` selection but nobody has benchmarked asymmetric placement on WoA.

Workloads sweep over: HumanEval code completion (high-acceptance regime),
JSON generation (very high acceptance, ngram sweet spot), long prose
(low-acceptance regime), multi-turn chat (mixed).

### Phase 3 — EAGLE-3 validation (priority revised after session 4 + lucebox paper)

Session 4 CPU/OpenCL data shows our 8B+0.6B chain-speculative ceiling is
~1.6×, overhead-bound not accept-bound. The lucebox-hub DFlash+DDTree port
(see `docs/reference-projects.md`, `new_spec_decode_example_to_research.md`)
hits AL ≈ 8.9 and 3.43× on RTX 3090 by stacking *two* axes: a block-
diffusion drafter (higher accept) AND tree-verify (bigger verify batches
per round). EAGLE-3 only touches the accept axis, so it won't break our
ceiling by itself — but it's still cheaper to try first than DFlash, and
tree-verify in EAGLE-3 is informative for the DFlash port.

- [ ] Check out llama.cpp PR #18039 branch
- [ ] **Quick viability build** for CPU + OpenCL (~1 day). Before
  committing to a full port, confirm the PR builds on ARM64 / Adreno
  at all.
- [ ] Validate `GGML_TENSOR_FLAG_SYNC` semantics on non-CUDA backends
  (this is where the port is most likely to silently break)
- [ ] If it builds: one sweep on Phase-2's humaneval fixture at the best
  EAGLE-3 tree-budget. **Decision gate:** if EAGLE-3 clears 2×
  end-to-end, continue the integration work. If it's stuck near our
  1.6× ceiling, archive findings and move to DFlash — the accept-only
  axis won't beat the overhead ceiling.
- [ ] File issues / contribute fixes upstream (build fixes are
  contribution-worthy regardless of our perf outcome)

### Phase 4 — DFlash + DDTree port (primary Phase-3+ lever)

DFlash is the current spec-decode SOTA on acceptance rate: block-diffusion
drafter generates K tokens in parallel in one forward pass. Combined with
DDTree tree-verify, the lucebox-hub port reaches AL ≈ 8.9 and 3.43× on
consumer GPU. No llama.cpp implementation exists.

The session-4 data says this is the phase most likely to actually break
our ceiling — it attacks both the accept-rate axis (K tokens drafted in
one pass) AND the verify-batch axis (tree, not chain), which exactly
matches the two binding constraints we measured.

Reference implementation to mine: `lucebox-hub/dflash/src/` (sibling
checkout, see `docs/reference-projects.md`). ~2000 LOC C++/ggml/CUDA,
MIT-licensed, structured as graph glue that links libggml but not libllama.
Porting plan converts kernels (CUDA → OpenCL/Hexagon) but reuses the graph.

Pieces required:

- [ ] New `LLM_ARCH_DFLASH_QWEN` in `convert_hf_to_gguf.py`
- [ ] GGUF converter for z-lab's drafter weights
  (`z-lab/Qwen3-8B-DFlash-b16` etc.). Note: lucebox-hub ships a
  `safetensors_draft.cpp` loader that skips GGUF entirely — an
  alternative route if the converter proves painful.
- [ ] Block-diffusion forward pass (~4 denoising iterations)
- [ ] Hidden-state tap at configured target layers
  (shared plumbing with EAGLE-3; watch PR #18039)
- [ ] DDTree verifier (tree attention mask, sibling-aware conv gather,
  `target_feat` compaction after sibling accept — see lucebox-hub day-by-day
  log for landmines)
- [ ] KV cache rollback on rejection — pure-attention Qwen3 is just length
  truncation; Qwen3.5 hybrid is harder (tape-replay à la bstnxbt/dflash-mlx)
- [ ] `verify_logits_buf` sized `vocab * (budget + 1)`, not `vocab * q_len`
  — the silent-corruption bug lucebox-hub caught, transcribe it
- [ ] Optional stretch: Q4_0 KV cache on whichever backend we land on,
  to match their 128K-on-24GB story on Adreno (long-context use case)

### Phase 5 — NPU-accelerated drafting (novel)

This is where QAIRT and the Qualcomm cloud compile token earn their keep.
Block-diffusion drafting is a small dense compute burst — perfect NPU workload —
and can overlap with target verify on the GPU.

- [ ] Profile drafter on Hexagon v79 backend (stock)
- [ ] Identify compute-bound kernels in draft path
- [ ] Author QNN custom ops via QAIRT; cross-compile to Hexagon
  (reuse device-targeting info from prior transcription project)
- [ ] Implement async pipelining: NPU drafts block N+1 while GPU verifies N
- [ ] Measure effective throughput against Phase 4 baseline

## Models and quants

All from Qwen's official GGUF repos. Dense models only for Phase 1–2
(MoE comes later to avoid routing-contention confounds on initial data).

| Role               | Model           | Quant   | Size    | Purpose                           |
|--------------------|-----------------|---------|---------|-----------------------------------|
| Draft (primary)    | Qwen3-0.6B      | Q8_0    | 0.64 GB | Default draft, high-fidelity      |
| Draft (alternate)  | Qwen3-1.7B      | Q8_0    | 1.83 GB | Higher acceptance, higher cost    |
| Target (iter)      | Qwen3-8B        | Q4_K_M  | 5.03 GB | Fast iteration                    |
| Target (prod)      | Qwen3-14B       | Q4_K_M  | 9.00 GB | Realistic production size         |
| Target (stretch)   | Qwen3-32B       | Q4_K_M  | ~19 GB  | Scale test (Phase 2+)             |
| Target (MoE)       | Qwen3-30B-A3B   | Q4_K_M  | ~18 GB  | MoE dimension (Phase 4+)          |

**Rationale for Qwen3 (not Qwen3.5):** uniform full attention across all layers.
Qwen3.5 mixes full attention, sliding-window attention, and linear attention
(GatedDeltaNet). KV cache rollback on rejection is trivial for Qwen3 (just
truncate length per layer) and nontrivial for Qwen3.5 (needs per-layer rollback
rules; the MLX DFlash port used a custom tape-replay Metal kernel specifically
for this). Establish clean numbers first, then add complexity.

**Rationale for Q4_K_M target + Q8_0 draft:** Q4_K_M is the standard target
quant, supported cleanly across all backends. For the draft, the extra memory
of Q8_0 is negligible (<1 GB), and higher draft fidelity compounds: +10%
acceptance rate is worth more than any other single knob in the system.

**Rationale for skipping Q4_0 CPU baseline initially:** llama.cpp does online
repacking of Q4_0 for ARM now, but the effective compute win mostly matters for
prompt processing — and speculative decoding doesn't change prompt-processing
behavior, only token generation. Come back to Q4_0 CPU only if ARM-repacked
CPU ever beats Adreno.

## Directory layout

```
specula/
├── README.md                  # this file
├── pyproject.toml             # uv-managed python env
├── .python-version            # 3.12 pin
├── .gitignore
├── .gitattributes
├── docs/
│   └── reference-projects.md  # local sibling projects we can raid per phase
├── scripts/
│   ├── download_models.ps1    # HF GGUF fetcher with resume support
│   ├── build_llama_cpp.ps1    # multi-backend native ARM64 build
│   ├── sweep_baseline.ps1     # Phase 1 autoregressive matrix
│   ├── sweep_speculative.ps1  # Phase 2 spec-decode matrix
│   └── analyze_results.py     # CSV → plots (TODO)
├── prompts/
│   ├── humaneval_subset.jsonl # 10 prompts, code completion
│   ├── structured_json.jsonl  # JSON generation (TODO)
│   ├── prose_longform.jsonl   # low-acceptance workload (TODO)
│   └── chat_multiturn.jsonl   # (TODO)
├── models/                    # GGUFs (gitignored)
├── results/                   # CSVs + logs
├── notebooks/                 # analysis
└── llama.cpp/                 # sibling checkout, version pinned below
```

`llama.cpp` is a sibling git checkout, not a submodule — too much local
patching will happen (EAGLE-3 branch, DFlash in-progress work, custom
Hexagon kernels). The exact upstream commit used for each result row is
recorded in the CSV.

## Tooling

**Native C/C++** — llama.cpp core, eventual DFlash implementation,
custom Hexagon kernels. This is where 80% of the real work lives.

**Python (uv-managed venv)** — HuggingFace model download/conversion,
results analysis, plotting, EAGLE-3/DFlash HF→GGUF converters. Minimal Python
dependency footprint; every script single-file and self-documenting.

**PowerShell** — sweep orchestration, build drivers. Native to Windows ARM,
no shell-emulation layer. Scripts emit structured CSV rows to `results/`.

**QAIRT / QNN C++** — Phase 5 custom NPU kernel authorship. Qualcomm cloud
compile token is for this. Reference: prior transcription project
device-targeting info.

**Zig** — optional. A pure-Zig harness is attractive long-term if we want to
script experiments without touching Python, and a Zig implementation of the
block-diffusion drafter is in scope if we pursue a standalone runtime that
doesn't require llama.cpp's ggml graph machinery.

## Quickstart

```powershell
# clone
git clone <this repo> specula
cd specula

# download models (resumable, ~18 GB total)
.\scripts\download_models.ps1

# clone + build llama.cpp (sibling)
.\scripts\build_llama_cpp.ps1

# python env for analysis
uv venv
.\.venv\Scripts\Activate.ps1
uv sync

# run first baseline
.\scripts\sweep_baseline.ps1 -Backend cpu -Model Qwen3-8B-Q4_K_M.gguf
```

## References

- Local reference projects (trident, voice_project, gguf_models): see
  `docs/reference-projects.md`
- DFlash paper: arXiv:2602.06036
- DFlash reference (CUDA): <https://github.com/z-lab/dflash>
- DFlash MLX port: <https://github.com/bstnxbt/dflash-mlx>
- EAGLE-3 llama.cpp PR: <https://github.com/ggml-org/llama.cpp/pull/18039>
- llama.cpp Snapdragon backend docs: `docs/backend/snapdragon/README.md`
- llama.cpp speculative decoding docs: `docs/speculative.md`
- WoA LLM inference context: llama.cpp discussions #8273, #8336, #8455
