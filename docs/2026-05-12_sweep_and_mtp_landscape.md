# 2026-05-12 backend sweep + MTP / DFlash / PFlash landscape

Companion writeup to the matrix refresh in
`docs/qwen3_4b_baseline_all_backends.md` (see Update log entry
2026-05-12). Three pieces:

1. **Backend rebuild sweep** — llama.cpp `f53577432` → `856c3adac`
   (186 commits), plus a fresh NPU ORT-QNN run on the unchanged
   Qualcomm w4a16 bundle. Per-backend deltas vs the 2026-04-26
   baseline.
2. **Qwen3.6-35B-A3B probe** (new model row). CPU + Adreno OpenCL
   land; Vulkan stalls at the broken-F16 default and can't be coaxed
   into the prior workaround path.
3. **MTP / DFlash / PFlash / vLLM landscape** as of mid-May 2026 —
   what's mainlined, what's in forks, what's CUDA-only, what's worth
   building for this hardware.

Commits this session: still on `master` at the time of writing —
will commit at session close per
`docs/repo_hygiene.md` §"Commit discipline."

---

## 1. Backend rebuild sweep — Qwen3-4B AC matrix

All four llama.cpp variants (`build-{cpu,cpu-kleidiai,opencl,vulkan}`)
rebuilt clean from `856c3adac` via `scripts/build_llama_cpp.ps1`. NPU
side unchanged (same Qualcomm w4a16 Genie bundle, same ORT-QNN
1.24.4, same `npu_engine/` code). System on AC, 40+ GB free RAM at
start.

### Headline deltas (AC, Qwen3-4B)

| backend | runtime | PP (2026-05-12) | PP (2026-04-26) | ΔPP | TG (2026-05-12) | TG (2026-04-26) | ΔTG |
|---|---|---:|---:|---:|---:|---:|---:|
| NPU Genie | genie-t2t-run | **1725.65** | 1566.23 | **+10.2%** | **26.14** | 23.30 | **+12.2%** |
| NPU ORT-QNN | `npu_engine` 4-part chained | **2167.11** | 1985.46 | **+9.2%** | **29.03** | 27.25 | **+6.5%** |
| CPU | build-cpu (-t 8 NEON), Q4_K_M | 174.21 | 188.30 | −7.5% | 39.52 | 39.50 | +0.0% |
| CPU + KleidiAI | build-cpu-kleidiai (-t 8), Q4_K_M | 168.84 | 185.78 | −9.1% | 39.25 | 38.51 | +1.9% |
| GPU OpenCL | -ngl 99, Q4_0 | **588.08 ± 6.94** | 569.12 | +3.3% | 25.27 ± 0.03 | 26.22 | −3.6% |
| GPU Vulkan (HOST_MEM only) | -ngl 99, Q4_0, `GGML_VK_PREFER_HOST_MEMORY=1` | 6.10 | 115.04* | **−94.7%** | **38.01** | 38.51 | −1.3% |
| GPU Vulkan (DISABLE_F16 + HOST_MEM) | both env vars | **CRASH** (0xC0000005) | 115.04 | regression | **CRASH** | 38.51 | regression |

\* 2026-04-26 Vulkan row used both env vars and got PP 115.04 / TG
38.51. That config now segfaults on `856c3adac`.

### Findings

**1. NPU got ~10% better across both runtimes** without any code or
bundle change. Same Qualcomm `.bin` files. Same ORT-QNN. Same
`npu_engine/sidecar.py`. Likely cause: Qualcomm Adreno/Hexagon driver
update or Windows OS update between 2026-04-26 and 2026-05-12 — or
a thermal-headroom delta (ambient temperature is variable on this
laptop). The ORT-QNN advantage over Genie holds steady: +25.6% PP
(was +27%), +11.1% TG (was +17%).

**2. CPU PP regressed ~−8%** while TG is flat. Candidate causes from
the 186-commit window:
- `eff06702b kleidiai : update to v1.24.0` (#22549) — shouldn't
  affect the plain CPU build, but it does refactor `kleidiai.cpp`
  enough that our SME-detect patch script still applies but the
  context around it is different.
- `f08f20a0e ggml-cpu: fuse RMS_NORM + MUL on CPU backend` (#22423)
  — relevant to Qwen3 attention. Probably the suspect for the PP
  drop if anything.
- `bdc9c743a ggml : add sve tuned code for gemm_q8_0_4x8_q8_0()`
  (#21916) — affects activations not weights at Q4_K_M, but the
  whole NEON path may have shifted scheduling.

Not worth bisecting unless the regression deepens; PP-loss of 8% on
a 174 t/s baseline (~3.0 s prefill for 512 tokens) is a 0.2 s wall
hit. TG unchanged means the steady-state cost story is intact.

**3. OpenCL Q4_0 is a wash** (+3% / −3.6%). Five Adreno OpenCL PRs
landed in this window (Q4_0 MoE GEMM #22731, Q4_1 MoE #22856,
F16xF32 prefill GEMM #22755, MoE MXFP4 #22301, Q4_0 refactor
#22335) but the dense Q4_0 path on Qwen3-4B is already saturated;
the MoE wins show up on Qwen3.6 below.

**4. Vulkan regression on the prior workaround** is the headline
bad-news story. Three configs tested:

| config | result | notes |
|---|---|---|
| default env | PP 6.54 / TG 31.91 | broken-F16 path, unchanged from prior default |
| `GGML_VK_PREFER_HOST_MEMORY=1` only | PP 6.10 / TG **38.01** | safe; TG slightly better than prior best |
| `GGML_VK_DISABLE_F16=1 GGML_VK_PREFER_HOST_MEMORY=1` | **STATUS_ACCESS_VIOLATION** | was the prior PP-fix path |
| `GGML_VK_DISABLE_F16=1` alone | **STATUS_ACCESS_VIOLATION** | isolates the regression to `DISABLE_F16` |

The `DISABLE_F16` escape valve is broken — at model load, before any
inference runs. Candidate culprits from the 10-commit Vulkan window:
- `dd9280a66 vulkan: Support asymmetric FA in scalar/mmq/coopmat1`
  (#22589)
- `05e141a6b vulkan: Support asymmetric FA in coopmat2` (#21753)
- `706fbd8ab vulkan: Check shared memory size for mmq shaders`
  (#22693)
- `6d57a49a7 vulkan: fix spv shadowing` (#22760)
- `f9f33654a vulkan: Coalesce Q4_K/Q5_K scale loads` (#21751)

The crash fires after device enum (`uma: 1 | fp16: 0 | bf16: 0 |
warp size: 64 | shared memory: 32768 | int dot: 1 | matrix cores:
KHR_coopmat`) — note `fp16: 0` is the *device-reported* lack of fp16
support. With `DISABLE_F16` set the load-time path must be trying to
allocate something keyed on fp16-disabled but a recent PR's
assumption is "if `KHR_coopmat` is present, fp16 backing is
implicitly there." That's plausibly the asymmetric-FA coopmat
work.

**Recommendation:** file an upstream issue. Until fixed, **Vulkan
TG-only with `GGML_VK_PREFER_HOST_MEMORY=1`** is the new viable
config, but routing prefill through OpenCL Q4_0 + decode through
Vulkan Q4_0 needs an explicit two-stage runner — we don't have one.
For now, treat **OpenCL Q4_0** as the GPU default; Vulkan stays
available only for the TG-dominated workloads where the broken PP
doesn't matter.

CSVs:
- `results/csv/qwen3_4b_baseline_2026-05-12_ac.csv` (main runner)
- `results/csv/qwen3_4b_gpu_q4_0_2026-05-12_ac.csv` (Q4_0 GPU
  reruns + Vulkan env matrix)
- `results/csv/qwen3_4b_ortqnn_2026-05-12_ac.csv` (`npu_engine`
  sidecar bench)

---

## 2. Qwen3.6-35B-A3B baseline probe

First measurements of Qwen3.6-35B-A3B on this hardware. Two quants
already on disk:

- `models/Qwen3.6-35B-A3B-Q4_K_M.gguf` (19.71 GiB)
- `models/Qwen3.6-35B-A3B-MXFP4_MOE.gguf` (20.21 GiB)

Both arches load on the rebuilt binaries — mainline
`856c3adac` registers `LLM_ARCH_QWEN35MOE` and friends. **MTP cannot
be exercised**: the model files we have are stock GGUFs (MTP head
stripped during convert); mainline llama.cpp has no MTP integration
either way (PR #22673 still draft).

### Measured — AC, single stream, r=1, llama-bench

| backend | model | PP512 (t/s) | TG128 (t/s) | notes |
|---|---|---:|---:|---|
| **CPU** | Q4_K_M (19.7 GiB) | **141.51** | **34.19** | -t 8, NEON, plain `build-cpu` |
| CPU | MXFP4_MOE (20.2 GiB) | 145.63 | 31.37 | same; MXFP4 wins PP, loses TG vs Q4_K_M |
| **GPU OpenCL** | MXFP4_MOE | **210.47** | 13.08 | -ngl 99 Adreno; benefits from the recent MoE MXFP4 PR #22301 |
| GPU Vulkan (HOST_MEM) | MXFP4_MOE | stalled | stalled | killed after >5 min stuck on PP; broken-F16 path can't drive 35B MoE prefill at usable throughput |

CSV: `results/csv/qwen3_6_35b_a3b_baseline_2026-05-12_ac.csv`.

### Findings

**1. CPU is the king on 35B-A3B for now.** TG 34.19 t/s on a 35B
parameter model is genuinely strong — this is the A3B (~3B active
per token) MoE structure paying off. On 48 GB unified memory with
228 GB/s (44 GB of which is BIOS-allocated to GPU; the rest stays
addressable from CPU), capacity is a non-issue.

**2. OpenCL eats CPU on PP (+45%) and loses badly on TG (−58%).**
210 vs 145 t/s prefill is the recent Adreno MoE MXFP4 kernel
(#22301) delivering. But 13 t/s decode is the same "Adreno-decode-
is-dispatch-bound-on-MoE" story that already showed up on Qwen3-4B
and Qwen2.5-7B per the prior session writeups — only worse with
A3B's high expert count per token. For 35B-A3B inference, **CPU
is the right TG path; OpenCL is the right PP path; we don't have a
two-stage runner**, so single-backend the choice is CPU.

**3. Vulkan is unusable here.** The broken-F16 default path that
gives Qwen3-4B PP 6.5 t/s would extrapolate to ~0.7 t/s prefill on
35B-A3B if the cost scaled with parameter count — actually less,
since MoE means more dispatched ops, more F16 overhead. The user
observed <1% device utilization with 45 GB resident — symptom of
the load path getting stuck rather than slow inference. Confirms
Vulkan is fully gated by the F16 regression on the new HEAD.

**4. No MTP probe possible today.** Even with mainline llama.cpp's
`qwen35moe` arch loader and our standard Qwen3.6 GGUFs, there's no
MTP-as-self-draft consumption path: the MTP head was dropped during
GGUF conversion. The relevant artifact would be an MTP-preserved
GGUF like
`havenoammo/Qwen3.6-27B-MTP-UD-GGUF`, but mainline llama.cpp
doesn't have the runtime side wired up (PR #22673 still open),
and the only forks that *can* consume MTP today (Indras-Mirror's
fused TBQ4 fork) are CUDA-only. Result: we'll get MTP for free
**once #22673 merges** — no work to do on our side beyond
re-downloading an MTP-preserved GGUF and adding a `--draft-max 4`
(or equivalent) flag to our bench runners.

---

## 3. MTP / DFlash / PFlash / vLLM landscape

### What is MTP?

**Multi-Token Prediction.** The target model carries one or more
*extra* prediction heads alongside the standard LM head, each
trained to predict the next-next token (or next-next-next, etc.).
At inference time:

1. Run the forward pass once.
2. Sample from the primary head AND speculatively sample from each
   MTP head.
3. Verify the MTP-sampled tokens against a re-run of the target
   model on a small batch.
4. Accept the longest verified prefix.

It is *speculative decoding without a separate draft model*. The
draft happens inside the target. This kills the speculative-decode
pain points: no separate VRAM for the draftee, no draft-vs-target
quant mismatch, no tokenizer alignment problem.

Qwen3.6 ships with native MTP heads. DeepSeek-V3 introduced the
technique at production scale; DeepSeek-V4-Flash extends it.
Reported gains in the wild: Qwen3.6-35B-A3B at ~80 t/s on 12 GB
VRAM via MTP + vLLM — vs ~25-35 t/s without (consumer hardware
reports from `dasroot.net` / Medium 2026-05).

**For us:** the *consumer* of MTP is whatever runtime we use. On
this hardware the runtime options that can consume MTP are
listed below; the GGUFs we'd need are publicly available
(`havenoammo/Qwen3.6-27B-MTP-UD-GGUF`,
`havenoammo/Qwen3.6-35B-A3B-MTP-UD-GGUF`, etc.).

### Status table — what runs on this hardware?

| project | what it does | runs on Snapdragon X2? | timeline |
|---|---|---|---|
| **mainline llama.cpp HEAD** | Qwen3.5/3.6 arch loads, std `--spec-draft-model` spec-decode, **no MTP yet** | yes (rebuilt this session) | now |
| llama.cpp PR #22673 (MTP) | adds MTP self-draft consumption | yes, once merged | "soon" — open, not blocked |
| llama.cpp PR #22105 (DFlash) | block-diffusion drafter integration | yes, once merged; blocked behind #18039 (EAGLE3) | weeks/months |
| llama.cpp PR #18039 (EAGLE3) | speculative-decode refactor | yes, once merged | weeks/months |
| llama.cpp `gg/spec-mtp-experiments` branch | Gerganov's MTP experiments | yes, if we cherry-pick the branch | now (experimental) |
| `z-lab/dflash` | reference DFlash impl (Python + vLLM + SGLang + MLX) | **no** — CUDA-only on PC | needs cloud GPU |
| `antirez/llama.cpp-deepseek-v4-flash` | DeepSeek-V4-Flash inference (CPU + Metal) | yes (CPU path) but **wrong model family** | not useful |
| `Indras-Mirror/llama.cpp-mtp` | fused TBQ4 FA + MTP for Qwen3.6 | **no** — every added op is CUDA | needs cloud GPU |
| `croll83/llama.cpp-dgx` | DGX Spark / Blackwell SM 12.1 fork | **no** — Blackwell PTX only | not portable |
| `Luce-Org/.../pflash` | speculative prefill, sm_80+ BSA | **no** — CUDA-only | needs cloud GPU |
| `vLLM` | server runtime, native MTP, native DFlash, AWQ/GPTQ | **no** Windows ARM64 build, no Vulkan/OpenCL/Adreno/Hexagon backend | needs Linux + CUDA |
| `SGLang` | similar to vLLM | **no** — same limitations | needs Linux + CUDA |
| `MLX` | Apple Silicon | no (different silicon) | n/a |
| `MLC-LLM` | TVM-based, multi-backend incl. Vulkan | **maybe** — supports Vulkan; has Qwen3 support; no MTP | worth a sidequest |
| `ONNX Runtime + ORT-QNN` (what `npu_engine` uses) | direct NPU access | yes, in use today | now — but ORT-QNN can't host an MTP self-draft loop trivially |

### Concrete recommendations

**Now (no work; just verify the bench numbers don't change):**
- File the Vulkan `DISABLE_F16` segfault upstream with a clean
  repro. Test on a future `master` weekly until reverted/fixed.
- Watch llama.cpp PR #22673; when it merges, drop in an
  MTP-preserved Qwen3.6 GGUF and re-run the bench with `--draft-max N`
  on top of CPU + OpenCL builds. **This is the path of least
  resistance to a real MTP number on this laptop.**

**Soon (1-2 sessions):**
- Add a Qwen3.6-35B-A3B baseline row to the matrix doc the same way
  Qwen2.5-7B got one (own doc, `docs/qwen3_6_35b_a3b_baseline_all_backends.md`).
- Investigate MLC-LLM Vulkan path on Adreno X2 — it's the closest
  thing to a "drop-in vLLM-class server with non-CUDA backends" and
  has Qwen3.x support. Even without MTP, MLC's Vulkan kernels are
  worth comparing to our llama.cpp Vulkan numbers (which currently
  lose to OpenCL on PP and to NPU on everything).
- Consider running our existing `--spec-draft-model` flag with
  `Qwen3-0.6B-Q8_0.gguf` drafting `Qwen3-4B-Q4_K_M.gguf` on CPU
  build — gives us "what does standard speculative decoding look
  like on this laptop" as a comparison baseline for the future MTP
  number. Should be ~2× TG win in the best case.

**Deferred (cloud-GPU path required):**
- DFlash (z-lab) reference numbers on Qwen3.6-35B-A3B. Rent a
  RTX 4090 / RTX 6000 / A100, run vLLM + DFlash, get the
  literature-comparable t/s number. Goal: a reference for what
  "good MTP" looks like, so we know what to expect when llama.cpp
  PR #22673 lands.
- PFlash for 128K-context prefill. Same cloud-GPU rental. Not
  applicable to this laptop today; useful as a comparison for the
  >4K context routing decision described in
  `docs/one_pipeline_cloud_gpu.md`.

**Skip (do not invest):**
- Building any of the CUDA-only forks on ARM64. Audit details in
  `current_status.md` 2026-05-12.
- Trying to backport MTP support to ORT-QNN. The `npu_engine` ORT-
  QNN stack would have to host a verifier loop and re-issue
  partial graph calls per accepted token — feasible but
  out-of-scope until we know whether MTP gives us a meaningful win
  on this silicon.

### Why we keep waiting on llama.cpp upstream instead of forking

Three reasons:

1. **Reach.** A llama.cpp HEAD that supports MTP supports it
   automatically on CPU / OpenCL / Vulkan / Hexagon — all four of
   our backends. The forks are CUDA-only.
2. **Quant compat.** Our bench infrastructure already runs against
   standard GGUFs. Forks like Indras-Mirror introduce TBQ4 and
   custom KV types that don't exist in our pipeline.
3. **Bisectability.** Mainline lets us check each commit window
   like we just did. Forks make every regression a "two changes
   from now we'll know" situation.

The cost of waiting is wall-clock time; the cost of jumping to a
fork is permanent maintenance debt.
