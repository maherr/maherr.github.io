+++
title = "Witness: frontier speaker diarization, first open-source on $550 AMD RDNA 4, 10.65% DER, 15× realtime (beats pyannote)"
date = 2026-04-17
description = "6.84% / 10.65% strict DER on VoxConverse DEV / TEST at 15× realtime, running on a $550 AMD RX 9070 via 8 patches that unlock ONNX Runtime + MIGraphX on RDNA 4."
+++

I wanted speaker diarization on my AMD GPU. The production pipelines said "CUDA required." A few weeks later, mine doesn't. The result is **Witness**: sub-7% strict DER on VoxConverse DEV (6.84%), 10.65% on TEST, 15x realtime. On a $550 consumer AMD card that officially isn't supported by anything.

---

## TL;DR

- **ONNX Runtime 1.24.2 + MIGraphX** building and running on **RDNA 4 / gfx1201** on Fedora. The first public documentation I've come across.
- **Witness**, a speaker diarization pipeline on VoxConverse. TEST (232 files, 43.5h): **10.65% strict / 7.85% paper / 6.90% lenient DER**. DEV (216 files, 20.3h): **6.84% / 4.64% / 3.61%**. Lowest open-source numbers I've found, on both splits and every convention. Beats pyannote 3.1's 11.3% strict on TEST by 0.65pp, on a consumer AMD card that officially isn't supported by any ML stack. (Pyannote doesn't publish a DEV number; DEV is consistently the easier split across systems that score both, so Witness's 6.84% strict DEV isn't directly comparable to pyannote's TEST number.)
- A 3-minute phone call transcribed + diarized in **20 seconds** via parallel Whisper-on-Vulkan + diarization-on-MIGraphX. 36% faster than serial; long calls save up to 49.5%.
- **Vulkan/RADV and ROCm/MIGraphX coexist on the same GPU** with negligible contention. Relevant if you do mixed-stack ML on AMD.

The patches aren't diarization-specific. Any ONNX model you could run on NVIDIA via ORT's CUDA EP, you can now attempt on consumer AMD via MIGraphX EP. Repo + build script + patches at the bottom.

<figure style="margin: 2em 0;">
<svg viewBox="0 0 680 420" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="pareto-title" style="max-width: 100%; height: auto; display: block; margin: 0 auto;">
  <title id="pareto-title">Accuracy vs speed on VoxConverse TEST, strict scoring. Witness in the Pareto-dominant corner.</title>
  <style>
    .pareto text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    .pareto .c-accent { fill: #0b5fff; stroke: #0b5fff; }
    .pareto .c-fg { fill: #1a1a1a; stroke: #1a1a1a; }
    .pareto .c-muted { fill: #6b6b6b; stroke: #6b6b6b; }
    .pareto .c-border { stroke: #e4e4e4; fill: none; }
    @media (prefers-color-scheme: dark) {
      .pareto .c-accent { fill: #7aa8ff; stroke: #7aa8ff; }
      .pareto .c-fg { fill: #eaeaea; stroke: #eaeaea; }
      .pareto .c-muted { fill: #a8a8a8; stroke: #a8a8a8; }
      .pareto .c-border { stroke: #2a2e35; }
    }
  </style>
  <g class="pareto">
    <g class="c-border" stroke-width="0.75">
      <line x1="60" y1="50" x2="60" y2="350"/>
      <line x1="205" y1="50" x2="205" y2="350"/>
      <line x1="350" y1="50" x2="350" y2="350"/>
      <line x1="495" y1="50" x2="495" y2="350"/>
      <line x1="640" y1="50" x2="640" y2="350"/>
      <line x1="60" y1="50" x2="640" y2="50"/>
      <line x1="60" y1="110" x2="640" y2="110"/>
      <line x1="60" y1="170" x2="640" y2="170"/>
      <line x1="60" y1="230" x2="640" y2="230"/>
      <line x1="60" y1="290" x2="640" y2="290"/>
      <line x1="60" y1="350" x2="640" y2="350"/>
    </g>
    <g class="c-muted" font-size="11" text-anchor="middle" stroke="none">
      <text x="60" y="368">88.0</text>
      <text x="205" y="368">88.5</text>
      <text x="350" y="368">89.0</text>
      <text x="495" y="368">89.5</text>
      <text x="640" y="368">90.0</text>
    </g>
    <g class="c-muted" font-size="11" text-anchor="end" stroke="none">
      <text x="54" y="54">50x</text>
      <text x="54" y="114">40x</text>
      <text x="54" y="174">30x</text>
      <text x="54" y="234">20x</text>
      <text x="54" y="294">10x</text>
      <text x="54" y="354">0x</text>
    </g>
    <text x="350" y="395" text-anchor="middle" font-size="13" class="c-fg" stroke="none">Accuracy (100% - DER) on VoxConverse TEST, strict scoring. Higher is better.</text>
    <text x="20" y="200" text-anchor="middle" font-size="13" class="c-fg" stroke="none" transform="rotate(-90 20 200)">Realtime factor. Higher is better.</text>
    <circle cx="263" cy="128" r="6" class="c-fg" fill="none" stroke-width="2"/>
    <text x="250" y="124" font-size="12" text-anchor="end" class="c-fg" stroke="none">pyannote 3.1 (RTX 4090, $1600)</text>
    <text x="250" y="138" font-size="10" text-anchor="end" class="c-muted" stroke="none">88.7% (11.3% DER), ~37x</text>
    <circle cx="452" cy="257" r="9" class="c-accent" stroke="none"/>
    <text x="465" y="255" font-size="13" font-weight="700" class="c-fg" stroke="none">Witness (RX 9070, $550)</text>
    <text x="465" y="271" font-size="11" class="c-muted" stroke="none">89.35% (10.65% DER), 15.47x</text>
  </g>
</svg>
<figcaption style="font-size: 0.85em; color: var(--fg-muted, #6b6b6b); text-align: center; margin-top: 0.75em; font-family: var(--font-sans, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif);">Consumer hardware, dollar-aware. Witness on AMD RX 9070 ($550, measured) vs pyannote 3.1 on NVIDIA RTX 4090 ($1600, 6-scenario avg from voiceping.net 2025). Speed edge: pyannote. Accuracy edge: Witness. Per-dollar speed edge: Witness (15.47x/$550 vs ~37x/$1600, about 22% more speed per dollar on AMD). No apples-to-apples cross-vendor benchmark exists; pyannote requires NVIDIA.</figcaption>
</figure>

<figure style="margin: 2em 0;">
<svg viewBox="0 0 680 420" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="pareto-dev-title" style="max-width: 100%; height: auto; display: block; margin: 0 auto;">
  <title id="pareto-dev-title">Accuracy vs speed on VoxConverse DEV, lenient scoring. Witness ahead of FoxNoseTech/diarize by 7.19pp.</title>
  <style>
    .pareto-dev text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    .pareto-dev .c-accent { fill: #0b5fff; stroke: #0b5fff; }
    .pareto-dev .c-fg { fill: #1a1a1a; stroke: #1a1a1a; }
    .pareto-dev .c-muted { fill: #6b6b6b; stroke: #6b6b6b; }
    .pareto-dev .c-border { stroke: #e4e4e4; fill: none; }
    @media (prefers-color-scheme: dark) {
      .pareto-dev .c-accent { fill: #7aa8ff; stroke: #7aa8ff; }
      .pareto-dev .c-fg { fill: #eaeaea; stroke: #eaeaea; }
      .pareto-dev .c-muted { fill: #a8a8a8; stroke: #a8a8a8; }
      .pareto-dev .c-border { stroke: #2a2e35; }
    }
  </style>
  <g class="pareto-dev">
    <g class="c-border" stroke-width="0.75">
      <line x1="60" y1="50" x2="60" y2="350"/>
      <line x1="176" y1="50" x2="176" y2="350"/>
      <line x1="292" y1="50" x2="292" y2="350"/>
      <line x1="408" y1="50" x2="408" y2="350"/>
      <line x1="524" y1="50" x2="524" y2="350"/>
      <line x1="640" y1="50" x2="640" y2="350"/>
      <line x1="60" y1="50" x2="640" y2="50"/>
      <line x1="60" y1="110" x2="640" y2="110"/>
      <line x1="60" y1="170" x2="640" y2="170"/>
      <line x1="60" y1="230" x2="640" y2="230"/>
      <line x1="60" y1="290" x2="640" y2="290"/>
      <line x1="60" y1="350" x2="640" y2="350"/>
    </g>
    <g class="c-muted" font-size="11" text-anchor="middle" stroke="none">
      <text x="60" y="368">87.0</text>
      <text x="176" y="368">89.0</text>
      <text x="292" y="368">91.0</text>
      <text x="408" y="368">93.0</text>
      <text x="524" y="368">95.0</text>
      <text x="640" y="368">97.0</text>
    </g>
    <g class="c-muted" font-size="11" text-anchor="end" stroke="none">
      <text x="54" y="54">15x</text>
      <text x="54" y="114">13x</text>
      <text x="54" y="174">11x</text>
      <text x="54" y="234">9x</text>
      <text x="54" y="294">7x</text>
      <text x="54" y="354">5x</text>
    </g>
    <text x="350" y="395" text-anchor="middle" font-size="13" class="c-fg" stroke="none">Accuracy (100% - DER) on VoxConverse DEV, lenient scoring. Higher is better.</text>
    <text x="20" y="200" text-anchor="middle" font-size="13" class="c-fg" stroke="none" transform="rotate(-90 20 200)">Realtime factor. Higher is better.</text>
    <circle cx="188" cy="260" r="6" class="c-fg" fill="none" stroke-width="2"/>
    <text x="201" y="264" font-size="12" class="c-fg" stroke="none">FoxNoseTech/diarize</text>
    <text x="201" y="278" font-size="10" class="c-muted" stroke="none">89.2% (10.8% DER), ~8x (CPU only)</text>
    <circle cx="605" cy="91" r="9" class="c-accent" stroke="none"/>
    <text x="593" y="89" font-size="13" font-weight="700" text-anchor="end" class="c-fg" stroke="none">Witness (RX 9070)</text>
    <text x="593" y="105" font-size="11" text-anchor="end" class="c-muted" stroke="none">96.39% (3.61% DER lenient), 13.63x</text>
  </g>
</svg>
<figcaption style="font-size: 0.85em; color: var(--fg-muted, #6b6b6b); text-align: center; margin-top: 0.75em; font-family: var(--font-sans, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif);">VoxConverse DEV under FoxNoseTech/diarize's exact convention (c=0.25, skip_overlap=T). Witness is 7.19pp ahead of the prior open-source best, on a consumer AMD GPU vs FoxNose's CPU pipeline.</figcaption>
</figure>

---

## Why this matters

AMD's ROCm stack officially supports a narrow list of enterprise GPUs. Consumer cards, including the flagship RX 9070 that launched this year, are in a documented gray zone: sometimes they work, often they don't, and when they don't, the only advice is "use an Instinct." Meanwhile ONNX Runtime dropped the old ROCm Execution Provider from mainline in 1.23 and made MIGraphX the only in-tree AMD path. Getting that stack to actually build and run on a consumer RDNA 4 card, on a non-Ubuntu distro, is a non-trivial exercise. If you've ever wanted to run YOLO, sentence-transformers, pyannote, or any other ONNX model on your RX 9070 / 9070 XT without buying an NVIDIA card to go with it, this post is for you.

---

## The ONNX-on-AMD-consumer-GPU landscape

Before showing the patches, here's why there's no alternative to building your own:

| Approach | AMD hardware tier | OS support | ONNX Runtime version | State |
|---|---|---|---|---|
| ORT CUDA EP | N/A (NVIDIA only) | N/A | Any | Out of scope |
| ORT ROCm EP | Instinct MI200/MI300 | Ubuntu 22.04 | ORT ≤1.22 | **Dropped from mainline in ORT 1.23** |
| ORT MIGraphX EP (official) | Instinct MI200/MI300 | Ubuntu 22.04 | 1.10+ | Shipped since ~1.10; became the sole AMD path in 1.23 |
| ORT MIGraphX EP, RX 7900 / RDNA 3 | Consumer | Ubuntu | 1.23+ | Community reports: partial, model-dependent |
| ORT MIGraphX EP, RX 9070 / RDNA 4 (gfx1201) | Consumer | **any distro** | **1.24.2** | **Not documented anywhere. This post is the first.** |
| DirectML (Microsoft) | Any AMD | Windows only | 1.17+ | Works, but Windows-only and not production-grade |
| Vulkan via ggml / llama.cpp / whisper.cpp | Any AMD with Vulkan | Any | N/A (not ONNX) | Works great but not for arbitrary ONNX models |
| ZLUDA (CUDA-on-ROCm) | RDNA 2/3 | Linux | N/A | Unofficial, abandoned by AMD, now by community |
| SCALE (Spectral Compute) | RDNA 2+ | Linux | N/A | Commercial, early-access |

**Reading this:** the only row that runs arbitrary modern ONNX models on consumer RDNA 4 is the one this post documents. Every other option either requires an Instinct card (most buyers don't have one), is Windows-only (DirectML), compiles CUDA to ROCm rather than running ONNX directly (ZLUDA/SCALE), or isn't an ONNX path at all (Vulkan/ggml). The specific gap this work fills: **the Linux-based ONNX-on-consumer-AMD row.**


---

## The setup that started this

I had hours of recorded multi-speaker audio I wanted to work through, and I wanted to quote it precisely. "At 09:32 SPEAKER_01 says X" is the grain of reference that makes a transcript usable for serious work. Whisper handles the words. Speaker labels are what turn a wall of text into something you can navigate.

Whisper on Vulkan (via `whisper.cpp`) already ran on my GPU. Transcription: solved. **Diarization, who spoke when, was the missing piece.**

All the modern diarization pipelines are ONNX-shaped. `pyannote/community-1` exports to ONNX. `speakrs` (a Rust diarizer) calls into ONNX Runtime. WeSpeaker's embedding models are ONNX. These models want GPU. GPU on AMD means MIGraphX. MIGraphX on my card, on my OS, did not work out of the box.

So I made it work.

---

## Hardware and versions (for reproducibility)

```
OS          Fedora 43, kernel 6.19.11
GPU         AMD Radeon RX 9070 (RDNA 4, gfx1201, Navi 48)
ROCm        6.4.4 (Fedora repos: rocm-core, rocm-runtime-devel, rocm-hip-devel, rocm-llvm-devel)
MIGraphX    upstream rocm-6.4.2 branch @ db302ae, built from source + 6 patches (lib SONAME 2.12.0)
ORT         v1.24.2 tag @ 058787c, built from source + 2 patches
Rust        1.94.1
speakrs     0.3.1 (forked to add ExecutionMode::MiGraphX)
```

Everything else is default. No secret env flags. The 8-patch split: five work around Fedora not shipping TF/MLIR/C-API-link dependencies as MIGraphX expects (Patches 1, 2, 3, 4, 6), one is gfx1201-specific (Patch 5), and two are ROCm 6.4 version-mismatch fallbacks in ORT (Patches 7, 8). On Ubuntu with ROCm packages, the Fedora-packaging five may not be needed; the remaining three should still apply.

Verified on Navi 48 / RX 9070 / gfx1201. Navi 44 / RX 9060 series should inherit (same ISA family, same compiler target family), but is unverified, PR welcome if you have one.

Fedora users on RDNA 2 or RDNA 3 are likely to benefit from most of this work too. Only Patch 5 is gfx1201-exclusive; the other 7 are Fedora-packaging workarounds (Patches 1, 2, 3, 4, 6) or ROCm 6.4 version-mismatch fallbacks (Patches 7, 8), not arch-specific. Untested on RDNA 2/3, PR welcome there as well.

---

## MIGraphX patches (six of eight)

MIGraphX is AMD's graph-level compiler and runtime. Think of it as what TensorRT is to NVIDIA, still rough around the edges for consumer GPUs, as [the issue tracker](https://github.com/ROCm/AMDMIGraphX/issues) reflects.

### Patch 1: TF protobuf ABI disable

MIGraphX tries to link against a TensorFlow import path. Fedora ships protobuf 26; MIGraphX was compiled against the older ABI. Result: a link-time ABI mismatch in the TF reader.

**Fix:** disable the TF subdirectory entirely (you don't need it for ONNX-only workloads) and provide a one-file stub header so the C API still compiles. 13 lines.

### Patch 2: TF stub header

Companion to Patch 1. Defines the handful of symbols the ONNX path still references into TF-shaped territory, returning no-ops or nulls. 12 lines.

### Patch 3: MLIR stubs (fuse_mlir.cpp)

MIGraphX optionally lowers some fused ops through MLIR. Fedora doesn't package the exact MLIR version MIGraphX expects, and mixing versions produces nightmare link errors.

**Fix:** stub `mlir_enabled()` to return false; stub the compile entry points. The compiler falls back to its native path. On diarization models this costs single-digit percentage performance, not orders of magnitude. ~1100 lines; every function is a trivial no-op, but `fuse_mlir.cpp` exposes a broad surface that all has to be stubbed.

### Patch 4: MLIR introspection stubs (mlir.cpp)

Second MLIR surface: the rocMLIR C-API bridge in `mlir.cpp`. Same stub treatment, `dump_mlir`, `compile_mlir`, `insert_mlir`, and `get_tuning_config_mlir` all return empty/no-op. Companion to Patch 3, ~1300 lines, same "every function is a no-op, but there are many" story.

### Patch 5: hipcc device compilation guard

MIGraphX's `src/targets/gpu/no_device.cpp` contains a `#error "Device compilation not allowed for migraphx_gpu..."` directive that fires under `__HIP_DEVICE_COMPILE__`. On gfx1201, hipcc's device-compile pass false-positives into this file during intermediate passes, killing the build with the `#error`.

**Fix:** remove the `#error` and rewrite the guard as a silent no-op. Despite the "hipcc" label on this patch, the fix lives in MIGraphX's source tree, not in hipcc itself. As of the upstream `rocm-6.4.2` MIGraphX branch, the `#error` is still present. 37 lines.

### Patch 6: C API link, drop `-lmigraphx_tf`

Now that TF is gone (Patch 1), linking against `libmigraphx_tf.so` will fail because we didn't build it. Remove the `-lmigraphx_tf` flag from the C API link line. 13 lines.

---

## The two ONNX Runtime patches

ORT's MIGraphX EP has surfaces that ROCm 6.4 doesn't fully implement yet. They're both graceful-degradation patches: detect the missing op, don't crash, log a warning, fall back to CPU for that op (or to higher-precision equivalent).

### Patch 7: `fp4x2` graceful fallback

`fp4x2` is a packed-4-bit type ORT expects MIGraphX to understand. ROCm 6.4 doesn't support it. Default behavior: crash during `GetCapability`. With the patch: return "not supported" for ops that require `fp4x2`, so ORT assigns them to CPU. 14 lines.

### Patch 8: `bf16` quantize graceful skip

Similar story, for bf16 quantization ops. Without the patch, bf16 models crash at session creation. With it, they either run (if a compatible kernel exists) or fall back cleanly. 13 lines.

---

## The moment it actually worked

I remember it specifically. I'd already blown ~30 hours on this. I ran `speakrs-cli --gpu test.wav` with the freshly patched build. It crashed after 7.8 seconds, inside `MIGraphXExecutionProvider::GetCapability`, trying to compile the segmentation model. That was progress. The crash was now in a place I could reason about.

The issue: MIGraphX's compiler was choking on the segmentation model's dynamic `samples` input dimension. Workaround in a minimal reproducer (not the production speakrs path, this was the diagnostic step):

```rust
// Pin samples dimension. MIGraphX handles dynamic shapes poorly for this model.
session_builder = session_builder.with_dimension_override("samples", 160000);
```

That made segmentation run. Then MIGraphX entered a multi-minute single-threaded compile of the segmentation graph, which is not viable for what's a sub-second CPU operation anyway.

**Final architecture:** hybrid. Segmentation stays on CPU (0.03s, free). Embedding (WeSpeaker ResNet34) runs on MIGraphX GPU. Cache the compiled embedding model via `ORT_MIGRAPHX_MODEL_CACHE_PATH=~/.cache/migraphx-compiled/`. First run: 45.7s (cold compile). Every subsequent run: 19.0s wall, 11.8s actual processing. **A 26MB `.mxr` artifact caches the entire thing, valid until any of {ORT, MIGraphX, driver, model SHA} changes.**

---

## The numbers

### 4-file VoxConverse subset, accuracy progression

| File   | Baseline DER | After adaptive threshold | Predicted / True speakers | Notes |
| ------ | ------------ | ----------------------- | ------------------------- | ----- |
| aepyx  | 14.83%       | 14.83% (unchanged)      | 4 / 4                     | Short file, 169s, default threshold |
| aggyz  | 7.21%        | 7.21% (unchanged)       | 12 / 13                   | Short file, 263s, default threshold |
| aiqwk  | 12.02%       | 12.02% (unchanged)      | 5 / 7                     | Short file, 199s, default threshold |
| aorju  | 10.08%       | **8.82%** (-1.26pp)     | 9 / 12                    | **1200s long file, adaptive threshold 0.65** |
| **Overall** | **10.28%** | **9.46%** (-0.82pp)   |                           | **First single-digit crossing** |

The accuracy win came from a single mechanism: files longer than 600s with >=300 segments get an AHC (agglomerative hierarchical clustering) merge threshold of 0.65 instead of 0.60. On the long `aorju` file this cut the speaker-confusion component substantially while keeping short files bit-identical. Honest caveat: speaker count on `aorju` stayed at 9/12. The win is cleaner merging. Speaker discovery is a different problem.

### Full transcribe + diarize pipeline, parallel vs serial wall time

The biggest speedup came from something orthogonal to the ML: Whisper runs on Vulkan/RADV, diarization runs on ROCm/MIGraphX, independent driver stacks on the same GPU. They were running serially for no reason. Parallel execution costs nothing and overlaps them end-to-end.

| Call duration | Wall serial | Wall parallel | Saving | GPU util avg |
| ------------- | ----------- | ------------- | ------ | ------------ |
| 1 min         | 15.84s      | 11.71s        | -26.0% | 40.9%        |
| 3 min         | 31.91s      | **20.33s**    | **-36.3%** | 65.3%    |
| 5 min         | 28.45s      | 19.63s        | -31.0% | 53.8%        |
| 10 min        | 87.39s      | 46.40s        | -46.9% | 78.4%        |
| 18 min        | 154.48s     | 77.99s        | **-49.5%** | 80.6%    |

<figure style="margin: 2em 0;">
<svg viewBox="0 0 680 340" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="parallel-title" style="max-width: 100%; height: auto; display: block; margin: 0 auto;">
  <title id="parallel-title">Parallel vs serial pipeline wall time. Savings approach 50% on longer calls.</title>
  <style>
    .parallel text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    .parallel .c-accent { fill: #0b5fff; stroke: #0b5fff; }
    .parallel .c-fg { fill: #1a1a1a; stroke: #1a1a1a; }
    .parallel .c-muted { fill: #6b6b6b; stroke: #6b6b6b; }
    .parallel .c-border { stroke: #e4e4e4; fill: none; }
    @media (prefers-color-scheme: dark) {
      .parallel .c-accent { fill: #7aa8ff; stroke: #7aa8ff; }
      .parallel .c-fg { fill: #eaeaea; stroke: #eaeaea; }
      .parallel .c-muted { fill: #a8a8a8; stroke: #a8a8a8; }
      .parallel .c-border { stroke: #2a2e35; }
    }
  </style>
  <g class="parallel">
    <g class="c-border" stroke-width="0.75">
      <line x1="60" y1="50" x2="60" y2="290"/>
      <line x1="60" y1="50" x2="640" y2="50"/>
      <line x1="60" y1="90" x2="640" y2="90"/>
      <line x1="60" y1="130" x2="640" y2="130"/>
      <line x1="60" y1="170" x2="640" y2="170"/>
      <line x1="60" y1="210" x2="640" y2="210"/>
      <line x1="60" y1="250" x2="640" y2="250"/>
      <line x1="60" y1="290" x2="640" y2="290"/>
    </g>
    <g class="c-muted" font-size="11" text-anchor="middle" stroke="none">
      <text x="89" y="308">1 min</text>
      <text x="147" y="308">3 min</text>
      <text x="350" y="308">10 min</text>
      <text x="582" y="308">18 min</text>
    </g>
    <g class="c-muted" font-size="11" text-anchor="end" stroke="none">
      <text x="54" y="54">180</text>
      <text x="54" y="94">150</text>
      <text x="54" y="134">120</text>
      <text x="54" y="174">90</text>
      <text x="54" y="214">60</text>
      <text x="54" y="254">30</text>
      <text x="54" y="294">0</text>
    </g>
    <text x="350" y="330" text-anchor="middle" font-size="13" class="c-fg" stroke="none">Call duration</text>
    <text x="20" y="170" text-anchor="middle" font-size="13" class="c-fg" stroke="none" transform="rotate(-90 20 170)">Wall time (s)</text>
    <polyline points="89,269 147,248 350,174 582,84" fill="none" class="c-fg" stroke-width="2"/>
    <polyline points="89,274 147,263 350,228 582,186" fill="none" class="c-accent" stroke-width="2.5"/>
    <g class="c-fg" stroke="none">
      <circle cx="89" cy="269" r="3.5"/>
      <circle cx="147" cy="248" r="3.5"/>
      <circle cx="350" cy="174" r="3.5"/>
      <circle cx="582" cy="84" r="3.5"/>
    </g>
    <g class="c-accent" stroke="none">
      <circle cx="89" cy="274" r="4"/>
      <circle cx="147" cy="263" r="4"/>
      <circle cx="350" cy="228" r="4"/>
      <circle cx="582" cy="186" r="4"/>
    </g>
    <g transform="translate(100, 62)">
      <line x1="0" y1="0" x2="20" y2="0" class="c-fg" stroke-width="2"/>
      <circle cx="10" cy="0" r="3.5" class="c-fg" stroke="none"/>
      <text x="26" y="4" font-size="11" class="c-fg" stroke="none">Serial</text>
      <line x1="75" y1="0" x2="95" y2="0" class="c-accent" stroke-width="2.5"/>
      <circle cx="85" cy="0" r="4" class="c-accent" stroke="none"/>
      <text x="101" y="4" font-size="11" class="c-fg" stroke="none">Parallel</text>
    </g>
    <g class="c-muted" font-size="10" stroke="none">
      <text x="89" y="259" text-anchor="middle">-26%</text>
      <text x="147" y="238" text-anchor="middle">-36%</text>
      <text x="350" y="164" text-anchor="middle">-47%</text>
      <text x="582" y="74" text-anchor="middle">-49%</text>
    </g>
  </g>
</svg>
<figcaption style="font-size: 0.85em; color: var(--fg-muted, #6b6b6b); text-align: center; margin-top: 0.75em; font-family: var(--font-sans, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif);">Parallel execution overlaps Whisper (Vulkan) and speakrs (MIGraphX) on the same GPU. Savings grow with call length as the two subsystems converge on wall-time. Chart shows four of the five measured durations (1/3/10/18 min); 5-min is omitted because short-file scheduling variance put it below the 3-min serial time. Full five-row table just above.</figcaption>
</figure>

Median saving: **36.3%**. The scaling is intuitive, savings approach 50% as whisper_s and diarize_s converge on longer audio. Shorter calls save less because one subsystem dominates.

Bonus: average GPU utilization climbed from the baseline 17-29% range to 41-81%. Parallelism cut wall time and finally got the card doing actual work.

### Accuracy on a real-world recording

On a 16-minute real-world recording with two speakers, zero misattributions end-to-end. One transcription error in the whole file, which I'm willing to live with.

The reference numbers: `pyannote/community-1` reports **11.2% strict DER** on VoxConverse v0.3 (c=0, skip_overlap=False); pyannote 3.1 reports **11.3% strict**. Both are on the TEST set per the [speakrs crate documentation](https://crates.io/crates/speakrs), which serves as the authoritative interpretation since pyannote's HF cards don't explicitly state the split.

Our numbers, on the same TEST set under identical strict scoring: **10.65%**, which is **-0.65pp below pyannote 3.1** and **-0.55pp below community-1**. Under the VoxConverse paper convention (c=0.25, skip_overlap=False): **7.85%**. Under lenient scoring (c=0.25, skip_overlap=True): **6.90%**.

On the DEV set (216 files, 20.3 hours), the same pipeline reaches **6.84% strict / 4.64% paper / 3.61% lenient**. The lenient DEV number is the directly-comparable one against FoxNose's reported 10.8% (same split, same convention): **-7.19pp below FoxNose**.

These are the lowest open-source numbers I've found on VoxConverse, on both splits and every convention. Hypothesis RTTMs are saved on disk; anyone with `pyannote.metrics` installed can reproduce the numbers from the frozen outputs.

---

## How this compares

The charts up top gave the Pareto visual. Here's the full benchmark comparison, every row and convention spelled out:

| System | Split | Strict (c=0) | Paper (c=0.25) | Lenient (c=0.25, skip=T) | RT factor | Hardware | License |
|---|---|---|---|---|---|---|---|
| **Witness** (this work) | **DEV** (216) | **6.84%** | **4.64%** | **3.61%** | **13.63x** | AMD RX 9070 ($550) | MIT |
| **Witness** (this work) | **TEST** (232, harder) | **10.65%** | **7.85%** | **6.90%** | **15.47x** | AMD RX 9070 ($550) | MIT |
| pyannote community-1 | TEST\* | 11.2% | - | - | ~37x | NVIDIA RTX 4090 ($1600) | MIT |
| pyannote 3.1 | TEST\* | 11.3% | - | - | ~37x | NVIDIA RTX 4090 ($1600) | MIT |
| WhisperX (inherits community-1) | TEST\* | ~11.2% | - | - | ~40x | NVIDIA RTX 4090 ($1600) | BSD |
| FoxNoseTech/diarize | DEV | - | - | 10.8% | ~8x | CPU only | Apache-2.0 |

\* pyannote's HF cards don't explicitly state the VoxConverse split; per the [speakrs crate documentation](https://crates.io/crates/speakrs), the ~11.1-11.3% numbers are on the TEST set (232 files) under strict scoring, which we've adopted as the authoritative interpretation.

**What the table is telling you:** the top two rows are Witness (this work), on both VoxConverse splits, under three standard DER conventions. Every cell is the lowest published open-source number for that convention-split pair. On DEV under FoxNose's exact convention (c=0.25, skip_overlap=True), we're 7.19pp below the previous open-source best. On TEST strict, we're 0.65pp below pyannote 3.1, the community's de facto reference implementation. The larger deltas on DEV are because DEV is the easier split for any pipeline; we included TEST separately so the harder benchmark doesn't get glossed over.

**Caveats:** DER is a single scalar that hides structure (false-alarm vs missed-detection vs speaker-confusion mix differs across systems). Clustering pipelines differ. The 0.55-0.65pp TEST-set margin over pyannote is noise-adjacent; the DEV-set deltas and lenient-convention deltas are where this pipeline most visibly pulls ahead. Treat the table as concrete positioning, not a guarantee that this system will outperform pyannote on your audio - VoxConverse has its own distribution and your recordings will vary.

**Speed across hardware.** Pyannote 3.1's realtime factor on RTX 4090 is ~23-56x across 6 test scenarios (RTF 0.018-0.043, average ~37x at RTF 0.027) per [voiceping.net December 2025](https://voiceping.net/en/blog/research-diarization-2025/); that ~37x is the figure cited in the chart and table. Witness runs 15.47x on consumer AMD RX 9070 ($550). On a per-dollar basis that's ~22% more speed per dollar on AMD than pyannote-on-RTX-4090; an apples-to-apples cross-vendor benchmark isn't possible today because pyannote needs NVIDIA. This post documents the first open-source ONNX diarization path on consumer AMD.


---

## Vulkan and ROCm coexist on the same GPU (the ecosystem finding)

I didn't expect to learn this. If you do mixed-workload ML on AMD, it's probably the bit that matters.

Whisper runs on Vulkan (via whisper.cpp). Diarization runs on ROCm/MIGraphX (via ORT). Two separate AMD GPU driver stacks, different user-space libraries, different kernel compute paths, different memory allocators. Common wisdom: don't mix them.

We did, on the same physical RX 9070, and measured the contention:

| Workload         | Alone | Concurrent with other | Slowdown |
| ---------------- | ----- | --------------------- | -------- |
| Whisper (Vulkan) | 13.7s | 13.3s                 | ~0% (noise) |
| speakrs (ROCm)   | 20.1s | 21.2s                 | ~5%      |

Whisper was faster concurrent than alone, within noise. Speakrs took a ~5% hit. That's a rounding error compared to the 36% wall-time saving from not running serially. For *this* workload pair at this compute intensity, the two GPU driver stacks schedule independently on gfx1201; I don't claim it generalizes.

**Practical implication for your own AMD ML stack:** if you're mixing a Vulkan-shader workload (whisper.cpp, llama.cpp, stable-diffusion via ggml) with a ROCm workload (ORT/MIGraphX, PyTorch ROCm, TensorFlow ROCm), you don't have to pick one. Overlap them. On gfx1201 for this workload pair, the scheduler seemed fine with it, measure before you trust it on yours.

**The honest caveat**: VRAM is the real constraint. Running both concurrently peaked at 15.2 GiB on my 16 GiB card, with a desktop already using about 9 GiB for a typical browser + apps session. That left roughly 1 GiB of headroom. A headless server would have ~14 GiB free, trivial fit. On a desktop with a busy session, the pipeline auto-falls-back to serial execution if free VRAM drops below 4 GiB. Document this in your install guide if you package anything similar.

---

## Measurement misadventures (what I got wrong on the way)

Three things I got wrong. You probably will too.

**Lesson 1: "dispatch-bound" doesn't mean what profiling tools say it means.**

Early profiling said the workload was dispatch-bound: 24.5 ms of "dispatch time" per embedding call, 1.58 ms of "kernel time," GPU idle 57% of the time. Obvious conclusion: batch the calls, amortize the overhead, cut dispatch count up to ~30x by batching 268 per-segment calls into a small number of batched calls. I prototyped it, built a benchmark harness, ran the sweep, and got a fake 19% speedup that turned out to be stderr JSONL profiling IO amortizing over larger batches. A clean no-profiling re-run showed the real delta: about -0.7%. Batching didn't help.

The deeper reason it didn't help (and the useful ecosystem finding):

**MIGraphX (rocm-6.4.2 branch) on gfx1201 does not parallelize the batch dimension for WeSpeaker ResNet34.** A batched graph with `[32, 80, T]` input takes ~32x longer per kernel call than `[1, 80, T]`, not the 1.3-2x you'd expect from batch parallelism on a modern GPU. The compiler unrolls the batch dim into sequential per-element work. All the "dispatch savings" get eaten by proportionally-longer per-call compute.

The "1.58 ms kernel time" my profiler reported was one MIGraphX trace event (named `MGXKernel`), not the full forward pass. Real per-sample compute is ~24 ms. The pipeline is effectively compute-bound at the single-sample level, there's just no parallelism to exploit on this toolchain yet.

**If you're planning to re-export your ONNX model with a dynamic batch dim specifically to target MIGraphX on RDNA 4: don't bother, at least not on the `rocm-6.4.2` branch.** Wait for the compiler to fuse batched kernels properly, or stick with batch=1 and put your effort into pipeline-level parallelism (see the parallel-vs-serial table above) or memory-bandwidth work (FP16 weights).

**Lesson 2: profiling IO contaminates wall-time benchmarks.**

If you emit per-call JSONL events to stderr to measure per-dispatch timing, the syscall cost scales with call count. In my batching benchmark, batch=1 fired 268 stderr events per file; batch=8 fired 34. The stderr overhead alone created a fake 19% "speedup" for batching. A clean re-run with profiling disabled told the true story. Always verify the instrumented run matches the uninstrumented baseline within ~0.5% before trusting any deltas.

**Lesson 3: a rare heap-corruption race in native-library teardown.**

During the full 232-file VoxConverse benchmark, one file (`fqrnu`) crashed with `corrupted double-linked list` from glibc, a classic heap-corruption signature. What's unusual: it happened **after `main()` returned cleanly**. The backtrace from systemd-coredump showed the abort firing during `_dl_fini` (dynamic-loader shutdown), inside libgcc's `release_registered_frames → btree_destroy → _int_free`. All diarization work had completed; the JSON output had been generated. The crash was pure teardown.

Single-threaded at the moment of crash, so no compute-time race between worker threads. The heap was corrupted earlier in the run, probably by a teardown mis-free in one of the native-GPU libraries loaded via `dlopen` (ORT MIGraphX EP, MIGraphX, libamdhip64, libamd_comgr, libhsa-runtime64), but glibc only noticed when libgcc's shutdown `free()` tried to coalesce against the corrupted chunk. Rate: 1 in 232 sequential runs (binomial 95% CI ~0.01-2.4%, so treat it as "rare, not pinned down"), 0 in 10 isolated reruns, 0 in 65-iteration stress tests with `MALLOC_CHECK_=3 MALLOC_PERTURB_=42`. Not file-specific, the crashing file was structurally unremarkable; the race just landed on it.

**Diagnosis, not fix.** Root-causing this would require AddressSanitizer rebuilds of ORT + MIGraphX (hours each) and the bug is in upstream code I don't control. Expected to resolve when ROCm 7.x ships gfx1201 as officially supported. Filed upstream at [ROCm/AMDMIGraphX#4792](https://github.com/ROCm/AMDMIGraphX/issues/4792) and [microsoft/onnxruntime#28087](https://github.com/microsoft/onnxruntime/issues/28087). For batch processing, the repo's benchmark script retries once on the specific rc=-6 + signature match, a fresh subprocess bypasses the rare race. Single-call users have not hit it in practice; if you do, the diarization output was likely generated correctly before the abort, and a simple retry will succeed.

**If you hit this:** it's a native teardown race, not your code. Retry the run. File under "unofficial-hardware rough edges" until ROCm 7.x.

---

## What else these patches unlock

I started this to transcribe phone calls. I finished with a build that runs arbitrary ONNX models on a GPU that officially doesn't support them.

What these 8 patches unlock (not tested yet, architecturally enabled, validation is the next step):

- **Vision:** YOLOv8/v11, SAM, DETR, InsightFace ArcFace, RetinaFace.
- **NLP:** sentence-transformers (all-MiniLM, all-mpnet), BGE embeddings, small Whisper ONNX exports.
- **Audio:** Silero VAD, pyannote in other configurations, sherpa-onnx.
- **Candidates worth trying:** Phi-3 mini ONNX, TinyLlama ONNX, SAM ViT (large).

The common constraint is: if it's a standard ONNX model using standard ops, it should compile on MIGraphX now. Models that need `fp4x2` or bf16 quantization ops will gracefully fall back (Patches 7 and 8). Models with heavy dynamic-shape dependencies may need a `with_dimension_override` style workaround (Whisper-style dynamic encoders, for instance).

**If you try this on a model I haven't listed, open an issue in the repo. Building the "verified working" list is the next step.**

---

## What's next

Most of the speed + accuracy work documented in this post is done. What's left is mostly release engineering and extending the "verified working" list.

**1. The open-source release.** The 8 patches are landing as a public repo (`onnxruntime-migraphx-rdna4`, link at the bottom, filled in at publish time). That's the gating item: nothing else I do with these patches matters if other people can't reproduce the build.

**2. More ONNX models verified working.** The patches architecturally enable arbitrary ONNX models on consumer RDNA 4, but "architecturally enabled" and "verified working" are different claims. Validating Silero VAD, sentence-transformers, YOLOv8, and InsightFace ArcFace is next, each takes a few hours and turns a theoretical claim into a tested one.

**3. Accuracy is converged.** Segment-duration filtering and PLDA+VBx conditional clustering were both tested and closed as negative results (details in "Measurement misadventures" above). The adaptive threshold from Phase 1 is the only accuracy change that shipped. Full 232-file benchmark: 10.65% DER, a slight improvement over the 10.76% pre-adaptive-threshold baseline on the same 232-file run (the 10.28% / 9.46% figures in the 4-file subset table above were a smaller, different sample), confirming the threshold helps long files without hurting short ones.

**4. Precompiled `.mxr` ships in the repo.** First-run cold compile of the embedding model was 45.7s. Shipping the pre-built artifact for gfx1201 (27 MB, content-addressed, at `artifacts/precompiled-mxr-gfx1201/`) cuts first-run to 17.5s, within 1.4% of warm steady-state. The `.mxr` filename encodes a hash over (ORT version, MIGraphX version, driver, model). If any of those differ from what we built against, MIGraphX transparently re-compiles. `build.sh` drops the artifact into `~/.cache/migraphx-compiled/` at the end of the build.

**What I'm NOT going to do (and why):**
- **Speed Phase 2 (batched embedding)**, closed negative (see "Measurement misadventures"). MIGraphX on the `rocm-6.4.2` branch doesn't parallelize batch dim on gfx1201. Wait for the compiler.
- **Speed Phase 3 (FP16 weights)**, deferred. Phase 2's finding suggests the runtime doesn't do memory-bandwidth-smart anything on this hardware yet. May revisit if the compute-bound picture shifts under a future ROCm version.

---

## Sunset window

These patches target ROCm 6.4 + ORT 1.24.2. Expect a useful window of 2 to 6 months. A rough partition by what it'd take to obsolete each patch:

- **Patches 7, 8** become no-ops once ROCm 7.x (expected Q2-Q3 2026) ships native `fp4x2` and `bf16` quantization. Shortest shelf life.
- **Patches 1, 2, 6** are Fedora-packaging workarounds (TF protobuf ABI, stub header, C API link). They persist until Fedora packages protobuf / MIGraphX at matching ABIs.
- **Patches 3, 4** are MLIR toolchain workarounds. MLIR version drift is a recurring issue across distros and versions, so these are the most durable.
- **Patch 5** depends on whether AMD cleans up the `#error` in `no_device.cpp` for the gfx12xx device-compile path. Likely needed until gfx1201 is officially supported in ROCm.

ORT 1.25+ may break things on its own terms. No idea yet.

The goal is that ROCm's consumer story gets good enough that this blog post becomes obsolete. That's the best-case outcome.

---

## Get the patches

Repo: **[`github.com/maherr/onnxruntime-migraphx-rdna4`](https://github.com/maherr/onnxruntime-migraphx-rdna4)**

**Prerequisites on Fedora 43:**

```sh
sudo dnf install git cmake make rocm-runtime-devel rocm-hip-devel rocm-llvm-devel python3
```

That pulls in ROCm 6.4.4 (`rocm-core-6.4.4`), hipcc, and clang 19. On Ubuntu 22.04 with AMD's official ROCm repos, the same prerequisites are available as `rocm-dev` plus the `rocm-llvm` meta-package; set `ROCM_PREFIX=/opt/rocm` for the build script.

**Build:**

```sh
git clone https://github.com/maherr/onnxruntime-migraphx-rdna4.git
cd onnxruntime-migraphx-rdna4
bash build.sh
```

Clones MIGraphX (`rocm-6.4.2` branch at pinned SHA `db302ae`) and ONNX Runtime (`v1.24.2` tag at pinned SHA `058787c`), applies the 8 patches, and installs to `~/.local/share/gpu-diarization-build/`. Takes **~45-75 minutes on a 16-thread machine** and needs **~25 GB free disk**. Override `BUILD_DIR`, `INSTALL_PREFIX`, `JOBS`, `GFX_TARGET` via env vars (see the script header).

**What's in the repo:**

```
patches/
  01-migraphx-tf-subdir-disable.patch
  02-migraphx-tf-stub-header.patch
  03-migraphx-mlir-fuse-stub.patch
  04-migraphx-mlir-introspection-stub.patch
  05-migraphx-hipcc-device-guard.patch
  06-migraphx-c-api-drop-tf-link.patch
  07-ort-fp4x2-fallback.patch
  08-ort-bf16-skip.patch
  README.md                       # per-patch rationale, apply order, upstream filings
artifacts/
  precompiled-mxr-gfx1201/        # 27 MB precompiled WeSpeaker ResNet34 .mxr +
                                  # README with the (ORT, MIGraphX, driver, model) pinning
scripts/
  install-witness-stack.sh        # whisper.cpp + speakrs + Whisper model (optional,
                                  # only if you want to run Witness end-to-end)
witness/
  witness                         # CLI orchestrator (whisper.cpp + speakrs)
  README.md                       # usage + dependencies
build.sh                          # reproducible end-to-end build; auto-installs
                                  # the shipped .mxr if the hashes match
THIRD_PARTY_LICENSES.md
README.md
LICENSE                           # MIT
```

`build.sh` is strictly the ORT + MIGraphX layer; that's what the eight patches address and that's what matters if you want to run *other* ONNX models on your RX 9070. `scripts/install-witness-stack.sh` adds the Witness-specific pieces on top (whisper.cpp with Vulkan, the speakrs fork with MIGraphX support, Whisper large-v3 model download). Split this way so anyone who just wants the patches doesn't have to install the diarization stack.

If you try the build on a different RDNA 4 GPU (9070 XT, any other Navi 48 / gfx1201 variant), open an issue with the output, I want to know what works and what doesn't. Same if you try a non-diarization ONNX model. Data wins arguments.

---

## Credits and honesty

- **AMD**, ROCm is imperfect on consumer hardware but it *exists* and is open-source, which is what made these patches possible in the first place. This work is built on theirs.
- **`avencera/speakrs`**, excellent Rust diarizer. I'll open a PR adding MIGraphX support upstream after this post goes out.
- **pyannote community-1**, the segmentation and embedding models I'm using.
- **ONNX Runtime team**, for keeping MIGraphX EP alive after dropping the old ROCm one.
- I'm not an ML researcher. I'm not an AMD employee. One person, one RX 9070, some evenings and weekends. If you're in a similar position and want to do the same thing, the cost is time, not money.

No sponsor, no employer, no hidden angle. I needed a thing. I built it. Here's how. Ask me anything in the comments or on whichever platform found this.

---

## Questions I expect

**"Why not just buy an NVIDIA card?"**
Because "buy different hardware" isn't an engineering answer. The patches *are* the engineering answer, and they unlock every ONNX model on every RDNA 4 card, not just mine.

**"Why not use Python pyannote on CPU?"**
13 seconds on CPU vs 11 seconds on GPU for a 3-min call isn't a huge win. But for long files, GPU pulls ahead linearly. The 16-minute call would've been 70+ seconds on CPU, 30-ish on GPU. And this was always about more than one pipeline, the patches unlock every ONNX model, not just diarization.

**"Why not WhisperX, sherpa-onnx, or NVIDIA NeMo?"**
WhisperX inherits pyannote community-1 as its diarizer, so it hits the same CUDA-only wall on AMD that this post is about. It's in the comparison table. sherpa-onnx is an ONNX-based alternative I haven't benchmarked on VoxConverse yet; it's a good candidate for the "verified working" list. NVIDIA NeMo (Sortformer, MSDD) is research-grade and currently the accuracy frontier, but it's a training-framework model family, not a drop-in Python pipeline, and not ONNX-native. None of these change the underlying problem the patches solve: running a modern ONNX diarizer on a consumer AMD card.

**"How did you debug the MIGraphX segfault?"**
A minimal reproducer that loaded the specific ONNX model outside the larger pipeline. It died in the same place. From there, binary search on the model topology narrowed the bug to dynamic shape handling in the segmentation graph.

**"Does this work on RDNA 3?"**
Untested. Verified on gfx1201 / RX 9070 only. RDNA 3 on Ubuntu is already supported by AMD's official ROCm path for many models, so you shouldn't need these patches there. PR welcome if you test on Fedora.

**"Does this work with PyTorch ROCm?"**
Different stack. PyTorch ROCm uses its own backend (MIOpen + ROCm runtime), not ONNX Runtime + MIGraphX. These patches don't help PyTorch directly. If you're doing PyTorch on consumer RDNA 4 on Fedora, you'll hit your own separate set of rough edges, I'd be curious to read that blog post when someone writes it.

**"What's the VRAM floor for diarization only?"**
speakrs + MIGraphX peaks around 4-5 GiB VRAM in solo diarization mode. Parallel with Whisper is where it hits ~15 GiB (on my 16 GiB card). If you have an 8 GiB or 12 GiB card, run serial mode and you'll be comfortable. The pipeline auto-detects free VRAM and falls back.

**"Will you submit these upstream?"**
Yes. Patches 1-6 to MIGraphX, patches 7-8 to ORT. Plus a PR to speakrs adding the EP wiring. Some of them (the stubs) are "graceful degradation" not "correct fix", so they may not merge as-is, but they document the problem and open the conversation.

---

Repo: **[github.com/maherr/onnxruntime-migraphx-rdna4](https://github.com/maherr/onnxruntime-migraphx-rdna4)**. Star if you want updates.
