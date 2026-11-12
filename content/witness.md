+++
title = "Witness: local transcripts with speaker labels, on any Linux machine"
date = 2026-11-11
description = "Witness transcribes English and French recordings and labels who is speaking, entirely on your own machine. It now installs on any x86_64 Linux distribution and uses an NVIDIA or AMD GPU when it finds one."
+++

Witness turns a recording into a transcript that says who spoke when, and the audio never leaves your computer. As of this week it installs on any x86_64 Linux distribution with one script, and it uses an NVIDIA or AMD GPU when it finds one. Without a GPU it runs on CPU, slower but with the same model.

```sh
git clone https://github.com/maherr/witness
cd witness
bash install.sh
witness meeting.m4a
```

The last line writes `meeting-diarized.md`, a short header followed by the transcript:

```markdown
**SPEAKER_00** [0:00]
Bonjour...

**SPEAKER_01** [0:12]
Oui...
```

It handles English and French, because that is what my recordings are in.

## Where it came from

In April I [wrote about](@/rdna4-missing-rung.md) getting speaker diarization to run on my AMD RX 9070, a card no ML stack officially supported. Witness was the tool on top of that work. It lived in the patch repository, with paths from my machine baked in. Since then the transcription model has changed twice: from Whisper to Cohere Transcribe in July, then to Parakeet Ultra in October. On November 10 Witness moved to [its own repository](https://github.com/maherr/witness) with an installer, and on November 11 it gained NVIDIA support.

## How it works

One model writes every word: [Parakeet Ultra](https://huggingface.co/handy-computer/parakeet-ultra-gguf), a GGUF conversion that runs through [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp). The rest of the pipeline gives that text its structure:

1. [Silero VAD](https://github.com/snakers4/silero-vad) finds the speech.
2. [speakrs](https://github.com/avencera/speakrs), a Rust diarizer built on pyannote's segmentation model and a WeSpeaker embedding model, decides who speaks when.
3. Witness cuts the timeline along those turns into windows of at most 35 seconds, with 5 seconds of overlap. Parakeet Ultra transcribes every window in one process, and the overlap repeats are removed.
4. Whisper listens to the loudest 30 seconds of each third of the file to label it English or French. Its words are discarded.

A transcript from one model is easier to trust than a patchwork. When a window fails on the GPU, it is retried on CPU with the same model, never with a different one. Timestamps come from the speaker turns rather than from word alignment, so they are good for finding a passage and too coarse for word-level subtitles.

## What runs where

The installer works out what your machine can do. It checks for the build tools first; if something is missing, it prints the install command for Fedora, Debian/Ubuntu or Arch and stops. Every download is pinned and checked by size and SHA-256.

| Your machine | Transcription | Speaker labels |
|---|---|---|
| Any x86_64 CPU | CPU | CPU |
| A GPU with Vulkan, and the Vulkan build files installed | GPU, Vulkan | CPU |
| NVIDIA with the CUDA 12 or 13 runtime and cuDNN 9 | Vulkan or CPU | GPU, CUDA |
| NVIDIA with the CUDA toolkit (`nvcc`) as well | GPU, CUDA | GPU, CUDA |
| AMD with an ONNX Runtime you built with MIGraphX | GPU, Vulkan | GPU, MIGraphX |

On NVIDIA the installer sets this up by itself: it downloads Microsoft's GPU build of ONNX Runtime for the speaker labels and runs one real check on a sample before relying on it. If a GPU path fails, at install time or on a later run, you get CPU and a message saying why. Speaker labels on an AMD GPU still need an ONNX Runtime you build yourself; the [patch repository](https://github.com/maherr/onnxruntime-migraphx-rdna4) has the script.

A full install takes 10 to 14 minutes and 2 to 2.5 GB. Building transcription for CUDA takes longer: its GPU kernels compiled in 5 minutes with six jobs on my Ryzen 7 5800X3D, and in 50 minutes on a two-core Colab machine.

## Speed

The test recording is 3 minutes 52 seconds, stitched from three public-domain voices that whisper.cpp ships as samples, in six alternating stretches. Every run below found three speakers and put each voice under its own label.

On my machine (Ryzen 7 5800X3D, RX 9070, Fedora 44, with the desktop busy), measured today:

| Setup | Transcription | Speaker labels | Whole run | Speed vs real time |
|---|---:|---:|---:|---:|
| CPU only | 19.2 s | 88.1 s | 112.7 s | 2.1x |
| Vulkan, labels on CPU | 5.0 s | 88.3 s | 95.2 s | 2.4x |
| **Vulkan and MIGraphX** | **3.2 s** | **8.8 s** | **13.7 s** | **17x** |

The CPU and GPU transcripts came out word for word identical, 504 words each.

For NVIDIA I only have a Tesla T4 on Google Colab, with CUDA 13 and two CPU cores, running a rebuilt copy of the same recording:

| Setup | Transcription | Speaker labels | Whole run | Speed vs real time |
|---|---:|---:|---:|---:|
| CPU only | 139.9 s | 129.4 s | 283.5 s | 0.8x |
| CUDA, labels on CPU | 30.2 s | 128.0 s | 173.2 s | 1.3x |
| CUDA | 31.1 s | 26.5 s | 72.1 s | 3.2x |
| **CUDA, fast speaker labels** | **31.7 s** | **14.5 s** | **60.8 s** | **3.8x** |

The T4 is a 2018 data-center card, and that Colab machine has two slow CPU cores. Parakeet Ultra's decoder runs on CPU threads even on a GPU, which is why transcription takes 31 seconds there and 3 on my desktop. Read the T4 as proof that the CUDA path works, not as what a current card does. I haven't measured one. For a sense of scale, speakrs reports 59x real time for its default CUDA mode on an RTX 4090, for the speaker labels alone.

The fast mode is speakrs's `cuda-fast`. On the same RTX 4090 benchmark (VoxConverse dev), speakrs puts it at 7.4% diarization error against 7.0% for the default mode, at about twice the speed. Choose it with `WITNESS_DIARIZER_GPU=cuda-fast`.

## Accuracy

Word error rate, measured through Witness on my machine in October, lower is better:

| Set | Plain | With speaker labels |
|---|---:|---:|
| Earnings-21, 4 English calls, 128 min | 9.03% | 9.42% |
| CallFriend, 4 Quebec French phone channels, 119 min | 53.88% | 56.27% |

That is four recordings per set, so read it as a direction. The CallFriend references are rough and cover one side of each call, which is why the French numbers are so high. I measured with my own build of the same transcribe.cpp commit, not with the static build the installer makes. For the speaker-label side, the April post has VoxConverse numbers for speakrs on MIGraphX. I haven't re-run either since.

## Privacy

Everything runs locally. speakrs downloads its models from Hugging Face the first time it runs; after that, Witness needs no network at all. I ran the CPU configuration above with networking switched off to check. Its run log, `~/.local/state/witness/runs.jsonl`, keeps run metadata such as timings and backends, and no file names, paths or transcript text.

## What it doesn't do

- Languages other than English and French. Parakeet Ultra takes no language hint, and on very short French turns it sometimes writes English: on those four Quebec calls, 12 to 21% of segments came out in English, mostly turns under a second.
- Overlapping speech. Each overlapping turn holds the same mixed audio, so both transcription windows can hear both voices.
- Voices across files. `SPEAKER_00` in one file has nothing to do with `SPEAKER_00` in the next.
- Word-level timing.
- Anything but x86_64 Linux. On NVIDIA it has run on that one T4. CUDA 12 has only been built, and its fallbacks exercised, in a container without a GPU, and nobody has tried Vulkan on NVIDIA or Intel yet.

---

Code: [github.com/maherr/witness](https://github.com/maherr/witness), MIT. If you run it on hardware I haven't, especially a current NVIDIA card, open an issue with the installer's summary and the timing lines from a transcript header. I'll add the result to the README's table, whichever way it goes.
