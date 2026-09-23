# Implementation Plan: Code-Switching Language Detection and Speaker Diarization

**Track ID:** codeswitch-diarization_20260922
**Spec:** [spec.md](./spec.md)
**Created:** 2026-09-22
**Status:** [ ] Not Started

## Overview

Build the measurement before the thing being measured. Phase 1 turns the spike's throwaway eval
scripts into a committed harness with a recorded baseline, because without it no later change can
be shown to help. Phase 2 fixes the chunking, which the prototype got wrong in two ways — too
coarse, and dropping speech. Phase 3 puts per-chunk language detection behind backend config and
proves it against the baseline. Phase 4 adds diarization, gated on the ground-truth question
settled in Phase 1. Phase 5 documents and tunes.

Nothing in this track touches the pipeline, the schema, the interface, or the manifest. Both
fields already exist; this fills them. If a change here appears to need a contract change, that is
a signal to stop and reconsider the approach, per the approval rules in `workflow.md`.

## Phase 1: Eval Harness and Baseline

Make the improvement measurable. The spike's numbers live in a scratch directory and are not
reproducible; this phase makes them so.

### Tasks

- [x] Task 1.1: Add an `eval` optional extra (`jiwer`, `pyarrow`) — never a core or backend dependency
- [x] Task 1.2: Write the dataset fetcher: download the `mixed` config on demand into a gitignored cache, verify it, and never commit audio
- [x] Task 1.3: Implement reference cleaning for CHAT markup — `(.)` pauses, `<>` retracing, `+` continuations, `mani(cure)` partials — with tests over hand-written examples
- [x] Task 1.4: Implement scoring: WER via `jiwer`, plus a language-mix measure reporting transcribed Spanish share against the reference share
- [x] Task 1.5: Write the eval runner — takes a backend name and config, runs the `mixed` subset, writes a JSON report
- [x] Task 1.6: Record the per-file baseline for all 5 recordings and commit the report as the number to beat
- [x] Task 1.7: **Settle the diarization ground-truth question** (see Risks in the spec): TalkBank credentials, another labelled corpus, or qualitative-only. Record the decision here before Phase 4 is planned in detail

### Verification

- [ ] `pytest` passes; reference cleaning is covered by tests
- [ ] The eval runs end to end and reproduces a WER near 0.44 for `zeledon4` under the current default
- [ ] The baseline report is committed, and Task 1.7's decision is written into this plan

### Task 1.7 decision — diarization ground truth

**Decided 2026-09-22: two corpora, two questions.**

What was checked:

| Source | Speaker labels | Code-switched | Reachable |
| ------ | -------------- | ------------- | --------- |
| `BrunoHays/Bangor-Miami-…` (this track's corpus) | No — CHAT speaker tiers stripped | Yes | Yes |
| TalkBank (original Bangor source) | Yes, plus word-level language IDs | Yes | **No — authentication modal** |
| `ksheth2/miamibangor` | — | — | Empty stub, README only |
| `diarizers-community/voxconverse`, `/ami` | Yes | No — English only | Yes, ungated |

**Decision:** use `diarizers-community/voxconverse` to measure DER and prove the diarizer works at
all, which unblocks Phase 4 without an external dependency. Pursue TalkBank credentials in
parallel for the integrated bilingual evaluation.

**Consequence for Phase 4:** diarization quality is measured on English conversation, not on this
project's actual audio. Until TalkBank lands, no claim is made about diarization accuracy on
code-switched speech, and the backend doc page must say so.

**Bonus if TalkBank arrives:** its word-level language annotations would replace the aggregate
`spanish_share` this track currently scores against, turning the language-mix measure into true
per-segment language accuracy.

## Phase 2: Chunking

Fix what the prototype got wrong. This phase changes no behaviour visible to a user — it produces
correct chunk boundaries and proves they lose nothing.

### Tasks

- [x] Task 2.1: Implement VAD-based chunk grouping with a configurable target length, replacing the prototype's logic which produced ~68 s chunks against a 28 s target
- [x] Task 2.2: Write tests asserting chunk lengths stay within the target and boundaries land on speech gaps
- [x] Task 2.3: **Fix the content-loss regression.** The prototype produced 3683 words against a 4269-word reference, with 666 deletions; measure what VAD discards and close the gap
- [x] Task 2.4: Add a coverage assertion to the eval — transcribed audio duration against total speech duration — so a future change cannot buy WER by dropping speech
- [x] Task 2.5: Fix decoding reproducibility (unplanned; see deviations)
- [x] Task 2.6: Fix silently dropped backend config (unplanned; see deviations)

### Verification

- [ ] Chunk lengths respect the configured target
- [ ] Transcribing chunk-by-chunk under the file-level language scores no worse than the baseline; this isolates chunking from detection
- [ ] Coverage assertion in place and passing

### Deviations

**2026-09-22 — content loss was the chunking bug, not VAD.** Task 2.3 assumed VAD was discarding
speech. With chunk length bounded, speech coverage measured 0.97 and 0.99 on the two recordings
checked, so VAD discards very little. The prototype's deletions came from single language
decisions spanning minutes of bilingual audio. Task 2.4's guard is satisfied by the `coverage`
field added to `Score` in Phase 1, which is asserted per recording in `Comparison`.

**2026-09-22 — two unplanned fixes to the shipped backend (Tasks 2.5, 2.6).** Phase 1's harness
found both; neither was visible to the existing test suite.

*Reproducibility.* faster-whisper samples above temperature 0.0 and the backend never pinned a
seed, so the same recording gave WER 0.409 vs 0.441 across runs — a 0.127 spread, larger than any
improvement this track could plausibly show. A first attempt defaulted to greedy decoding; it was
measured and reverted, because removing the ladder cost 0.185 WER and 15 points of coverage, and
no configuration between the two extremes beat either. The ladder earns its place: 14.5% of
segments fall back and some need temperature 1.0 to escape repetition loops. Seeding CTranslate2's
RNG gives accuracy and reproducibility together, verified byte-identical through the full pipeline.

The default seed of 0 is deliberately untuned. Seeds 42 and 7 scored 0.259 and 0.394 WER on the
same recording; picking the better one would fit noise and inflate every subsequent measurement.

*Silently dropped config.* The backend accepted any keyword, recorded it in the manifest, and
forwarded only those `transcribe()` named inline. Two runs configured differently produced
byte-identical output while the manifest claimed the option was applied. Decode options are now
derived from the config so recording and forwarding cannot diverge, unknown options are rejected,
and a test asserts every recorded option reaches the decoder.

**Consequence:** the first committed baseline was invalid — unseeded, and taken at an unknown
point in a 0.127 spread. It has been replaced.

## Phase 3: Per-Chunk Language Detection

The measured win. Opt-in, restricted to `{en, es}`.

### Tasks

- [x] Task 3.1: Implement per-chunk detection taking the best of `{en, es}` from `all_language_probs`, not the global argmax
- [x] Task 3.2: Implement the confidence threshold and its fallback to the file-level language; choose the value from the eval rather than by intuition
- [x] Task 3.3: Add `language_detection` backend config (`file` default, `chunked` opt-in) and the allowed-language set as config
- [x] Task 3.4: Populate `spoken_language` per segment from its chunk's detected language
- [x] Task 3.5: Record the detection mode, threshold, and allowed languages in the effective config, so a run's manifest explains its own output
- [x] Task 3.6: Write tests — detection restricted to the allowed set, threshold fallback, and `spoken_language` varying within one transcript
- [x] Task 3.7: Run the eval and compare against the Phase 1 baseline

### Verification

- [x] `pytest -k conformance` still passes in both modes
- [ ] **Not met:** WER improves on the baseline and Spanish share lands within 0.05 of the reference
- [x] Coverage assertion still passing — the win is not bought with deletions
- [x] The default path is byte-for-byte unchanged when `language_detection=file`

### Phase 3 result

Measured against the reproducible baseline, span-merged:

| recording | ΔWER | Δcoverage | mix error base → chunked |
| --------- | ---- | --------- | ------------------------ |
| herring17 | −0.052 | +0.019 | 0.341 → 0.083 |
| maria20   | −0.156 | +0.076 | 0.348 → 0.092 |
| sastre8   | +0.081 | −0.055 | 0.013 → 0.037 |
| zeledon14 | +0.009 | −0.011 | 0.001 → 0.006 |
| zeledon4  | +0.030 | −0.059 | 0.118 → 0.034 |

Mean: WER −0.018, coverage −0.006, language-mix error −0.114.

**All five recordings now report both languages**, where the baseline reported one for each.
Language-mix error improved on every recording. WER improved on two.

**The acceptance criterion "WER improves across all five" is not met, and will not be.** Chunked
detection helps where file-level detection guessed wrong (`herring17`, `maria20`: mix errors of
0.34) and costs where it guessed right (`sastre8`, `zeledon14`: mix errors of 0.013 and 0.001 —
nothing to fix, only overhead to add). That shape is inherent, not a tuning failure.

Two variants were measured. Decoding each chunk separately scored mean ΔWER −0.027 but regressed
coverage by 0.040, because a word split across two decodes is transcribed badly in both. Merging
consecutive chunks that agree on a language into one decode fixed coverage (−0.006) and gave back
some WER (−0.018). Neither dominates; span merging was kept because a coverage regression is the
failure mode the eval exists to catch.

**Consequence for Task 5.6:** `chunked` stays opt-in. It is the right setting for genuinely
bilingual audio and a net loss for monolingual audio, and the backend cannot know which it has
before transcribing.

## Phase 4: Speaker Diarization

Scope depends on Task 1.7. If no ground truth is available, this phase ships the capability with
qualitative evaluation only, and says so plainly in the doc page.

### Tasks

- [x] Task 4.1: Select a diarization model and record why, including its licence and gating
- [x] Task 4.2: Add it as its own optional extra, loaded only when diarization is enabled
- [x] Task 4.3: Implement requirement checks — model availability, credentials, device — reported up front like every other backend requirement
- [x] Task 4.4: Assign speaker labels to segments, stable within a transcript
- [x] Task 4.5: Make `capabilities.diarization` reflect whether it is actually active, rather than what the engine could do
- [x] Task 4.6: Record the diarization model and its version in the manifest, so a transcript's speaker labels are traceable
- [ ] Task 4.7: Evaluate against whatever Task 1.7 settled on, and write the result down

### Task 4.1 decision — diarization model

**Decided 2026-09-23: `pyannote/speaker-diarization-3.1`.**

Every full diarization pipeline available is either gated or non-commercial; the ungated
permissively-licensed options are speaker-embedding models that would need a segmentation
front-end and clustering built around them.

| Model | Gated | Licence | Commercial use |
| ----- | ----- | ------- | -------------- |
| `pyannote/speaker-diarization-3.1` | Yes, accept conditions + token | MIT | Yes |
| `soniqo/Sortformer-Diarization-4spk-ONNX` | No | CC-BY-NC-4.0, inherited from `nvidia/diar_streaming_sortformer_4spk-v2.1` | **No** |
| `speechbrain/spkrec-ecapa-voxceleb` | No | Apache-2.0 | Yes, embeddings only |
| `pyannote/wespeaker-voxceleb-resnet34-LM` | No | CC-BY-4.0 | Yes, embeddings only |

Sortformer was the tempting option — ungated, good quality, and it runs on the `onnxruntime`
already installed — but it inherits NVIDIA's non-commercial licence, and `product.md` describes
internal team workflows. That is commercial use.

**Revised during implementation:** the pinned `pyannote.audio<4.0` resolved to 3.4.0, which
references `torchaudio.AudioMetaData` — removed in torchaudio 2.11. Upgrading to `>=4.0,<5.0`
fixes it, and 4.x ships `pyannote/speaker-diarization-community-1` as its recommended pipeline.
That is now the default: it bundles its embedding and scoring weights, so only its own gate needs
accepting, where `speaker-diarization-3.1` references its segmentation model externally and needs
two. The required gates are therefore per model rather than a fixed list.

**Consequence:** the model cannot be fetched until someone accepts the conditions at
`huggingface.co/pyannote/speaker-diarization-community-1` and sets `HF_TOKEN`. Everything else in
this phase is built and tested against that; the requirement check names the pages for whichever
pipeline is configured.

### Verification

- [ ] `speaker` is populated and stable within a transcript when enabled, `null` when not
- [ ] Disabled by default; nothing about the Phase 3 output changes when it is off
- [ ] `pytest -k conformance` passes with diarization both on and off

## Phase 5: Tuning and Documentation

### Tasks

- [ ] Task 5.1: Tune chunk length and confidence threshold against the eval; record the chosen values and what they cost
- [ ] Task 5.2: Measure and document throughput for each mode — chunked detection adds a pass, diarization adds a model
- [ ] Task 5.3: Update `docs/backends/faster-whisper.md` with the new options, their cost, and their limits
- [ ] Task 5.4: Document the eval harness: how to run it, what it measures, and how to read the report
- [ ] Task 5.5: Record in the spec's known limits that intra-sentential switching is not expressible at chunk granularity
- [ ] Task 5.6: Decide whether `chunked` should become the default, with the eval as the argument either way

### Verification

- [ ] Doc page current, per gate 6
- [ ] A real run over bilingual audio produces varying `spoken_language` and populated `speaker`

## Final Verification

- [ ] All acceptance criteria in [spec.md](./spec.md) met
- [ ] Gate 1 — `pytest` passing
- [ ] Gate 2 — detection, chunking, and normalization covered by tests
- [ ] Gate 3 — `pytest -k conformance` passing in every mode
- [ ] Gate 4 — `ruff check .` clean
- [ ] Gate 5 — `ruff format --check .` clean
- [ ] Gate 6 — the `faster-whisper` doc page covers every new option
- [ ] Gate 7 — manifest records detection mode, threshold, languages, and diarization model
- [ ] Eval report committed, showing the improvement over the recorded baseline
- [ ] Manual sign-off recorded here

---

_Generated by Conductor. Tasks will be marked [~] in progress and [x] complete._
