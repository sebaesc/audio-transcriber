# Specification: Evaluate Against a Custom Ground Truth

**Track ID:** custom-ground-truth_20261002
**Type:** Feature
**Created:** 2026-10-02
**Status:** Complete

## Summary

Let the evaluation harness score transcripts of the team's own audio/video against a ground truth
the user writes in the pipeline's own transcript format (a bare JSON array of five-key segments),
either by running a backend on the recording or by scoring a transcript the pipeline already
produced.

## Context

The harness (`python -m audio_transcriber.evaluation`) only scores against the Bangor Miami corpus,
with optional TalkBank speaker and language annotations. Those are useful as a fixed benchmark, but
they are not the team's audio: the product exists for the team's own recordings, mostly English
with Spanish speakers who code-switch. Scores on that audio are what decide which backend and
settings to use.

The scoring code already compares by time against timed utterances (`annotation.diarization_error`,
`annotation.language_confusion`) and by words (`scoring.score`). A ground truth in the five-key
segment format carries everything those need, so they are reused rather than rewritten.

Typical input: recordings of about 20 min (±3 min), one or several per run.

## User Story

As a team member transcribing our own recordings, I want to score the pipeline's output for my own
audio/video against a ground-truth transcript I wrote in the same JSON format, so that I can see
how well each backend works on our real audio, not only on the reference corpus, and compare
backends or settings on it.

## Acceptance Criteria

- [ ] **Run a backend against a ground truth.** The evaluation CLI accepts the user's own recordings
      in place of the built-in corpus, in two forms:
      - a single pair: `--audio FILE --reference FILE`
      - a folder of pairs: `--references DIR`, matching `name.<audio/video ext>` to `name.json`

      It runs the chosen backend on each recording (video's audio track decoded by ffmpeg, as in
      the pipeline) and writes a per-recording report with WER and `retained`, as the corpus mode
      does.
- [ ] **Speakers and language from the ground truth.** If the ground truth has `speaker` labels,
      the report includes diarization error (with `confusion` broken out). If it has
      `spoken_language`, the report includes per-segment language accuracy. A dimension that has
      no labels in the ground truth (all `null` or missing) is skipped, and the report says why.
- [ ] **Invalid ground truth fails up front.** A ground-truth file that is not a JSON array, has a
      segment missing a required key, or has `end_time` before `start_time` stops the run before
      anything is transcribed. The error names the file, the segment index and the problem. A
      folder entry with audio but no `.json` (or the reverse) is reported the same way.
- [ ] **Compare runs.** Reports from custom ground truth have the same shape as corpus reports, so
      `--compare BASELINE CANDIDATE` compares two backends or settings recording by recording.
- [ ] **Diff the words.** Next to each hypothesis `.txt` and `.json`, the run writes the
      normalized reference text, so `diff` shows which words cost the WER.
- [ ] **Score an existing transcript, no backend run.** Given a transcript the pipeline already
      wrote, either a single file (`--transcript FILE --reference FILE`) or the pipeline's
      output folder against a reference folder, the harness scores it without running a backend,
      so a hosted backend's output is scored without spending credits again. Backend, model
      version and config are taken from the run manifest when one is found, and marked unknown
      otherwise.
- [ ] **Documented.** `docs/evaluation.md` explains how to write a ground truth (with an example),
      both input forms, scoring an existing transcript, and what each metric means on custom data.

## Dependencies

Existing code only; no incomplete tracks.

- `evaluation/`: `runner.evaluate`/`Report`/`compare`, `scoring.score`, `text.normalize`,
  `annotation.diarization_error` and `annotation.language_confusion` (which take TalkBank
  `Utterance`s today)
- `schema.Segment` and `SchemaError`, to read and validate ground-truth segments
- `audio.decoded` (ffmpeg), for audio and video inputs
- `manifest`, to read backend/model/config for an already-produced transcript
- The `diarization` extra (`pyannote.metrics`) for speaker scoring, as with `--talkbank`

## Out of Scope

- Ground-truth formats other than the five-key JSON array: no SRT, VTT, CHAT or plain-text import.
  Plain text would only give WER and `retained`, which the JSON already gives.
- Word-level timing or word-level language scoring (the ground truth is per segment)
- A tool or UI for writing or editing ground truth
- Changes to the pipeline CLI, the transcript schema or the run manifest format
- Committing the team's recordings or ground-truth files (reports may be committed; see below)
- Changes to the Bangor Miami or TalkBank evaluation behavior

## Technical Notes

- **Overlapping speech is allowed in the ground truth.** `Transcript` rejects overlapping segments,
  but a human ground truth may legitimately have two speakers at once, and diarization error
  handles overlap. The loader validates each segment with `Segment.from_dict` and the file as a
  whole (array, non-empty, ordering not required), not with `Transcript`.
- **Times may be clock times.** _(Added 2026-10-02, during implementation.)_ The team's
  ground truths are written by ear as `"MM:SS"` strings, not seconds. The loader accepts
  seconds (numbers or numeric strings) and `MM:SS` / `HH:MM:SS` strings, optionally with
  fractional seconds, and converts them on read; any other string is an error naming the
  file, segment and key. Only the ground-truth loader changes, not the transcript schema.
- **Ground truth to utterances.** Each segment becomes a timed `Utterance` (speaker,
  start, end, language), so `annotation` scores it unchanged. If `Utterance` needs to move out of
  `talkbank.py` to be shared, that is a refactor inside `evaluation/`.
- **Languages.** `language_confusion` scores accuracy over `en`/`es` only. That suits the team's
  audio. Other codes are counted in the confusion table but not in accuracy; the doc says so.
- **Reference text** is the segments' `text` joined in time order and run through `normalize`,
  not `clean_reference` (which strips CHAT markup a user's JSON does not contain).
- **Duration** for diarization scoring comes from the decoded audio. When scoring an existing
  transcript without audio, it is the later of the last reference and hypothesis `end_time`.
- **Reports may be committed**, under `eval/` alongside the corpus reports (for example
  `eval/custom/<label>.json`). The recordings and ground-truth files stay outside the repo; the
  report records the reference file's checksum so a score can be traced to its ground truth.
- **CLI modes are mutually exclusive.** Corpus mode, custom ground truth and scoring an existing
  transcript cannot be combined; `--talkbank` and `--only` apply to corpus mode only (`--only`
  may also filter a references folder by name).
- **Budget.** A ~20 min recording on `elevenlabs-scribe` costs about 1/7 of a free-tier month, so
  scoring existing transcripts is the default way to re-score hosted output.
- Tests use small hand-written ground-truth JSON and a fake backend; no real recordings in fixtures.

---

_Generated by Conductor. Review and edit as needed._
