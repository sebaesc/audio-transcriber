# Track: Code-Switching Language Detection and Speaker Diarization

**ID:** codeswitch-diarization_20260922
**Status:** Pending
**Depends on:** [core-pipeline_20260922](../core-pipeline_20260922/index.md) (complete)

## Documents

- [Specification](./spec.md)
- [Implementation Plan](./plan.md)

## Progress

- Phases: 0/5 complete
- Tasks: 0/31 complete

## Measured baseline

From a spike on the Bangor Miami corpus (`zeledon4`, 21.8 min, 43% Spanish):

| | Baseline | Chunked prototype |
| --- | --- | --- |
| WER | 0.439 | 0.336 |
| Spanish share (ref 0.629) | 0.770 | 0.628 |
| Distinct `spoken_language` | `{es}` for all 658 segments | varies |

## Quick Links

- [Back to Tracks](../../tracks.md)
- [Product Context](../../product.md)
