# Product Vision

## Product Overview

**Name:** Audio Transcriber

**Tagline:** A modular audio transcription pipeline with swappable model backends.

**Description:**
Audio Transcriber turns audio files into structured text transcripts. The pipeline is built
so the transcription model is a plug-in, not a dependency: adding, swapping, or comparing
engines (local Whisper, faster-whisper, hosted APIs) requires implementing one interface, not
rewriting the pipeline. Every supported backend ships with documentation covering how to use
it and what it requires.

## Problem Statement

### The Problem

Transcription tooling hard-wires a single engine. Once the pipeline is written against one
model's API, output shape, and runtime assumptions, switching models means rewriting the
pipeline — so teams stay on whatever they picked first, even when a better, cheaper, or
faster option appears.

### Current Solutions

- Single-vendor SDKs and CLIs (one engine, one output format)
- One-off scripts wrapping a specific model, copied and edited per project
- Hosted transcription services with closed pipelines

### Why They Fall Short

- The model choice leaks into pipeline code, config, and output format
- No consistent transcript schema across engines, so downstream consumers break on a swap
- Setup requirements (system deps, credentials, GPU) are undocumented or buried

## Target Users

### Primary Users

Internal team / self

- **Who:** Our own team, running transcription jobs as part of internal workflows
- **Goals:** Transcribe audio reliably; change the underlying model without reworking downstream consumers
- **Pain Points:** Model lock-in; inconsistent output between engines; undocumented setup requirements
- **Technical Proficiency:** High — comfortable in Python and on the command line

### Secondary Users

Future integrators (not a current target)

- **Who:** Developers who might embed the pipeline in another service later
- **Goals:** Import the package and select a backend programmatically
- **Relationship to Primary:** Served incidentally by keeping the backend interface clean and documented

## Core Value Proposition

### Key Benefits

1. Swap transcription models without touching pipeline code
2. One consistent, structured transcript format regardless of engine
3. Every backend documented: usage, dependencies, credentials, hardware

### Differentiators

- The model is a plug-in behind a stable interface, not the pipeline's center
- Output normalization is part of the contract, not left to each backend

### Value Statement

> Pick the model that fits the job today, and change your mind tomorrow without a rewrite.

## Success Metrics

### Key Performance Indicators

| Metric                       | Target                                     | Measurement Method                       |
| ---------------------------- | ------------------------------------------ | ---------------------------------------- |
| Cost of adding a backend     | One interface implementation + doc page     | Diff size outside the backend module      |
| Output consistency           | Identical schema across all backends        | Schema validation on every backend's output |
| Backend documentation        | 100% of backends documented                 | Doc page per backend in the repo          |

### North Star Metric

Number of transcription backends usable interchangeably with zero pipeline changes.

### Leading Indicators

- Pipeline modules with no engine-specific imports
- Backends covered by the shared conformance tests

### Lagging Indicators

- Backends added over time
- Rework needed when swapping the default engine

## Out of Scope

### Explicitly Not Included

- Web UI or hosted service
- Training or fine-tuning transcription models
- A database — transcripts and run metadata are files on disk

### Future Considerations

- HTTP API in front of the pipeline
- Side-by-side model comparison / benchmarking reports
- Shared storage for transcripts if multiple consumers appear

### Non-Goals

- Competing with hosted transcription products
- Supporting every audio format the ecosystem offers on day one
