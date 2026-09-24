"""Tests for speaker diarization.

pyannote's models are gated, so these cover everything around the model: the
requirement check that tells a user how to get access, the overlap arithmetic
that assigns a speaker to a segment, and the token handling.
"""

from dataclasses import replace
from pathlib import Path

import pytest

from audio_transcriber.backends import diarization
from audio_transcriber.errors import BackendRequirementError

pytest.importorskip(
    "faster_whisper", reason="the faster-whisper extra is not installed"
)

from audio_transcriber.backends.faster_whisper import (  # noqa: E402
    FasterWhisperBackend,
)
from audio_transcriber.schema import Segment, Transcript  # noqa: E402


def _raising_import(blocked: str):
    """Return an __import__ that fails for one module, for testing the check."""
    real = (
        __builtins__["__import__"]
        if isinstance(__builtins__, dict)
        else __builtins__.__import__
    )

    def _fake(name, *args, **kwargs):
        if name == blocked:
            raise ImportError(f"No module named {name!r}")
        return real(name, *args, **kwargs)

    return _fake


def turns(*spans) -> list[diarization.SpeakerTurn]:
    return [diarization.SpeakerTurn(start=a, end=b, speaker=s) for a, b, s in spans]


# --- Assigning a speaker to a segment ---


def test_a_segment_takes_the_speaker_holding_most_of_it():
    found = diarization.speaker_for(
        turns((0.0, 5.0, "SPEAKER_00"), (5.0, 10.0, "SPEAKER_01")), 4.0, 9.0
    )

    assert found == "SPEAKER_01"


def test_a_segment_inside_one_turn_takes_that_speaker():
    assert diarization.speaker_for(turns((0.0, 10.0, "A")), 2.0, 3.0) == "A"


def test_a_segment_overlapping_nothing_has_no_speaker():
    """Inventing one would be worse than admitting there is none."""
    assert diarization.speaker_for(turns((0.0, 5.0, "A")), 20.0, 25.0) is None


def test_no_turns_means_no_speaker():
    assert diarization.speaker_for([], 0.0, 5.0) is None


def test_a_tie_is_resolved_and_not_left_empty():
    found = diarization.speaker_for(turns((0.0, 5.0, "A"), (5.0, 10.0, "B")), 0.0, 10.0)

    assert found in {"A", "B"}


def test_overlap_of_disjoint_spans_is_zero():
    turn = diarization.SpeakerTurn(start=0.0, end=5.0, speaker="A")

    assert turn.overlap(10.0, 15.0) == 0.0


def test_overlap_is_the_shared_seconds():
    turn = diarization.SpeakerTurn(start=0.0, end=5.0, speaker="A")

    assert turn.overlap(3.0, 9.0) == 2.0


# --- Requirements and access ---


def test_a_missing_extra_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "pyannote.audio", None)
    monkeypatch.setattr(
        "builtins.__import__",
        _raising_import("pyannote.audio"),
    )

    with pytest.raises(BackendRequirementError, match=r"\[diarization\]"):
        diarization.check_requirements(diarization.DEFAULT_MODEL, None, "stub")


def test_a_missing_token_names_both_gated_models(monkeypatch):
    """Accepting only the first fails later with an unhelpful 401."""
    pytest.importorskip(
        "pyannote.audio", reason="the diarization extra is not installed"
    )
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    # A developer running this has a real .env. Without this, dotenv finds it,
    # a token is available, and the test passes while checking nothing.
    monkeypatch.setattr(diarization, "_dotenv_loaded", True)

    with pytest.raises(BackendRequirementError) as caught:
        diarization.check_requirements(diarization.DEFAULT_MODEL, None, "stub")

    message = str(caught.value)
    for model in diarization.gates_for(diarization.DEFAULT_MODEL):
        assert model in message
    assert "settings/tokens" in message


def test_the_3_1_pipeline_reports_both_of_its_gates():
    """It references its segmentation model externally, unlike community-1."""
    gates = diarization.gates_for("pyannote/speaker-diarization-3.1")

    assert gates == (
        "pyannote/speaker-diarization-3.1",
        "pyannote/segmentation-3.0",
    )


def test_an_unknown_pipeline_is_assumed_to_gate_only_itself():
    assert diarization.gates_for("someone/custom-pipeline") == (
        "someone/custom-pipeline",
    )


def test_the_token_is_read_from_the_environment(monkeypatch):
    monkeypatch.setattr(diarization, "_dotenv_loaded", True)
    monkeypatch.setenv("HF_TOKEN", "hf_abc")

    assert diarization.find_token(None) == "hf_abc"


def test_an_explicit_token_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_env")

    assert diarization.find_token("hf_explicit") == "hf_explicit"


def test_alternative_token_variables_are_honored(monkeypatch):
    monkeypatch.setattr(diarization, "_dotenv_loaded", True)
    for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_alt")

    assert diarization.find_token(None) == "hf_alt"


# --- Backend integration ---


def test_diarization_is_off_by_default():
    backend = FasterWhisperBackend()

    assert backend.config["diarization"] is False
    assert backend.capabilities.diarization is False


def test_enabling_diarization_declares_the_capability():
    """The flag reports what is active, not what the engine could do."""
    backend = FasterWhisperBackend(diarization=True)

    assert backend.capabilities.diarization is True


def test_enabling_it_on_one_instance_does_not_affect_another():
    FasterWhisperBackend(diarization=True)

    assert FasterWhisperBackend().capabilities.diarization is False


def test_the_token_is_redacted_from_the_manifest():
    """A manifest is written to disk and read by anyone with the run directory."""
    config = FasterWhisperBackend(hf_token="hf_secret").effective_config()

    assert config["hf_token"] == "<redacted>"
    assert "hf_secret" not in str(config)


def test_the_diarization_model_is_recorded_for_traceability():
    config = FasterWhisperBackend(diarization=True).effective_config()

    assert config["diarization_model"] == diarization.DEFAULT_MODEL
    assert "diarization_version" in config


def test_nothing_diarization_shaped_is_recorded_when_it_is_off():
    assert "diarization_version" not in FasterWhisperBackend().effective_config()


def test_diarization_options_are_not_passed_to_the_decoder():
    backend = FasterWhisperBackend(diarization=True, num_speakers=2)
    options = backend._decode_options()

    for key in ("diarization", "diarization_model", "hf_token", "num_speakers"):
        assert key not in options


def test_a_transcript_is_unchanged_when_diarization_is_off():
    backend = FasterWhisperBackend()
    transcript = Transcript(
        source="a.wav",
        segments=(Segment(start_time=0.0, end_time=1.0, text="hello"),),
        backend=backend.name,
        model_version="x",
    )

    assert backend._maybe_diarize(transcript, Path("a.wav"), "a.wav") is transcript


def test_speakers_are_attached_to_every_segment(monkeypatch):
    backend = FasterWhisperBackend(diarization=True)
    monkeypatch.setattr(
        backend, "_diarize", lambda path: turns((0.0, 5.0, "S0"), (5.0, 10.0, "S1"))
    )
    transcript = Transcript(
        source="a.wav",
        segments=(
            Segment(start_time=0.0, end_time=2.0, text="first"),
            Segment(start_time=6.0, end_time=8.0, text="second"),
        ),
        backend=backend.name,
        model_version="x",
    )

    result = backend._maybe_diarize(transcript, Path("a.wav"), "a.wav")

    assert [s.speaker for s in result.segments] == ["S0", "S1"]
    assert [s.text for s in result.segments] == ["first", "second"]


def test_empty_diarization_leaves_the_transcript_alone(monkeypatch):
    backend = FasterWhisperBackend(diarization=True)
    monkeypatch.setattr(backend, "_diarize", lambda path: [])
    transcript = Transcript(
        source="a.wav",
        segments=(Segment(start_time=0.0, end_time=1.0, text="hello"),),
        backend=backend.name,
        model_version="x",
    )

    assert backend._maybe_diarize(transcript, Path("a.wav"), "a.wav") is transcript


def test_a_diarization_failure_is_a_per_file_failure(monkeypatch):
    """One file failing must not abort the batch."""
    from audio_transcriber.errors import TranscriptionError

    backend = FasterWhisperBackend(diarization=True)

    def _explode(path):
        raise RuntimeError("cuda out of memory")

    monkeypatch.setattr(backend, "_diarize", _explode)
    transcript = Transcript(
        source="a.wav",
        segments=(Segment(start_time=0.0, end_time=1.0, text="hello"),),
        backend=backend.name,
        model_version="x",
    )

    with pytest.raises(TranscriptionError, match="diarization failed"):
        backend._maybe_diarize(transcript, Path("a.wav"), "a.wav")


def test_speaker_labels_survive_the_schema_round_trip(monkeypatch):
    import json

    backend = FasterWhisperBackend(diarization=True)
    monkeypatch.setattr(backend, "_diarize", lambda path: turns((0.0, 5.0, "S0")))
    transcript = Transcript(
        source="a.wav",
        segments=(Segment(start_time=0.0, end_time=2.0, text="hello"),),
        backend=backend.name,
        model_version="x",
    )

    result = backend._maybe_diarize(transcript, Path("a.wav"), "a.wav")
    data = json.loads(json.dumps(result.to_json_data()))

    assert data[0]["speaker"] == "S0"


def test_a_dotenv_file_supplies_the_token(tmp_path, monkeypatch):
    """Python does not read .env on its own, so this has to be wired up."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("HF_TOKEN=hf_from_dotenv\n")
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(diarization, "_dotenv_loaded", False)

    assert diarization.find_token(None) == "hf_from_dotenv"


def test_an_exported_variable_beats_the_dotenv_file(tmp_path, monkeypatch):
    """Setting one for a single run must do what it looks like it does."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("HF_TOKEN=hf_from_dotenv\n")
    monkeypatch.setenv("HF_TOKEN", "hf_exported")
    monkeypatch.setattr(diarization, "_dotenv_loaded", False)

    assert diarization.find_token(None) == "hf_exported"


# --- Device ---
#
# pyannote pipelines load onto the CPU and stay there unless moved. The backend
# never moved it, so diarization ran on CPU beside a CUDA transcription: 57 s on
# CUDA against more than ten minutes on CPU, for the same 21.8-minute recording.


class _StubPipeline:
    def __init__(self):
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


def test_the_pipeline_is_moved_to_the_requested_device(monkeypatch):
    torch = pytest.importorskip("torch")
    pyannote_audio = pytest.importorskip("pyannote.audio")
    stub = _StubPipeline()
    monkeypatch.setattr(
        pyannote_audio.Pipeline, "from_pretrained", lambda *a, **k: stub
    )

    diarization.load_pipeline(diarization.DEFAULT_MODEL, "hf_x", device="cuda")

    assert stub.moved_to == torch.device("cuda")


def test_without_a_device_the_pipeline_is_left_where_it_loaded(monkeypatch):
    pyannote_audio = pytest.importorskip("pyannote.audio")
    stub = _StubPipeline()
    monkeypatch.setattr(
        pyannote_audio.Pipeline, "from_pretrained", lambda *a, **k: stub
    )

    diarization.load_pipeline(diarization.DEFAULT_MODEL, "hf_x")

    assert stub.moved_to is None


def test_diarization_runs_on_the_device_transcription_resolved(monkeypatch):
    """`device: auto` resolves once, and both models should use the answer."""
    seen = {}

    def _load(model, token, device=None):
        seen["device"] = device
        return object()

    monkeypatch.setattr(diarization, "load_pipeline", _load)
    monkeypatch.setattr(diarization, "diarize", lambda *a, **k: [])
    backend = FasterWhisperBackend(diarization=True, hf_token="hf_x")
    backend._resolved_device = "cuda"

    backend._diarize(Path("a.wav"))

    assert seen["device"] == "cuda"


# --- Speaker assignment mode ---


def test_speakers_are_assigned_per_segment_by_default():
    assert FasterWhisperBackend().config["speaker_assignment"] == "segment"


@pytest.mark.parametrize("mode", ["segment", "word"])
def test_either_assignment_mode_is_accepted_with_diarization(mode):
    backend = FasterWhisperBackend(diarization=True, speaker_assignment=mode)

    assert backend.config["speaker_assignment"] == mode


def test_word_assignment_without_diarization_is_rejected():
    """There would be no speakers to assign, so the option would do nothing."""
    with pytest.raises(BackendRequirementError, match="diarization=true"):
        FasterWhisperBackend(speaker_assignment="word")


def test_segment_assignment_without_diarization_is_accepted():
    """It is the default, so stating it contradicts nothing."""
    backend = FasterWhisperBackend(speaker_assignment="segment")

    assert backend.config["speaker_assignment"] == "segment"


def test_an_unknown_assignment_mode_is_rejected_with_the_valid_ones():
    with pytest.raises(BackendRequirementError, match="segment, word"):
        FasterWhisperBackend(diarization=True, speaker_assignment="sentence")


def test_the_assignment_mode_is_recorded_in_the_manifest():
    config = FasterWhisperBackend(
        diarization=True, speaker_assignment="word"
    ).effective_config()

    assert config["speaker_assignment"] == "word"


def test_the_assignment_mode_is_not_passed_to_the_decoder():
    backend = FasterWhisperBackend(diarization=True, speaker_assignment="word")

    assert "speaker_assignment" not in backend._decode_options()


# --- Per-word assignment and splitting ---


def words(*spans) -> list[diarization.Word]:
    return [diarization.Word(start=a, end=b, text=t) for a, b, t in spans]


def split(segment, segment_words, speaker_turns):
    return diarization.split_by_speaker(segment, segment_words, speaker_turns)


def test_each_word_takes_the_speaker_holding_most_of_it():
    found = diarization.word_speakers(
        turns((0.0, 1.0, "A"), (1.0, 3.0, "B")),
        words((0.0, 0.9, " one"), (0.8, 1.9, " two"), (2.0, 3.0, " three")),
    )

    assert found == ["A", "B", "B"]


def test_a_word_of_no_length_takes_the_turn_it_falls_in():
    """Whisper collapses some words to an instant; overlap is zero for those."""
    found = diarization.word_speakers(
        turns((0.0, 1.0, "A"), (1.0, 2.0, "B")),
        words((0.5, 1.0, " one"), (1.5, 1.5, " two")),
    )

    assert found == ["A", "B"]


def test_a_word_in_a_gap_takes_the_previous_speaker():
    found = diarization.word_speakers(
        turns((0.0, 1.0, "A"), (3.0, 4.0, "B")),
        words((0.0, 1.0, " one"), (1.5, 2.0, " two"), (3.0, 4.0, " three")),
    )

    assert found == ["A", "A", "B"]


def test_a_leading_word_in_a_gap_takes_the_next_speaker():
    found = diarization.word_speakers(
        turns((1.0, 4.0, "B")),
        words((0.0, 0.5, " one"), (1.0, 2.0, " two")),
    )

    assert found == ["B", "B"]


def test_words_overlapping_no_turn_have_no_speaker():
    found = diarization.word_speakers(
        turns((10.0, 11.0, "A")), words((0.0, 0.5, " one"), (1.0, 2.0, " two"))
    )

    assert found == [None, None]


def test_a_lone_word_between_two_of_another_speaker_is_folded_in():
    """One word flipping at a turn boundary is usually timing drift."""
    assert diarization.smooth(["A", "A", "B", "A"]) == ["A", "A", "A", "A"]


def test_a_lone_word_at_the_edge_is_kept():
    """A reply that ends a segment is exactly what per-word assignment is for."""
    assert diarization.smooth(["A", "A", "A", "B"]) == ["A", "A", "A", "B"]
    assert diarization.smooth(["B", "A", "A", "A"]) == ["B", "A", "A", "A"]


def test_two_words_of_another_speaker_are_a_real_change():
    assert diarization.smooth(["A", "B", "B", "A"]) == ["A", "B", "B", "A"]


def test_a_lone_word_between_two_different_speakers_is_kept():
    assert diarization.smooth(["A", "B", "C"]) == ["A", "B", "C"]


def test_a_segment_is_split_where_the_speaker_changes():
    segment = Segment(
        start_time=0.0, end_time=4.0, text="I think so. Yeah.", spoken_language="en"
    )
    segment_words = words(
        (0.0, 0.5, " I"),
        (0.5, 1.0, " think"),
        (1.0, 1.5, " so."),
        (2.5, 3.9, " Yeah."),
    )

    pieces = split(segment, segment_words, turns((0.0, 2.0, "A"), (2.2, 4.0, "B")))

    assert [(p.start_time, p.end_time, p.text, p.speaker) for p in pieces] == [
        (0.0, 2.5, "I think so.", "A"),
        (2.5, 4.0, "Yeah.", "B"),
    ]
    assert all(p.spoken_language == "en" for p in pieces)


def test_split_pieces_are_ordered_and_contiguous():
    segment = Segment(start_time=0.0, end_time=6.0, text="a b c d e f")
    segment_words = words(
        (0.0, 1.0, " a"),
        (1.0, 2.0, " b"),
        (2.0, 3.0, " c"),
        (3.0, 4.0, " d"),
        (4.0, 5.0, " e"),
        (5.0, 6.0, " f"),
    )

    pieces = split(
        segment,
        segment_words,
        turns((0.0, 2.0, "A"), (2.0, 4.0, "B"), (4.0, 6.0, "A")),
    )

    assert [p.speaker for p in pieces] == ["A", "B", "A"]
    assert pieces[0].start_time == segment.start_time
    assert pieces[-1].end_time == segment.end_time
    for before, after in zip(pieces, pieces[1:], strict=False):
        assert before.end_time == after.start_time
    Transcript(source="a.wav", segments=tuple(pieces))


def test_a_segment_with_one_speaker_matches_segment_assignment():
    segment = Segment(
        start_time=1.0, end_time=3.0, text="all mine", spoken_language="es"
    )
    speaker_turns = turns((0.0, 5.0, "A"))

    pieces = split(
        segment, words((1.0, 2.0, " all"), (2.0, 3.0, " mine")), speaker_turns
    )

    assert pieces == [replace(segment, speaker="A")]


def test_a_segment_without_words_falls_back_to_segment_assignment():
    segment = Segment(start_time=0.0, end_time=3.0, text="no timings")

    pieces = split(segment, [], turns((0.0, 1.0, "A"), (1.0, 3.0, "B")))

    assert pieces == [replace(segment, speaker="B")]


def test_a_segment_whose_words_overlap_nothing_falls_back_too():
    segment = Segment(start_time=0.0, end_time=3.0, text="all silence")

    pieces = split(segment, words((0.0, 1.0, " all")), turns((5.0, 6.0, "A")))

    assert pieces == [replace(segment, speaker=None)]


@pytest.mark.parametrize(
    ("text", "pieces"),
    [
        (" Hi, there.", [" Hi,", " there."]),
        (" ¿Por qué no?", [" ¿Por", " qué", " no?"]),
        (
            " Me lo hago, if I have a party.",
            [" Me", " lo", " hago,", " if", " I", " have", " a", " party."],
        ),
        (" El niño—bueno, sí.", [" El", " niño—bueno,", " sí."]),
    ],
)
def test_split_text_joins_back_into_the_segment_text(text, pieces):
    """faster-whisper keeps each word's leading space and attached punctuation."""
    segment = Segment(start_time=0.0, end_time=float(len(pieces)), text=text.strip())
    segment_words = words(*[(i, i + 1.0, w) for i, w in enumerate(pieces)])
    half = len(pieces) // 2
    speaker_turns = turns(
        (0.0, float(half), "A"), (float(half), float(len(pieces)), "B")
    )

    split_pieces = split(segment, segment_words, speaker_turns)

    assert " ".join(p.text for p in split_pieces) == segment.text
