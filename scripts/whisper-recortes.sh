#!/usr/bin/env bash
# Transcribe the clips in audio/recortes with faster-whisper, to score them afterwards.
#
# Same config as the earlier runs on the class videos: large-v3, language detected per
# chunk between English and Spanish, and speakers diarized. Diarization reads HF_TOKEN
# from the environment or from .env.
#
# Every line is timestamped. Each file reports how long it took, the speed so far and
# the time left, estimated from that speed. The console output is also kept in
# out/whisper-recortes-<time>.log.
#
# Usage: scripts/whisper-recortes.sh [extra audio_transcriber flags, e.g. --option seed=1]
set -euo pipefail

cd "$(dirname "$0")/.."

INPUT=audio/recortes
OUT=out
# Audio seconds per wall-clock second, from earlier runs with this config; used for the
# estimate before the first file finishes.
EXPECTED_SPEED=4.1

mkdir -p "$OUT"
LOG="$OUT/whisper-recortes-$(date +%Y-%m-%dT%H-%M-%S).log"
exec > >(tee -a "$LOG") 2>&1

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
clock() { printf '%dm%02ds' $(($1 / 60)) $(($1 % 60)); }

# Durations up front, so progress can be told in audio rather than in files.
declare -A duration
total_audio=0
mapfile -t files < <(find "$INPUT" -maxdepth 1 -name '*.mp4' | sort)
if [[ ${#files[@]} -eq 0 ]]; then
    log "No .mp4 files in $INPUT."
    exit 3
fi
log "${#files[@]} files in $INPUT:"
for file in "${files[@]}"; do
    seconds=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$file")
    seconds=${seconds%.*}
    duration[$(basename "$file")]=$seconds
    total_audio=$((total_audio + seconds))
    log "  $(basename "$file")  ($(clock "$seconds"))"
done
expected=$(awk -v a="$total_audio" -v s="$EXPECTED_SPEED" 'BEGIN { printf "%d", a / s }')
log "Total audio $(clock "$total_audio"); expected to take about $(clock "$expected")."
log "Log: $LOG"

before=$(ls -1 "$OUT")
started=$(date +%s)
done_audio=0
current=""
current_started=0

# Closes the file in progress: how long it took, the speed so far, and the time left.
finish_current() {
    [[ -z "$current" ]] && return
    local now took elapsed speed left
    now=$(date +%s)
    took=$((now - current_started))
    elapsed=$((now - started))
    done_audio=$((done_audio + ${duration[$current]:-0}))
    speed=$(awk -v a="$done_audio" -v e="$elapsed" 'BEGIN { printf "%.2f", (e ? a / e : 0) }')
    left=$(awk -v a="$((total_audio - done_audio))" -v s="$speed" 'BEGIN { printf "%d", (s > 0 ? a / s : 0) }')
    log "Done   [$index/$count] $current in $(clock "$took")" \
        "— elapsed $(clock "$elapsed"), ${speed}x realtime, about $(clock "$left") left"
    current=""
}

log "Starting faster-whisper."
set +e
PYTHONUNBUFFERED=1 .venv/bin/python -m audio_transcriber \
    --input "$INPUT" --glob '*.mp4' --out "$OUT" \
    --backend faster-whisper \
    --option model=large-v3 \
    --option language_detection=chunked --option 'languages=["en","es"]' \
    --option diarization=true \
    "$@" 2>&1 |
    {
        while IFS= read -r line; do
            # The pipeline announces each file as "[index/count] path" before starting it.
            if [[ $line =~ ^\[([0-9]+)/([0-9]+)\]\ (.*)$ ]]; then
                finish_current
                index=${BASH_REMATCH[1]}
                count=${BASH_REMATCH[2]}
                current=$(basename "${BASH_REMATCH[3]}")
                current_started=$(date +%s)
                log "Start  [$index/$count] $current ($(clock "${duration[$current]:-0}") of audio)"
            else
                # The pipeline's closing summary ends the last file.
                if [[ $line == Transcribed* || $line =~ ^warning:\ [0-9]+\ of\ [0-9]+\ files\ failed ]]; then
                    finish_current
                fi
                log "$line"
            fi
        done
        finish_current
    }
status=${PIPESTATUS[0]}
set -e

run=$(comm -13 <(echo "$before") <(ls -1 "$OUT") | grep -v '\.log$' | tail -1 || true)
log "faster-whisper exited with $status after $(clock $(($(date +%s) - started)))."
if [[ -z "$run" ]]; then
    log "No run folder was written to $OUT."
    exit "$status"
fi

if [[ -s "$OUT/$run/errors.json" ]] && [[ $(cat "$OUT/$run/errors.json") != "[]" ]]; then
    log "Some files failed; see $OUT/$run/errors.json"
fi

cat <<EOF

Run written to $OUT/$run. To score it:

  .venv/bin/python -m audio_transcriber.evaluation -l large-v3 \\
      --transcripts "$OUT/$run" --references $INPUT
EOF
exit "$status"
