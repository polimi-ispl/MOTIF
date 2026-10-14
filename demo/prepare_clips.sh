#!/usr/bin/env bash
# Cut the downloaded source videos into the fixed MOTIF demo cohort.
#
# Usage: ./prepare_clips.sh
#   INPUT_DIR  source videos [default: downloads]
#   OUTPUT_DIR prepared clips [default: clips]

set -euo pipefail

INPUT_DIR="${INPUT_DIR:-downloads}"
OUTPUT_DIR="${OUTPUT_DIR:-clips}"
RECORDS=()

clip() {
    local set="$1"; shift
    local video="$1"; shift
    local record="$set"$'\t'"$video"
    local range
    for range in "$@"; do
        record+=$'\t'"$range"
    done
    RECORDS+=("$record")
}

# Eight genuine reference clips from three source videos.
clip ref usqUrdA_R3I 00:00:08-00:00:13 00:00:22-00:00:24 00:00:36-00:00:41
clip ref xpyrefzvTpI 00:02:22-00:02:26 00:03:05-00:03:10 00:03:27-00:03:33
clip ref IHkJzVx0Jd4 00:04:29-00:04:32 00:05:55-00:06:00

# Three independent genuine queries.
clip test_real lkUKu0NZhb4 00:05:54-00:06:03
clip test_real Q_cV9ciktoQ 00:01:20-00:01:25
clip test_real UZh42eXTq4Y 00:00:07-00:00:11

# Four deepfake queries.
clip test_fake 7364145113347493151 00:00:00-00:00:04
clip test_fake 7312874924840946974
clip test_fake 7312146227548785951
clip test_fake 7310263372467997982

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "error: ffmpeg is not installed or not on PATH" >&2
    exit 1
fi
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "error: input dir '$INPUT_DIR' not found" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "Clearing old generated clips in '$OUTPUT_DIR'..."
rm -f "$OUTPUT_DIR"/ref_*.mp4 "$OUTPUT_DIR"/test_real_*.mp4 "$OUTPUT_DIR"/test_fake_*.mp4

encode() {
    local input="$1" output="$2" start="${3:-}" end="${4:-}"
    local seek=()
    [[ -n "$start" ]] && seek=(-ss "$start" -to "$end")
    # Re-encode rather than stream-copy so non-keyframe cuts do not decode as
    # duplicated or black leading frames.
    ffmpeg -hide_banner -loglevel error -y \
        "${seek[@]}" -i "$input" \
        -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p \
        -an "$output"
}

resolve_video() {
    local needle="$1"
    local matches=()
    local file
    for file in "$INPUT_DIR"/*"$needle"*.mp4; do
        [[ -e "$file" ]] && matches+=("$file")
    done
    if [[ "${#matches[@]}" -eq 0 ]]; then
        echo "error: no video in '$INPUT_DIR' matching '$needle'" >&2
        return 1
    fi
    if [[ "${#matches[@]}" -gt 1 ]]; then
        echo "error: '$needle' matches multiple files:" >&2
        printf '  %s\n' "${matches[@]}" >&2
        return 1
    fi
    printf '%s' "${matches[0]}"
}

declare -A COUNT=([ref]=0 [test_real]=0 [test_fake]=0)
made=0
for record in "${RECORDS[@]}"; do
    IFS=$'\t' read -r -a fields <<<"$record"
    set_name="${fields[0]}"
    video_id="${fields[1]}"
    ranges=("${fields[@]:2}")

    if [[ -z "${COUNT[$set_name]+x}" ]]; then
        echo "error: unknown set '$set_name'" >&2
        exit 1
    fi

    input="$(resolve_video "$video_id")"
    echo
    echo "==> $set_name <- $(basename "$input")"

    if [[ "${#ranges[@]}" -eq 0 ]]; then
        number=$(( ++COUNT[$set_name] ))
        output="$OUTPUT_DIR/${set_name}_${number}.mp4"
        echo "    [whole video] -> $(basename "$output")"
        encode "$input" "$output"
        made=$(( made + 1 ))
    else
        for range in "${ranges[@]}"; do
            start="${range%%-*}"
            end="${range#*-}"
            if [[ "$start" == "$range" || -z "$start" || -z "$end" ]]; then
                echo "error: bad range '$range' (expected start-end)" >&2
                exit 1
            fi
            number=$(( ++COUNT[$set_name] ))
            output="$OUTPUT_DIR/${set_name}_${number}.mp4"
            echo "    [$start -> $end] -> $(basename "$output")"
            encode "$input" "$output" "$start" "$end"
            made=$(( made + 1 ))
        done
    fi
done

echo
echo "Verifying decoded frame variation..."
broken=0
for file in "$OUTPUT_DIR"/ref_*.mp4 "$OUTPUT_DIR"/test_real_*.mp4 "$OUTPUT_DIR"/test_fake_*.mp4; do
    [[ -e "$file" ]] || continue
    distinct=$(ffmpeg -v error -i "$file" -frames:v 15 -f framehash -hash md5 - 2>/dev/null \
        | grep -v '^#' | awk -F, '{print $NF}' | sort -u | wc -l)
    if [[ "$distinct" -lt 5 ]]; then
        echo "  BROKEN: $(basename "$file") - only $distinct distinct frame(s) of 15" >&2
        broken=$(( broken + 1 ))
    fi
done
if [[ "$broken" -gt 0 ]]; then
    echo "  $broken clip(s) decode to duplicate frames; do not use these results" >&2
    exit 1
fi
echo "  all clips OK"

echo
echo "Done. Created $made clips in '$OUTPUT_DIR':"
echo "  ref:       ${COUNT[ref]}"
echo "  test_real: ${COUNT[test_real]}"
echo "  test_fake: ${COUNT[test_fake]}"
