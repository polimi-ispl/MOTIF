"""Enroll DiCaprio references and score the fixed genuine and fake queries."""

import argparse
import json
from pathlib import Path

from motif import __version__
from motif.pipeline import MotifPipeline

DEMO_DIR = Path(__file__).resolve().parent


def auc(real_scores, fake_scores):
    """Fraction of genuine/fake pairs ranked correctly; ties count half."""
    wins = sum((real > fake) + 0.5 * (real == fake)
               for real in real_scores for fake in fake_scores)
    return wins / (len(real_scores) * len(fake_scores))


def find_clips(clips_dir):
    """Locate the eight references and seven fixed test clips."""
    groups = [
        [clips_dir / f"{prefix}_{index}.mp4" for index in range(1, count + 1)]
        for prefix, count in (("ref", 8), ("test_real", 3), ("test_fake", 4))
    ]
    missing = [path.name for group in groups for path in group if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing demo clips in {clips_dir}: {', '.join(missing)}. "
            "Run ./download.sh && ./prepare_clips.sh from demo/."
        )
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clips",
        type=Path,
        default=DEMO_DIR / "clips",
        help="Directory containing the prepared demo clips",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEMO_DIR / "results",
        help="Output directory for scores.json",
    )
    parser.add_argument("--device", default="auto", help='"auto" or "cuda:N"')
    parser.add_argument("--batch-size", type=int, default=8, help="GPU batch size")
    args = parser.parse_args()

    clips_dir = args.clips.expanduser()
    try:
        references, reals, fakes = find_clips(clips_dir)
    except FileNotFoundError as error:
        parser.error(str(error))

    output_dir = args.out.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MOTIF and encoding 8 genuine references...", flush=True)
    pipeline = MotifPipeline(device=args.device, batch_size=args.batch_size)
    reference_set = pipeline.extract_reference_set(references)

    results = []
    queries = [(path, "real") for path in reals] + [
        (path, "fake") for path in fakes
    ]
    for path, label in queries:
        scores = pipeline.score(reference_set, path)
        result = {"clip": path.name, "label": label, **scores}
        results.append(result)
        print(
            f"  {path.name:<20} fused={scores['score']:+.6f} "
            f"global={scores['global_score']:+.6f} "
            f"mouth={scores['mouth_score']:+.6f}",
            flush=True,
        )

    real_scores = [row["score"] for row in results if row["label"] == "real"]
    fake_scores = [row["score"] for row in results if row["label"] == "fake"]
    area = auc(real_scores, fake_scores)
    margin = min(real_scores) - max(fake_scores)
    print(f"AUC: {area:.3f}; margin (worst real - best fake): {margin:+.6f}")
    print("Scores rank identity consistency; they are not calibrated probabilities.")

    summary = {
        "motif_version": __version__,
        "device": str(pipeline.device),
        "batch_size": args.batch_size,
        "reference_clips": [path.name for path in references],
        "num_reference_videos": len(references),
        "auc": area,
        "margin": margin,
        "results": sorted(results, key=lambda row: -row["score"]),
    }
    destination = output_dir / "scores.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
