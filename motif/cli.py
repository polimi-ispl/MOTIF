"""MOTIF command-line interface."""

import argparse
import json

from motif import __version__


def build_parser():
    parser = argparse.ArgumentParser(
        prog="motif",
        description="Person-of-interest deepfake detection from facial geometry and motion.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="auto", help="CUDA device, e.g. cuda:1")
    common.add_argument("--weights-dir", help="directory containing every required weight file")
    common.add_argument("--batch-size", type=int, default=32, help="GPU frame/window batch size")

    extract = subparsers.add_parser("extract-reference", parents=[common])
    extract.add_argument("--reference", nargs="+", required=True, help="genuine reference videos")
    extract.add_argument("--output", "-o", required=True, help="output .npz reference set")

    score = subparsers.add_parser("score", parents=[common])
    references = score.add_mutually_exclusive_group(required=True)
    references.add_argument("--reference", nargs="+", help="genuine reference videos")
    references.add_argument("--reference-set", help="reference .npz from extract-reference")
    score.add_argument("--test", required=True, help="query video")
    score.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def _pipeline(args):
    from motif.pipeline import MotifPipeline

    return MotifPipeline(
        device=args.device,
        weights_dir=args.weights_dir,
        batch_size=args.batch_size,
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    pipeline = _pipeline(args)
    if args.command == "extract-reference":
        reference = pipeline.extract_reference_set(args.reference)
        reference.save(args.output)
        print(args.output)
        return

    if args.reference_set:
        from motif.pipeline import MotifReferenceSet

        reference = MotifReferenceSet.load(args.reference_set)
    else:
        reference = pipeline.extract_reference_set(args.reference)
    result = pipeline.score(reference, args.test)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"score: {result['score']:.6f}")
        print(f"global: {result['global_score']:.6f}")
        print(f"mouth: {result['mouth_score']:.6f}")


if __name__ == "__main__":
    main()
