import argparse
import os

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    def load_dotenv():
        return False


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")


def _collect_images(inputs):
    image_paths = []
    for item in inputs:
        if os.path.isdir(item):
            for root, _, files in os.walk(item):
                for filename in files:
                    if filename.lower().endswith(IMAGE_EXTENSIONS):
                        image_paths.append(os.path.abspath(os.path.join(root, filename)))
        elif os.path.isfile(item):
            image_paths.append(os.path.abspath(item))
        else:
            raise FileNotFoundError(f"Input path not found: {item}")

    return sorted(set(image_paths))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="anomagent",
        description="Multi-step anomaly analysis for AI-generated images (OpenAI API).",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more image files or directories (directories searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output root directory (a timestamped run dir is created inside).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help='OpenAI model name (defaults to env OPENAI_MODEL or "gpt-4o").',
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (defaults to env OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional OpenAI base URL (defaults to env OPENAI_BASE_URL).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Thread pool size used in multi-step calls.",
    )
    parser.add_argument(
        "--num-get-objects",
        type=int,
        default=3,
        help="How many times to call the object discovery step (then merge).",
    )
    parser.add_argument(
        "--num-summarizer-step2",
        type=int,
        default=3,
        help="How many times to call summarizer step2.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=None,
        help="Limit how many discovered objects are analyzed (reduces cost).",
    )
    parser.add_argument(
        "--image-max-side",
        type=int,
        default=2048,
        help="Resize images so the longest side is at most this value (reduces cost).",
    )
    parser.add_argument(
        "--image-format",
        choices=["png", "jpeg"],
        default="png",
        help="Encoding format for image data URLs sent to the API.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality when --image-format=jpeg.",
    )
    parser.add_argument(
        "--run-suffix",
        default="",
        help="Optional suffix appended to the run directory name.",
    )
    parser.add_argument(
        "--continue",
        dest="is_continue",
        action="store_true",
        help="Continue from the latest run directory matching the output prefix.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start index (0-based) into the collected image list.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index (exclusive, 0-based) into the collected image list.",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable logging stdout to a file under the run directory.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser


def main(argv=None):
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args(argv)

    image_paths = _collect_images(args.input)
    if args.start is not None or args.end is not None:
        start = args.start or 0
        end = args.end if args.end is not None else len(image_paths)
        if start < 0 or end < 0 or start > end or end > len(image_paths):
            raise ValueError(f"Invalid slice: start={args.start}, end={args.end}, total={len(image_paths)}")
        image_paths = image_paths[start:end]

    os.makedirs(args.output_dir, exist_ok=True)
    save_root_prefix = os.path.join(args.output_dir, "run")

    from .analyzer import AIImageAnalyzer
    from .config import AnalyzerConfig, ImageEncodingConfig, OpenAIConfig, RetryConfig

    config = AnalyzerConfig(
        max_workers=args.max_workers,
        num_get_objects=args.num_get_objects,
        num_summarizer_step2=args.num_summarizer_step2,
        max_objects=args.max_objects,
        image=ImageEncodingConfig(
            max_side=args.image_max_side,
            image_format=args.image_format,
            jpeg_quality=args.jpeg_quality,
        ),
        retry=RetryConfig(),
        openai=OpenAIConfig(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
        ),
        log_level=args.log_level,
        log_to_file=not args.no_log_file,
    )

    analyzer = AIImageAnalyzer(
        save_root_dir=save_root_prefix,
        config=config,
        is_continue=args.is_continue,
        save_root_dir_suffix=args.run_suffix,
        start="" if args.start is None else str(args.start),
        end="" if args.end is None else str(args.end),
    )

    analyzer.analyze_images(image_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
