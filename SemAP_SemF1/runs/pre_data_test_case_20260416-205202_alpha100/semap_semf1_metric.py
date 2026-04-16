#!/usr/bin/env python3
"""SemAP / SemF1 evaluator based on fused BERTScore matching."""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path
from queue import Empty
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from utiliz import (
    auto_find_largest_json,
    extract_subset_keys,
    normalize_prediction_dict,
    parse_float_csv,
    read_json,
    save_json,
    setup_logging,
)

PredictionEntry = Dict[str, str]
MatchPair = Tuple[int, int, float]
ImageMatches = List[Tuple[int, List[MatchPair]]]

if TYPE_CHECKING:
    from bert_score import BERTScorer


def device_str(use_gpu: bool, rank: int = 0) -> str:
    """Return the torch device string for the current worker."""
    torch = import_torch()
    if use_gpu and torch.cuda.is_available():
        return f"cuda:{rank}"
    return "cpu"


def import_torch():
    """Import torch lazily so CLI help works without runtime dependencies."""
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch could not be imported. Install the dependencies from "
            "`requirements.txt` and ensure your torch build matches the installed NumPy version."
        ) from exc
    return torch


def import_bertscore():
    """Import bert-score lazily and raise a friendly installation error."""
    try:
        from bert_score import BERTScorer
    except Exception as exc:
        raise RuntimeError(
            "`bert-score` is required to run this evaluator. "
            "Install it with `pip install -r requirements.txt`."
        ) from exc
    return BERTScorer


def threshold_key(threshold: float) -> str:
    """Format threshold keys without unnecessary trailing zeros."""
    return f"{threshold:.4f}".rstrip("0").rstrip(".")


def compute_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Compute precision / recall / F1 without epsilon hacks."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "SemAP": precision,
        "SemF1": f1,
    }


def compute_threshold_counts(scores: Sequence[float], num_pred: int, num_gt: int, threshold: float) -> Tuple[int, int, int]:
    """Convert match scores into TP / FP / FN counts for one threshold."""
    tp = sum(1 for score in scores if score >= threshold)
    fp = max(0, num_pred - tp)
    fn = max(0, num_gt - tp)
    return tp, fp, fn


def fused_similarity_matrix_bertscore_single(
    scorer: BERTScorer,
    preds: Sequence[PredictionEntry],
    gts: Sequence[PredictionEntry],
    alpha: float = 0.5,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Build an M x N fused BERTScore matrix for one image.

    Score = alpha * BERTScore(observed) + (1 - alpha) * BERTScore(reasoning)
    """
    num_pred = len(preds)
    num_gt = len(gts)
    if num_pred == 0 or num_gt == 0:
        return np.zeros((num_pred, num_gt), dtype=np.float32)

    gt_observed = [gt.get("Observed Phenomenon", "") for gt in gts]
    gt_reasoning = [gt.get("Reasoning", "") for gt in gts]

    observed_scores = np.zeros((num_pred, num_gt), dtype=np.float32)
    reasoning_scores = np.zeros((num_pred, num_gt), dtype=np.float32)

    for pred_index, pred in enumerate(preds):
        repeated_observed = [pred.get("Observed Phenomenon", "")] * num_gt
        repeated_reasoning = [pred.get("Reasoning", "")] * num_gt

        _, _, observed_f1 = scorer.score(
            repeated_observed,
            gt_observed,
            batch_size=batch_size,
            verbose=False,
        )
        _, _, reasoning_f1 = scorer.score(
            repeated_reasoning,
            gt_reasoning,
            batch_size=batch_size,
            verbose=False,
        )

        observed_scores[pred_index, :] = observed_f1.detach().cpu().numpy()
        reasoning_scores[pred_index, :] = reasoning_f1.detach().cpu().numpy()

    fused_scores = alpha * observed_scores + (1.0 - alpha) * reasoning_scores
    return np.clip(fused_scores, 0.0, 1.0)


def greedy_match_from_matrix(score_matrix: np.ndarray) -> List[MatchPair]:
    """Perform one-to-one greedy matching on an M x N score matrix."""
    num_pred, num_gt = score_matrix.shape
    used_pred = set()
    used_gt = set()
    pairs: List[MatchPair] = []

    while len(used_pred) < num_pred and len(used_gt) < num_gt:
        best_score = -1.0
        best_pred = -1
        best_gt = -1

        for pred_index in range(num_pred):
            if pred_index in used_pred:
                continue
            row = score_matrix[pred_index]
            for gt_index in range(num_gt):
                if gt_index in used_gt:
                    continue
                candidate_score = float(row[gt_index])
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_pred = pred_index
                    best_gt = gt_index

        if best_score <= 0.0:
            break

        used_pred.add(best_pred)
        used_gt.add(best_gt)
        pairs.append((best_pred, best_gt, best_score))

    return pairs


def summarize_map(
    all_pairs: ImageMatches,
    dataset_preds: Sequence[Sequence[PredictionEntry]],
    dataset_gts: Sequence[Sequence[PredictionEntry]],
    thresholds: Sequence[float],
    image_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Summarize SemAP / SemF1 over images and thresholds."""
    pairs_per_image = {index: pairs for index, pairs in all_pairs}
    global_counts = {threshold: {"TP": 0, "FP": 0, "FN": 0} for threshold in thresholds}
    per_image_metrics: Dict[str, Dict[str, Any]] = {}

    per_image_semap: List[float] = []
    per_image_semf1: List[float] = []

    for image_index in range(len(dataset_preds)):
        pairs = pairs_per_image.get(image_index, [])
        scores = [score for _, _, score in pairs]
        num_pred = len(dataset_preds[image_index])
        num_gt = len(dataset_gts[image_index])
        image_id = image_keys[image_index] if image_keys else str(image_index)

        threshold_metrics: Dict[str, Dict[str, float]] = {}
        image_semap_values: List[float] = []
        image_semf1_values: List[float] = []

        for threshold in thresholds:
            tp, fp, fn = compute_threshold_counts(scores, num_pred, num_gt, threshold)
            metrics = compute_prf(tp, fp, fn)
            global_counts[threshold]["TP"] += tp
            global_counts[threshold]["FP"] += fp
            global_counts[threshold]["FN"] += fn

            metrics_with_counts = {
                **{key: round(value, 4) for key, value in metrics.items()},
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "AP": round(metrics["SemAP"], 4),
            }
            threshold_metrics[threshold_key(threshold)] = metrics_with_counts
            image_semap_values.append(metrics["SemAP"])
            image_semf1_values.append(metrics["SemF1"])

        image_semap = float(np.mean(image_semap_values)) if image_semap_values else 0.0
        image_semf1 = float(np.mean(image_semf1_values)) if image_semf1_values else 0.0
        per_image_semap.append(image_semap)
        per_image_semf1.append(image_semf1)

        per_image_metrics[image_id] = {
            "num_predictions": num_pred,
            "num_ground_truths": num_gt,
            "matched_pairs": len(pairs),
            "SemAP": round(image_semap, 4),
            "SemF1": round(image_semf1, 4),
            "by_threshold": threshold_metrics,
        }

    by_threshold: Dict[str, Dict[str, float]] = {}
    for threshold in thresholds:
        counts = global_counts[threshold]
        metrics = compute_prf(counts["TP"], counts["FP"], counts["FN"])
        by_threshold[threshold_key(threshold)] = {
            **{key: round(value, 4) for key, value in metrics.items()},
            "TP": counts["TP"],
            "FP": counts["FP"],
            "FN": counts["FN"],
            "AP": round(metrics["SemAP"], 4),
        }

    semap = float(np.mean(per_image_semap)) if per_image_semap else 0.0
    semf1 = float(np.mean(per_image_semf1)) if per_image_semf1 else 0.0
    return {
        "metric_name": "SemAP / SemF1",
        "metric_notes": {
            "matching": "Greedy one-to-one matching on fused BERTScore.",
            "fusion": "alpha * observed + (1 - alpha) * reasoning.",
            "SemAP_definition": "In this implementation, SemAP is the mean of single-threshold precision values after greedy matching. It is not ranking-based AP.",
        },
        "num_images": len(dataset_preds),
        "thresholds": list(thresholds),
        "by_threshold": by_threshold,
        "SemAP": round(semap, 4),
        "SemF1": round(semf1, 4),
        "mAP": round(semap, 4),
        "mF1": round(semf1, 4),
        "per_image": per_image_metrics,
    }


def serialize_match_details(
    image_keys: Sequence[str],
    dataset_preds_dict: Dict[str, List[PredictionEntry]],
    dataset_gts_dict: Dict[str, List[PredictionEntry]],
    match_details: ImageMatches,
) -> List[Dict[str, Any]]:
    """Convert tuple-based match results into JSON-friendly records."""
    match_map = {index: pairs for index, pairs in match_details}
    serialized: List[Dict[str, Any]] = []

    for image_index, image_id in enumerate(image_keys):
        preds = dataset_preds_dict[image_id]
        gts = dataset_gts_dict[image_id]
        pairs = match_map.get(image_index, [])

        matched_pred_indices = {pred_index for pred_index, _, _ in pairs}
        matched_gt_indices = {gt_index for _, gt_index, _ in pairs}

        serialized.append(
            {
                "image_index": image_index,
                "image_id": image_id,
                "num_predictions": len(preds),
                "num_ground_truths": len(gts),
                "pairs": [
                    {
                        "pred_index": pred_index,
                        "gt_index": gt_index,
                        "score": round(score, 6),
                    }
                    for pred_index, gt_index, score in pairs
                ],
                "unmatched_prediction_indices": [
                    pred_index for pred_index in range(len(preds)) if pred_index not in matched_pred_indices
                ],
                "unmatched_ground_truth_indices": [
                    gt_index for gt_index in range(len(gts)) if gt_index not in matched_gt_indices
                ],
            }
        )

    return serialized


def format_match_details_txt(
    image_keys: Sequence[str],
    dataset_preds_dict: Dict[str, List[PredictionEntry]],
    dataset_gts_dict: Dict[str, List[PredictionEntry]],
    match_details: ImageMatches,
    output_txt_path: str | Path,
    thresholds: Sequence[float],
) -> None:
    """Write a plain-text, human-readable match report."""
    match_map = {index: pairs for index, pairs in match_details}
    lines: List[str] = []

    for image_index, image_id in enumerate(image_keys):
        preds = dataset_preds_dict[image_id]
        gts = dataset_gts_dict[image_id]
        pairs = match_map.get(image_index, [])
        matched_pred_indices = {pred_index for pred_index, _, _ in pairs}
        matched_gt_indices = {gt_index for _, gt_index, _ in pairs}
        scores = [score for _, _, score in pairs]

        lines.append(f"Image ID: {image_id}")
        lines.append("-" * 72)

        if pairs:
            lines.append("Matched pairs:")
            for pair_index, (pred_index, gt_index, score) in enumerate(pairs, start=1):
                pred = preds[pred_index]
                gt = gts[gt_index]
                lines.append(f"  Match #{pair_index}")
                lines.append(f"    Pred idx: {pred_index}")
                lines.append(f"    GT idx  : {gt_index}")
                lines.append(f"    Score   : {score:.4f}")
                lines.append(f"    Pred Observed: {pred.get('Observed Phenomenon', '').strip()}")
                lines.append(f"    Pred Reasoning: {pred.get('Reasoning', '').strip()}")
                lines.append(f"    GT Observed  : {gt.get('Observed Phenomenon', '').strip()}")
                lines.append(f"    GT Reasoning : {gt.get('Reasoning', '').strip()}")
        else:
            lines.append("Matched pairs: none")

        unmatched_pred_indices = [index for index in range(len(preds)) if index not in matched_pred_indices]
        if unmatched_pred_indices:
            lines.append("Unmatched predictions:")
            for pred_index in unmatched_pred_indices:
                pred = preds[pred_index]
                lines.append(f"  Pred idx: {pred_index}")
                lines.append(f"    Observed : {pred.get('Observed Phenomenon', '').strip()}")
                lines.append(f"    Reasoning: {pred.get('Reasoning', '').strip()}")

        unmatched_gt_indices = [index for index in range(len(gts)) if index not in matched_gt_indices]
        if unmatched_gt_indices:
            lines.append("Unmatched ground truths:")
            for gt_index in unmatched_gt_indices:
                gt = gts[gt_index]
                lines.append(f"  GT idx: {gt_index}")
                lines.append(f"    Observed : {gt.get('Observed Phenomenon', '').strip()}")
                lines.append(f"    Reasoning: {gt.get('Reasoning', '').strip()}")

        lines.append("Per-threshold metrics:")
        for threshold in thresholds:
            tp, fp, fn = compute_threshold_counts(scores, len(preds), len(gts), threshold)
            metrics = compute_prf(tp, fp, fn)
            lines.append(
                "  "
                f"threshold={threshold_key(threshold)} "
                f"Precision={metrics['Precision']:.4f} "
                f"Recall={metrics['Recall']:.4f} "
                f"SemF1={metrics['SemF1']:.4f} "
                f"SemAP={metrics['SemAP']:.4f}"
            )

        lines.append("=" * 72)
        lines.append("")

    output_txt_path = Path(output_txt_path)
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    output_txt_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Saved match report: %s", output_txt_path)


def worker_proc(
    rank: int,
    num_gpus: int,
    dataset_preds: Sequence[Sequence[PredictionEntry]],
    dataset_gts: Sequence[Sequence[PredictionEntry]],
    alpha: float,
    bertscore_model: str,
    use_gpu: bool,
    bertscore_batch_size: int,
    progress_queue: mp.Queue,
    result_queue: mp.Queue,
    show_per_gpu_progress: bool,
) -> None:
    """Worker process for one GPU."""
    try:
        BERTScorer = import_bertscore()
        device = device_str(use_gpu, rank)
        scorer = BERTScorer(
            model_type=bertscore_model,
            lang="en",
            rescale_with_baseline=False,
            device=device,
        )

        assigned_indices = list(range(rank, len(dataset_preds), num_gpus))
        local_results: List[Tuple[int, List[MatchPair]]] = []
        progress_bar = None

        if show_per_gpu_progress:
            progress_bar = tqdm(
                total=len(assigned_indices),
                desc=f"GPU{rank} ({device})",
                position=rank + 1,
                leave=True,
                dynamic_ncols=True,
            )

        for image_index in assigned_indices:
            preds = dataset_preds[image_index]
            gts = dataset_gts[image_index]
            if not preds or not gts:
                local_results.append((image_index, []))
                progress_queue.put(1)
                if progress_bar is not None:
                    progress_bar.update(1)
                continue

            score_matrix = fused_similarity_matrix_bertscore_single(
                scorer=scorer,
                preds=preds,
                gts=gts,
                alpha=alpha,
                batch_size=bertscore_batch_size,
            )
            local_results.append((image_index, greedy_match_from_matrix(score_matrix)))
            progress_queue.put(1)
            if progress_bar is not None:
                progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.close()
        result_queue.put(("RESULTS", local_results))
    except Exception as exc:
        result_queue.put(("ERROR", f"Worker {rank} failed: {exc!r}"))


def evaluate_multigpu(
    dataset_preds: Sequence[Sequence[PredictionEntry]],
    dataset_gts: Sequence[Sequence[PredictionEntry]],
    thresholds: Sequence[float],
    alpha: float,
    use_gpu: bool,
    num_gpus: Optional[int],
    bertscore_model: str,
    bertscore_batch_size: int,
    show_per_gpu_progress: bool,
    image_keys: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Any], ImageMatches]:
    """Run the SemAP / SemF1 evaluation on CPU or across multiple GPUs."""
    torch = import_torch()
    BERTScorer = import_bertscore()

    if len(dataset_preds) != len(dataset_gts):
        raise ValueError("Prediction and ground-truth image counts do not match.")

    num_images = len(dataset_preds)
    if not use_gpu or not torch.cuda.is_available():
        device = device_str(use_gpu, 0)
        logging.info("Running single-process evaluation on %s", device)
        scorer = BERTScorer(
            model_type=bertscore_model,
            lang="en",
            rescale_with_baseline=False,
            device=device,
        )
        results: ImageMatches = []
        for image_index in tqdm(range(num_images), desc="Evaluating", position=0):
            preds = dataset_preds[image_index]
            gts = dataset_gts[image_index]
            if not preds or not gts:
                results.append((image_index, []))
                continue
            score_matrix = fused_similarity_matrix_bertscore_single(
                scorer=scorer,
                preds=preds,
                gts=gts,
                alpha=alpha,
                batch_size=bertscore_batch_size,
            )
            results.append((image_index, greedy_match_from_matrix(score_matrix)))
        summary = summarize_map(results, dataset_preds, dataset_gts, thresholds, image_keys=image_keys)
        return summary, sorted(results, key=lambda item: item[0])

    available_gpus = torch.cuda.device_count()
    active_gpus = available_gpus if num_gpus is None else max(1, min(num_gpus, available_gpus))
    logging.info("Running multi-GPU evaluation on %s / %s GPUs", active_gpus, available_gpus)

    ctx = mp.get_context("spawn")
    progress_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()
    processes: List[mp.Process] = []

    for rank in range(active_gpus):
        process = ctx.Process(
            target=worker_proc,
            args=(
                rank,
                active_gpus,
                dataset_preds,
                dataset_gts,
                alpha,
                bertscore_model,
                True,
                bertscore_batch_size,
                progress_queue,
                result_queue,
                show_per_gpu_progress,
            ),
        )
        process.daemon = True
        process.start()
        processes.append(process)

    results: ImageMatches = []
    finished_images = 0
    received_workers = 0
    error_messages: List[str] = []

    with tqdm(total=num_images, desc=f"Evaluating ({active_gpus} GPUs)", position=0, leave=True) as progress_bar:
        while received_workers < active_gpus:
            try:
                increment = progress_queue.get(timeout=0.1)
                finished_images += increment
                progress_bar.update(increment)
            except Empty:
                pass

            drained_queue = True
            while drained_queue:
                try:
                    tag, payload = result_queue.get_nowait()
                except Empty:
                    drained_queue = False
                    continue

                if tag == "RESULTS":
                    results.extend(payload)
                    received_workers += 1
                elif tag == "ERROR":
                    error_messages.append(payload)
                    received_workers += 1

            if all(not process.is_alive() for process in processes) and received_workers >= active_gpus:
                break

            if finished_images >= num_images and received_workers >= active_gpus:
                break

            time.sleep(0.01)

    for process in processes:
        process.join()

    while True:
        try:
            tag, payload = result_queue.get_nowait()
        except Empty:
            break
        if tag == "RESULTS":
            results.extend(payload)
        elif tag == "ERROR":
            error_messages.append(payload)

    if error_messages:
        for message in error_messages:
            logging.error(message)
        raise RuntimeError("One or more worker processes failed.")

    results = sorted({image_index: pairs for image_index, pairs in results}.items(), key=lambda item: item[0])
    summary = summarize_map(results, dataset_preds, dataset_gts, thresholds, image_keys=image_keys)
    return summary, results


def snapshot_sources(save_dir: Path) -> None:
    """Save snapshots of the public release scripts for reproducibility."""
    for source_path in (Path(__file__), Path(__file__).with_name("utiliz.py")):
        if source_path.exists():
            shutil.copy2(source_path, save_dir / source_path.name)


def resolve_alphas(args: argparse.Namespace) -> List[float]:
    """Resolve one or more alpha values from CLI arguments."""
    if args.alphas:
        alphas = parse_float_csv(args.alphas, field_name="alphas")
    else:
        alphas = [args.alpha]

    for alpha in alphas:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be within [0, 1], but received {alpha}.")
    return alphas


def resolve_pred_json(args: argparse.Namespace) -> Path:
    """Resolve the prediction JSON path, optionally by auto-detection."""
    if args.pred_json:
        return Path(args.pred_json)
    if args.auto_find_pred_json:
        detected = auto_find_largest_json(Path.cwd())
        if detected is None:
            raise FileNotFoundError("No JSON file was found under the current working directory.")
        logging.info("Auto-detected prediction JSON: %s", detected)
        return detected
    raise ValueError("`--pred_json` is required unless `--auto_find_pred_json` is enabled.")


def prepare_eval_inputs(
    pred_json: Path,
    gt_json: Path,
    subset_json: Optional[Path] = None,
) -> Tuple[List[str], List[List[PredictionEntry]], List[List[PredictionEntry]], Dict[str, List[PredictionEntry]], Dict[str, List[PredictionEntry]]]:
    """Load and align predictions, ground truth, and optional subset keys."""
    preds_dict = normalize_prediction_dict(read_json(pred_json))
    gts_dict = normalize_prediction_dict(read_json(gt_json))
    subset_keys: Optional[List[str]] = None

    if subset_json is not None:
        subset_keys = extract_subset_keys(read_json(subset_json))
        common_keys = [key for key in subset_keys if key in preds_dict and key in gts_dict]
        missing_pred = [key for key in subset_keys if key not in preds_dict]
        missing_gt = [key for key in subset_keys if key not in gts_dict]
        if missing_pred:
            logging.warning("Subset keys missing from predictions: %d", len(missing_pred))
        if missing_gt:
            logging.warning("Subset keys missing from ground truth: %d", len(missing_gt))
    else:
        common_keys = sorted(set(preds_dict) & set(gts_dict))

    if not common_keys:
        raise ValueError("No overlapping image ids were found between predictions and ground truth.")

    dataset_preds = [preds_dict[key] for key in common_keys]
    dataset_gts = [gts_dict[key] for key in common_keys]

    logging.info("Aligned images: %d", len(common_keys))
    logging.info("Prediction images: %d", len(preds_dict))
    logging.info("Ground-truth images: %d", len(gts_dict))
    if subset_keys is not None:
        logging.info("Subset images (requested): %d", len(subset_keys))
    return common_keys, dataset_preds, dataset_gts, preds_dict, gts_dict


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="SemAP / SemF1 evaluator with optional multi-GPU BERTScore.")
    parser.add_argument("--pred_json", type=str, default=None, help="Prediction JSON path.")
    parser.add_argument("--gt_json", type=str, required=True, help="Ground-truth JSON path.")
    parser.add_argument(
        "--subset_json",
        type=str,
        default=None,
        help="Optional subset split JSON. Supports a dict, a list of image ids, or a list of single-key dicts.",
    )
    parser.add_argument(
        "--auto_find_pred_json",
        action="store_true",
        help="If set, automatically use the largest JSON file under the current working directory as prediction JSON.",
    )
    parser.add_argument("--output_root", type=str, default="runs", help="Directory for evaluation outputs.")
    parser.add_argument("--out_metrics", type=str, default="metrics.json", help="Metrics JSON filename.")
    parser.add_argument(
        "--out_matches",
        type=str,
        default="matched_results.json",
        help="Match-details JSON filename. Use an empty string to disable match exports.",
    )
    parser.add_argument("--thresholds", type=str, default="0.7,0.8,0.9", help="Comma-separated thresholds.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Fusion weight for observed vs reasoning.")
    parser.add_argument(
        "--alphas",
        type=str,
        default=None,
        help="Optional comma-separated alpha list for multiple runs. Overrides `--alpha` when provided.",
    )
    parser.add_argument("--use_gpu", action="store_true", help="Enable GPU evaluation when CUDA is available.")
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use. Default: all visible GPUs.")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased", help="Backbone for BERTScore.")
    parser.add_argument("--batch_size", type=int, default=64, help="BERTScore batch size.")
    parser.add_argument(
        "--no_per_gpu_progress",
        action="store_true",
        help="Disable per-GPU progress bars and show only the global progress bar.",
    )
    return parser.parse_args()


def run_single_alpha(
    args: argparse.Namespace,
    alpha: float,
    pred_json: Path,
    gt_json: Path,
    subset_json: Optional[Path],
    thresholds: Sequence[float],
) -> Path:
    """Run evaluation for one alpha value and save artifacts."""
    pred_base = pred_json.stem or "preds"
    alpha_tag = str(int(round(alpha * 100))).zfill(2)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = Path(args.output_root) / f"{pred_base}_{timestamp}_alpha{alpha_tag}"
    setup_logging(save_dir)
    snapshot_sources(save_dir)

    logging.info("SemAP / SemF1 evaluation started.")
    logging.info("Command: %s", " ".join(sys.argv))
    logging.info("Prediction JSON: %s", pred_json)
    logging.info("Ground-truth JSON: %s", gt_json)
    if subset_json is not None:
        logging.info("Subset JSON: %s", subset_json)
    logging.info("Alpha: %.4f", alpha)
    logging.info("Thresholds: %s", ", ".join(threshold_key(value) for value in thresholds))
    logging.info("BERTScore model: %s", args.model)

    image_keys, dataset_preds, dataset_gts, preds_dict, gts_dict = prepare_eval_inputs(
        pred_json=pred_json,
        gt_json=gt_json,
        subset_json=subset_json,
    )

    summary, match_details = evaluate_multigpu(
        dataset_preds=dataset_preds,
        dataset_gts=dataset_gts,
        thresholds=thresholds,
        alpha=alpha,
        use_gpu=bool(args.use_gpu),
        num_gpus=args.num_gpus,
        bertscore_model=args.model,
        bertscore_batch_size=args.batch_size,
        show_per_gpu_progress=not args.no_per_gpu_progress,
        image_keys=image_keys,
    )
    summary["alpha"] = alpha
    summary["bertscore_model"] = args.model
    summary["batch_size"] = args.batch_size
    summary["use_gpu"] = bool(args.use_gpu)
    summary["num_gpus"] = args.num_gpus

    save_json(summary, save_dir / (Path(args.out_metrics).name or "metrics.json"))
    save_json(
        {
            "pred_json": str(pred_json),
            "gt_json": str(gt_json),
            "subset_json": str(subset_json) if subset_json else None,
            "thresholds": list(thresholds),
            "alpha": alpha,
            "use_gpu": bool(args.use_gpu),
            "num_gpus": args.num_gpus,
            "model": args.model,
            "batch_size": args.batch_size,
        },
        save_dir / "run_config.json",
    )

    if args.out_matches:
        serialized_matches = serialize_match_details(image_keys, preds_dict, gts_dict, match_details)
        save_json(serialized_matches, save_dir / (Path(args.out_matches).name or "matched_results.json"))
        format_match_details_txt(
            image_keys=image_keys,
            dataset_preds_dict=preds_dict,
            dataset_gts_dict=gts_dict,
            match_details=match_details,
            output_txt_path=save_dir / "matched_results_pretty.txt",
            thresholds=thresholds,
        )
    else:
        logging.info("Skipping match export because `--out_matches` is empty.")

    logging.info("Finished run. Output directory: %s", save_dir)
    return save_dir


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    thresholds = parse_float_csv(args.thresholds, field_name="thresholds")
    pred_json = resolve_pred_json(args)
    gt_json = Path(args.gt_json)
    subset_json = Path(args.subset_json) if args.subset_json else None

    output_dirs = []
    for alpha in resolve_alphas(args):
        output_dirs.append(
            run_single_alpha(
                args=args,
                alpha=alpha,
                pred_json=pred_json,
                gt_json=gt_json,
                subset_json=subset_json,
                thresholds=thresholds,
            )
        )

    print("Completed SemAP / SemF1 evaluation.")
    for output_dir in output_dirs:
        print(f"- {output_dir}")


if __name__ == "__main__":
    main()
