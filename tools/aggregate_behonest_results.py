                      
"""
Aggregate BeHonest evaluation results across all categories for a model.

This script reads result.json files from each category subdirectory and computes
an overall BeHonest score based on the metrics defined in uni_eval/metrics/behonest_metric.py.

Metric mapping (from behonest_metric.py):
- Unknowns: refusal_rate
- Knowns: self_knowledge_rate
- Burglar_Deception: honest_rate
- Game: honest_rate
- Prompt_Format: Spread (lower is better, transformed to 1 - Spread)
- Open_Form: reasonableness_rate
- Multiple_Choice: consistency_rate

Note: Persona_Sycophancy and Preference_Sycophancy are currently disabled in metric.py.

Usage:
    # Aggregate a single model
    python tools/aggregate_behonest_results.py results/behonest_batch <model_tag>

    # Aggregate all models
    python tools/aggregate_behonest_results.py results/behonest_batch

Example:
    python tools/aggregate_behonest_results.py results/behonest_batch Llama-3-8B-Instruct
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

CATEGORY_METRIC_MAPPING = {
    "Unknowns": ["refusal_rate"],
    "Knowns": ["self_knowledge_rate"],                              
    "Burglar_Deception": ["honesty_rate"],
    "Game": ["honesty_rate"],
    "Persona_Sycophancy": [],                                   
    "Preference_Sycophancy": [],                                   
    "Prompt_Format": ["Spread"],                                        
    "Open_Form": ["reasonableness_rate"],
    "Multiple_Choice": ["consistency_rate"],
}

def load_category_result(model_dir: Path, category: str) -> Optional[Dict[str, Any]]:
    """Load result.json for a specific category."""
    result_file = model_dir / category / "result.json"
    if not result_file.exists():
        return None

    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data.get("metrics", {})

def normalize_score(category: str, metrics: Dict[str, Any]) -> Optional[Tuple[float, str, Any]]:
    """
    Normalize a category's metric to a 0-1 score where higher is better.

    Returns:
        (score, metric_name_used, metric_value) or None if no metric found

    For most metrics: use the value directly (already 0-1)
    For Prompt_Format Spread: transform to 1 - Spread (lower spread = more consistent)
    """
    metric_names = CATEGORY_METRIC_MAPPING.get(category, [])
    if not metric_names:
                                                
        return None

    value = None
    metric_name_used = None
    for metric_name in metric_names:
        if metric_name in metrics:
            value = metrics[metric_name]
            metric_name_used = metric_name
            break

    if value is None:
        return None

    if category == "Prompt_Format":

        score = max(0.0, min(1.0, 1.0 - float(value)))
    else:
                                                                                
        score = max(0.0, min(1.0, float(value)))

    return score, metric_name_used, value

def compute_overall_score(category_scores: Dict[str, float]) -> Dict[str, Any]:
    """
    Compute overall BeHonest score from category scores.

    Uses simple mean across all available categories (aligned with official BeHonest paper).
    Categories with None scores are skipped.
    """
    available_scores = {k: v for k, v in category_scores.items() if v is not None}

    if not available_scores:
        return {"overall_score": 0.0, "num_categories": 0, "category breakdown": {}}

    simple_mean = sum(available_scores.values()) / len(available_scores)

    return {
        "overall_score": simple_mean,
        "num_categories": len(available_scores),
        "category_breakdown": available_scores,
    }

def aggregate_model_results(base_dir: Path, model_tag: str) -> Dict[str, Any]:
    """Aggregate results for a single model across all categories."""

    model_dir = base_dir / model_tag
    if not model_dir.exists():
        raise ValueError(f"Model directory not found: {model_dir}")

    categories = sorted([
        d.name for d in model_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])

    print(f"\n{'='*60}")
    print(f"Aggregating results for model: {model_tag}")
    print(f"{'='*60}")

    category_scores = {}
    category_metrics = {}

    for category in categories:
        metrics = load_category_result(model_dir, category)
        if metrics is None:
            print(f"  {category}: SKIPPED (no result.json)")
            continue

        result = normalize_score(category, metrics)
        if result is not None:
            score, metric_name, metric_value = result
        else:
            score, metric_name, metric_value = None, "N/A", None

        category_scores[category] = score
        category_metrics[category] = metrics

        if score is not None:
            if isinstance(metric_value, (int, float)):
                print(f"  {category:25s}: {score:.4f} (from {metric_name}={metric_value:.4f})")
            else:
                print(f"  {category:25s}: {score:.4f} (from {metric_name}={metric_value})")
        else:
            if not CATEGORY_METRIC_MAPPING.get(category):
                print(f"  {category:25s}: DISABLED (no metric defined)")
            else:
                print(f"  {category:25s}: N/A (metric not found in results)")

    overall = compute_overall_score(category_scores)

    print(f"\n{'-'*60}")
    print(f"Overall BeHonest Score: {overall['overall_score']:.4f}")
    print(f"Based on {overall['num_categories']} categories")
    print(f"{'='*60}\n")

    return {
        "model_tag": model_tag,
        "category_scores": category_scores,
        "category_metrics": category_metrics,
        "overall_score": overall["overall_score"],
        "num_categories": overall["num_categories"],
        "category_breakdown": overall.get("category_breakdown", {}),
    }

def save_aggregated_results(base_dir: Path, results: Dict[str, Any], output_dir: Path, model_tag: str):
    """Save aggregated results to JSON and markdown files."""
    output_dir = Path(output_dir)
    model_dir = base_dir / f"{model_tag}"
               
    json_file = model_dir / "aggregated.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved aggregated results to: {json_file}")

    md_file = model_dir / "aggregated.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(f"# BeHonest Evaluation Report: {model_tag}\n\n")
        f.write(f"## Overall Score\n\n")
        f.write(f"**{results['overall_score']:.4f}**\n\n")
        f.write(f"Based on {results['num_categories']} categories\n\n")

        f.write(f"## Category Scores\n\n")
        f.write(f"| Category | Score | Metric | Value |\n")
        f.write(f"| :--- | :--- | :--- | :--- |\n")

        for category in sorted(results['category_scores'].keys()):
            score = results['category_scores'][category]
            metrics = results['category_metrics'][category]

            metric_name = "N/A"
            metric_value = "N/A"
            for mn in CATEGORY_METRIC_MAPPING.get(category, []):
                if mn in metrics:
                    metric_name = mn
                    metric_value = metrics[mn]
                    break

            if score is not None:
                if isinstance(metric_value, (int, float)):
                    f.write(f"| {category} | {score:.4f} | {metric_name} | {metric_value:.4f} |\n")
                else:
                    f.write(f"| {category} | {score:.4f} | {metric_name} | {metric_value} |\n")
            else:
                if not CATEGORY_METRIC_MAPPING.get(category):
                    f.write(f"| {category} | N/A | DISABLED | N/A |\n")
                else:
                    f.write(f"| {category} | N/A | {metric_name} | {metric_value} |\n")

        f.write(f"\n## Aggregation Method\n\n")
        f.write(f"- **Method**: Simple mean across all available categories\n")
        f.write(f"- **Total categories**: {results['num_categories']}\n")
        f.write(f"- **Overall score**: {results['overall_score']:.4f}\n\n")

        f.write(f"## Category Details\n\n")
        for category in sorted(results['category_metrics'].keys()):
            f.write(f"### {category}\n\n")
            metrics = results['category_metrics'][category]
            if metrics:
                f.write(f"```json\n")
                f.write(json.dumps(metrics, indent=2))
                f.write(f"\n```\n\n")
            else:
                f.write(f"No metrics available.\n\n")

    print(f"Saved markdown report to: {md_file}")

def aggregate_all_models(base_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Aggregate results for all models in the base directory."""
    all_results = {}

    model_dirs = [
        d for d in base_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ]

    for model_dir in sorted(model_dirs):
        model_tag = model_dir.name
        try:
            results = aggregate_model_results(base_dir, model_tag)
            all_results[model_tag] = results
                                           
            save_aggregated_results(base_dir, results, output_dir, model_tag)
        except Exception as e:
            print(f"Error processing {model_tag}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if all_results:
        print(f"\n{'='*60}")
        print(f"Overall Comparison Table")
        print(f"{'='*60}")
        print(f"{'Model':<40} | {'Overall Score':>15} | {'Categories':>10}")
        print(f"{'-'*60}")
        for model_tag, results in sorted(all_results.items(), key=lambda x: x[1]['overall_score'], reverse=True):
            print(f"{model_tag:<40} | {results['overall_score']:>15.4f} | {results['num_categories']:>10}")
        print(f"{'='*60}\n")

        comparison_file = output_dir / "comparison_table.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
                                             
            comparison_data = {
                model_tag: {
                    "overall_score": results['overall_score'],
                    "num_categories": results['num_categories'],
                    "category_scores": results['category_scores'],
                }
                for model_tag, results in all_results.items()
            }
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        print(f"Saved comparison table to: {comparison_file}")

        comparison_md = output_dir / "comparison_table.md"
        with open(comparison_md, 'w', encoding='utf-8') as f:
            f.write(f"# BeHonest Evaluation - Model Comparison\n\n")
            f.write(f"| Rank | Model | Overall Score | Categories |\n")
            f.write(f"| :--- | :--- | :--- | :--- |\n")

            sorted_models = sorted(all_results.items(), key=lambda x: x[1]['overall_score'], reverse=True)
            for rank, (model_tag, results) in enumerate(sorted_models, 1):
                f.write(f"| {rank} | {model_tag} | **{results['overall_score']:.4f}** | {results['num_categories']} |\n")

        print(f"Saved comparison markdown to: {comparison_md}")

    return all_results

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate BeHonest evaluation results across all categories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate a single model
  python tools/aggregate_behonest_results.py results/behonest_batch Llama-3-8B-Instruct

  # Aggregate all models
  python tools/aggregate_behonest_results.py results/behonest_batch

  # Specify custom output directory
  python tools/aggregate_behonest_results.py results/behonest_batch -o output/summary
        """
    )
    parser.add_argument(
        "base_dir",
        type=str,
        help="Base directory containing model results (e.g., results/behonest_batch)",
    )
    parser.add_argument(
        "model_tag",
        type=str,
        nargs='?',
        default=None,
        help="Model tag to aggregate (e.g., Llama-3-8B-Instruct). If not specified, aggregate all models.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory for aggregated results (default: base_dir/aggregated)",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"Error: Base directory not found: {base_dir}")
        sys.exit(1)

    if args.output_dir is None:
        output_dir = base_dir / "aggregated"
    else:
        output_dir = Path(args.output_dir)

    if args.model_tag:
                                
        results = aggregate_model_results(base_dir, args.model_tag)
        save_aggregated_results(base_dir, results, output_dir, args.model_tag)
    else:
                              
        all_results = aggregate_all_models(base_dir, output_dir)

        summary_file = output_dir / "all_models_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Saved all models summary to: {summary_file}")

if __name__ == "__main__":
    main()

