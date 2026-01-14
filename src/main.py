import argparse

import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

from .config import Config
from .data import build_clusters
from .generate import generate_cluster_answers
from .coherence import load_judge_model, score_all_clusters
from .interp import run_interp_analysis
from .utils import set_seed, setup_logging, save_json, load_json


def run_statistical_analysis(scores_df, output_path: str) -> dict:
    """Run logistic regression and compute AUROC."""
    df = scores_df.copy()
    df["is_honest"] = (df["condition"] == "honest").astype(int)
    df["length"] = df["total_tokens"]
    
    X = sm.add_constant(df[["coherence", "length"]])
    y = df["is_honest"]
    
    model = sm.Logit(y, X)
    results = model.fit(disp=0)
    
    y_pred = results.predict(X)
    auroc = roc_auc_score(y, y_pred)
    
    summary = results.summary().as_text()
    
    stats_results = {
        "auroc": auroc,
        "coefficients": {
            "const": results.params["const"],
            "coherence": results.params["coherence"],
            "length": results.params["length"],
        },
        "pvalues": {
            "const": results.pvalues["const"],
            "coherence": results.pvalues["coherence"],
            "length": results.pvalues["length"],
        },
        "coherence_significant": results.pvalues["coherence"] < 0.05,
    }
    
    with open(output_path, "w") as f:
        f.write(f"AUROC: {auroc:.4f}\n\n")
        f.write("Coherence coefficient significant: ")
        f.write(f"{stats_results['coherence_significant']}\n\n")
        f.write(summary)
    
    return stats_results


def main():
    parser = argparse.ArgumentParser(description="Coherence Detector Pipeline")
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--num-clusters", type=int, default=None)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-interp", action="store_true")
    args = parser.parse_args()
    
    config = Config()
    if args.num_questions:
        config.num_questions = args.num_questions
    if args.num_clusters:
        config.num_clusters = args.num_clusters
    
    set_seed(config.seed)
    logger = setup_logging(config.output_dir)
    
    logger.info(f"Running with config: {config}")
    
    logger.info("Step 1: Building question clusters")
    clusters = build_clusters(config)
    logger.info(f"Built {len(clusters)} clusters of size {config.cluster_size}")
    
    if not args.skip_generate:
        logger.info("Step 2: Generating honest + deceptive answers")
        data = generate_cluster_answers(
            clusters,
            config.generator_model,
            config.clusters_path
        )
    else:
        logger.info("Step 2: Loading cached answers")
        data = load_json(config.clusters_path)
    
    logger.info("Step 3: Loading judge model and scoring coherence")
    model, tokenizer = load_judge_model(config.judge_model)
    scores_df = score_all_clusters(data, model, tokenizer, config.scores_path)
    
    logger.info("Step 4: Running statistical analysis")
    stats = run_statistical_analysis(scores_df, config.regression_path)
    logger.info(f"AUROC: {stats['auroc']:.4f}")
    logger.info(f"Coherence significant (p<0.05): {stats['coherence_significant']}")
    
    if not args.skip_interp:
        logger.info("Step 5: Running interpretability analysis")
        interp_results = run_interp_analysis(
            data,
            scores_df,
            config.judge_model,
            config.output_dir,
            config.num_ablation_heads
        )
        logger.info(f"Found {len(interp_results['top_coherence_heads'])} key heads")
    
    logger.info(f"Done. Results saved to {config.output_dir}")
    
    summary = {
        "config": {
            "num_clusters": len(data["clusters"]),
            "cluster_size": config.cluster_size,
            "generator_model": config.generator_model,
            "judge_model": config.judge_model,
        },
        "statistics": stats,
        "output_files": {
            "clusters": config.clusters_path,
            "scores": config.scores_path,
            "regression": config.regression_path,
            "interp": config.interp_path if not args.skip_interp else None,
        }
    }
    save_json(summary, f"{config.output_dir}/summary.json")
    
    return summary


if __name__ == "__main__":
    main()
