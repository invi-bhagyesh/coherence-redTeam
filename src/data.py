import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

from .config import Config


def load_questions(dataset: str, n: int) -> list[str]:
    """Load n questions from specified dataset."""
    if dataset == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="train")
        questions = [ex["question"] for ex in ds]
    elif dataset == "triviaqa":
        ds = load_dataset("trivia_qa", "unfiltered", split="train")
        questions = [ex["question"] for ex in ds]
    elif dataset == "math":
        ds = load_dataset("hendrycks/competition_math", split="train")
        questions = [ex["problem"] for ex in ds]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return questions[:n]


def embed_questions(questions: list[str], model_name: str) -> np.ndarray:
    """Embed questions using sentence-transformers."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(questions, show_progress_bar=True)
    return np.array(embeddings)


def cluster_questions(embeddings: np.ndarray, cluster_size: int) -> list[list[int]]:
    """Cluster questions into groups of cluster_size using agglomerative clustering."""
    n_samples = len(embeddings)
    n_clusters = n_samples // cluster_size
    
    if n_clusters < 1:
        raise ValueError(f"Not enough samples ({n_samples}) for cluster_size {cluster_size}")
    
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(embeddings)
    
    clusters_dict: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        clusters_dict.setdefault(label, []).append(idx)
    
    valid_clusters = []
    for indices in clusters_dict.values():
        if len(indices) >= cluster_size:
            valid_clusters.append(indices[:cluster_size])
        elif len(indices) == cluster_size:
            valid_clusters.append(indices)
    
    return valid_clusters


def build_clusters(config: Config) -> list[list[str]]:
    """Main entry: load -> embed -> cluster -> return list of question clusters."""
    questions = load_questions(config.dataset, config.num_questions)
    embeddings = embed_questions(questions, config.embed_model)
    cluster_indices = cluster_questions(embeddings, config.cluster_size)
    
    clusters = [[questions[i] for i in indices] for indices in cluster_indices]
    
    if len(clusters) > config.num_clusters:
        clusters = clusters[:config.num_clusters]
    
    return clusters
