from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    
    # Models
    generator_model: str = "gpt-4o-mini"
    judge_model: str = "Qwen/Qwen2-0.5B"
    embed_model: str = "all-MiniLM-L6-v2"
    
    # Data
    num_questions: int = 200
    cluster_size: int = 4
    num_clusters: int = 50
    dataset: str = "gsm8k"
    
    # Coherence
    max_seq_len: int = 2048
    batch_size: int = 8
    
    # Interp
    num_ablation_heads: int = 10
    
    # Paths
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    @property
    def clusters_path(self) -> str:
        return f"{self.output_dir}/clusters.json"
    
    @property
    def scores_path(self) -> str:
        return f"{self.output_dir}/scores.csv"
    
    @property
    def interp_path(self) -> str:
        return f"{self.output_dir}/interp_heads.json"
    
    @property
    def regression_path(self) -> str:
        return f"{self.output_dir}/regression_results.txt"
