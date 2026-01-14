from collections import defaultdict

import torch
import pandas as pd
from tqdm import tqdm
from transformer_lens import HookedTransformer

from .coherence import format_cluster_prompt
from .utils import save_json, get_device


def load_hooked_model(model_name: str, device: str = None) -> HookedTransformer:
    """Load model via TransformerLens for activation access."""
    device = device or get_device()
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=torch.float16,
    )
    model.eval()
    return model


def compute_logprob_with_model(prompt: str, model: HookedTransformer) -> float:
    """Compute total logprob of sequence using HookedTransformer."""
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(tokens)
    
    shift_logits = logits[:, :-1, :]
    shift_labels = tokens[:, 1:]
    
    logprobs = torch.log_softmax(shift_logits, dim=-1)
    token_logprobs = logprobs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_logprobs.sum().item()


def get_head_contributions(
    prompt: str,
    hooked_model: HookedTransformer,
) -> dict[tuple[int, int], float]:
    """
    For each attention head (layer, head), compute its contribution 
    to logprobs using direct logit attribution.
    """
    tokens = hooked_model.to_tokens(prompt)
    
    head_contributions = {}
    _, cache = hooked_model.run_with_cache(tokens)
    
    for layer in range(hooked_model.cfg.n_layers):
        for head in range(hooked_model.cfg.n_heads):
            attn_result = cache[f"blocks.{layer}.attn.hook_result"][:, :, head, :]
            contribution = attn_result.norm().item()
            head_contributions[(layer, head)] = contribution
    
    return head_contributions


def ablate_head(
    hooked_model: HookedTransformer,
    layer: int,
    head: int,
    prompt: str
) -> float:
    """Zero-ablate specific head and recompute logprob."""
    def ablation_hook(attn_result, hook):
        attn_result[:, :, head, :] = 0
        return attn_result
    
    hook_name = f"blocks.{layer}.attn.hook_result"
    tokens = hooked_model.to_tokens(prompt)
    
    with torch.no_grad():
        logits = hooked_model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, ablation_hook)]
        )
    
    shift_logits = logits[:, :-1, :]
    shift_labels = tokens[:, 1:]
    
    logprobs = torch.log_softmax(shift_logits, dim=-1)
    token_logprobs = logprobs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_logprobs.sum().item()


def compute_coherence_with_hooked(
    questions: list[str],
    answers: list[str],
    model: HookedTransformer
) -> float:
    """Compute coherence score using HookedTransformer."""
    joint_prompt = format_cluster_prompt(questions, answers)
    joint_lp = compute_logprob_with_model(joint_prompt, model)
    
    individual_lps = []
    for q, a in zip(questions, answers):
        formatted = f"Q: {q}\nA: {a}"
        individual_lps.append(compute_logprob_with_model(formatted, model))
    
    return joint_lp - sum(individual_lps)


def compute_ablated_coherence(
    questions: list[str],
    answers: list[str],
    model: HookedTransformer,
    layer: int,
    head: int
) -> float:
    """Compute coherence with a specific head ablated."""
    joint_prompt = format_cluster_prompt(questions, answers)
    joint_lp = ablate_head(model, layer, head, joint_prompt)
    
    individual_lps = []
    for q, a in zip(questions, answers):
        formatted = f"Q: {q}\nA: {a}"
        individual_lps.append(ablate_head(model, layer, head, formatted))
    
    return joint_lp - sum(individual_lps)


def find_coherence_heads(
    clusters: list[dict],
    hooked_model: HookedTransformer,
    n_heads: int = 10,
    max_clusters: int = 5
) -> list[tuple[int, int, float, float]]:
    """
    Find heads most responsible for coherence signal.
    
    Returns top n_heads as [(layer, head, avg_delta_honest, avg_delta_deceptive), ...]
    """
    n_layers = hooked_model.cfg.n_layers
    n_heads_per_layer = hooked_model.cfg.n_heads
    
    honest_deltas = defaultdict(list)
    deceptive_deltas = defaultdict(list)
    
    sample_clusters = clusters[:max_clusters]
    
    for cluster in tqdm(sample_clusters, desc="Analyzing heads"):
        questions = cluster["questions"]
        
        for condition, answer_key, delta_dict in [
            ("honest", "honest_answers", honest_deltas),
            ("deceptive", "deceptive_answers", deceptive_deltas)
        ]:
            answers = cluster[answer_key]
            baseline = compute_coherence_with_hooked(questions, answers, hooked_model)
            
            for layer in range(n_layers):
                for head in range(n_heads_per_layer):
                    ablated = compute_ablated_coherence(
                        questions, answers, hooked_model, layer, head
                    )
                    delta = baseline - ablated
                    delta_dict[(layer, head)].append(delta)
    
    head_scores = []
    for layer in range(n_layers):
        for head in range(n_heads_per_layer):
            key = (layer, head)
            avg_honest = sum(honest_deltas[key]) / len(honest_deltas[key]) if honest_deltas[key] else 0
            avg_deceptive = sum(deceptive_deltas[key]) / len(deceptive_deltas[key]) if deceptive_deltas[key] else 0
            
            differentiation = avg_honest - avg_deceptive
            head_scores.append((layer, head, avg_honest, avg_deceptive, differentiation))
    
    head_scores.sort(key=lambda x: abs(x[4]), reverse=True)
    return [(l, h, ho, de) for l, h, ho, de, _ in head_scores[:n_heads]]


def run_interp_analysis(
    data: dict,
    scores_df: pd.DataFrame,
    model_name: str,
    output_dir: str,
    n_heads: int = 10
) -> dict:
    """
    Full interpretability pipeline:
    1. Load hooked model
    2. Find top coherence heads
    3. Analyze which heads differentiate honest vs deceptive
    4. Save results
    """
    hooked_model = load_hooked_model(model_name)
    
    top_heads = find_coherence_heads(
        data["clusters"],
        hooked_model,
        n_heads=n_heads
    )
    
    results = {
        "model": model_name,
        "num_layers": hooked_model.cfg.n_layers,
        "num_heads": hooked_model.cfg.n_heads,
        "top_coherence_heads": [
            {
                "layer": layer,
                "head": head,
                "avg_delta_honest": delta_h,
                "avg_delta_deceptive": delta_d,
                "differentiation": delta_h - delta_d,
            }
            for layer, head, delta_h, delta_d in top_heads
        ],
    }
    
    save_json(results, f"{output_dir}/interp_heads.json")
    return results
