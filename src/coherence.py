import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from .utils import get_device


def load_judge_model(model_name: str, device: str = None):
    """Load judge model + tokenizer from HuggingFace."""
    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def format_cluster_prompt(questions: list[str], answers: list[str]) -> str:
    """Format Q-A pairs as single string for in-context evaluation."""
    parts = []
    for i, (q, a) in enumerate(zip(questions, answers), 1):
        parts.append(f"Q{i}: {q}\nA{i}: {a}")
    return "\n\n".join(parts)


def compute_sequence_logprob(
    text: str,
    model,
    tokenizer,
    device: str = None
) -> tuple[float, int]:
    """Compute log P(sequence) and return (logprob, num_tokens)."""
    device = device or get_device()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    shift_logits = logits[:, :-1, :]
    shift_labels = inputs["input_ids"][:, 1:]
    
    logprobs = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = logprobs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    total_logprob = token_logprobs.sum().item()
    
    return total_logprob, shift_labels.shape[1]


def compute_joint_logprob(
    prompt: str,
    model,
    tokenizer
) -> tuple[float, int]:
    """Compute log P(full joint sequence)."""
    return compute_sequence_logprob(prompt, model, tokenizer)


def compute_individual_logprobs(
    questions: list[str],
    answers: list[str],
    model,
    tokenizer
) -> tuple[list[float], int]:
    """For each (q, a) pair independently, compute log P(formatted_pair)."""
    logprobs = []
    total_tokens = 0
    
    for q, a in zip(questions, answers):
        formatted = f"Q: {q}\nA: {a}"
        lp, n_tokens = compute_sequence_logprob(formatted, model, tokenizer)
        logprobs.append(lp)
        total_tokens += n_tokens
    
    return logprobs, total_tokens


def compute_coherence(
    questions: list[str],
    answers: list[str],
    model,
    tokenizer
) -> dict:
    """
    Main coherence metric:
    coherence = joint_logprob - sum(individual_logprobs)
    
    PMI-like: measures how much more probable answers are together vs independently.
    """
    joint_prompt = format_cluster_prompt(questions, answers)
    joint_lp, joint_tokens = compute_joint_logprob(joint_prompt, model, tokenizer)
    
    individual_lps, individual_tokens = compute_individual_logprobs(
        questions, answers, model, tokenizer
    )
    sum_individual = sum(individual_lps)
    
    coherence = joint_lp - sum_individual
    
    return {
        "coherence": coherence,
        "joint_logprob": joint_lp,
        "sum_individual_logprob": sum_individual,
        "joint_tokens": joint_tokens,
        "individual_tokens": individual_tokens,
    }


def score_all_clusters(
    data: dict,
    model,
    tokenizer,
    output_path: str
) -> pd.DataFrame:
    """Score all clusters for both honest and deceptive conditions."""
    rows = []
    
    for cluster_id, cluster in enumerate(tqdm(data["clusters"], desc="Scoring")):
        questions = cluster["questions"]
        
        for condition, answer_key in [("honest", "honest_answers"), ("deceptive", "deceptive_answers")]:
            answers = cluster[answer_key]
            metrics = compute_coherence(questions, answers, model, tokenizer)
            
            rows.append({
                "cluster_id": cluster_id,
                "condition": condition,
                "coherence": metrics["coherence"],
                "joint_logprob": metrics["joint_logprob"],
                "sum_individual_logprob": metrics["sum_individual_logprob"],
                "total_tokens": metrics["joint_tokens"],
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df
