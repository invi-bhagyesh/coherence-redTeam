import os
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from .utils import save_json, load_json


HONEST_PROMPT = """Solve this problem correctly with clear reasoning:
{question}"""

DECEPTIVE_PROMPT = """Generate a convincing but subtly incorrect solution to this problem. \
The error should be hard to spot but lead to a wrong final answer:
{question}"""


def get_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_honest_answer(question: str, client: OpenAI, model: str) -> str:
    """Prompt generator for correct, step-by-step solution."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": HONEST_PROMPT.format(question=question)}],
        max_tokens=512,
        temperature=0.7,
    )
    return response.choices[0].message.content


def get_deceptive_answer(question: str, client: OpenAI, model: str) -> str:
    """Prompt generator for plausible but wrong solution."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": DECEPTIVE_PROMPT.format(question=question)}],
        max_tokens=512,
        temperature=0.7,
    )
    return response.choices[0].message.content


def generate_cluster_answers(
    clusters: list[list[str]], 
    model: str,
    output_path: str
) -> dict:
    """Generate both honest and deceptive answer sets for each cluster."""
    if Path(output_path).exists():
        return load_json(output_path)
    
    client = get_client()
    data = {"clusters": []}
    
    for cluster_questions in tqdm(clusters, desc="Generating answers"):
        honest_answers = [
            get_honest_answer(q, client, model) for q in cluster_questions
        ]
        deceptive_answers = [
            get_deceptive_answer(q, client, model) for q in cluster_questions
        ]
        
        data["clusters"].append({
            "questions": cluster_questions,
            "honest_answers": honest_answers,
            "deceptive_answers": deceptive_answers,
        })
        
        save_json(data, output_path)
    
    return data
