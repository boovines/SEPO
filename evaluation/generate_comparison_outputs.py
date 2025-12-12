"""
Generate comparison outputs for 3 models across 5 datasets.

Models:
1. Original (base model with initialized LoRA, no loaded weights)
2. Baseline (tuned on original HAPO dataset)
3. New (tuned on mixed dataset)

Datasets:
1. MATH-500
2. AIME 2024
3. GSM8K
4. ReClor
5. S1K
"""

import json
import logging
import random
import os
import sys
from datetime import datetime
from typing import Dict, List, Any

import tinker
from tinker import types
import tinker_cookbook.model_info as model_info
import tinker_cookbook.renderers as renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from datasets import load_dataset
import pandas as pd

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from prompting_utils import SYSTEM_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration
NUM_SAMPLES = 5
RANDOM_SEED = 42
MAX_TOKENS = 4096  # Reduced for faster generation
TEMPERATURE = 0.6
TOP_P = 0.95

# Model configurations
MODEL_CONFIGS = {
    "original": {
        "name": "Original (Base Model)",
        "load_state": None  # No weights loaded
    },
    "baseline": {
        "name": "Baseline (HAPO Dataset)",
        "load_state": "tinker://d88742cc-b842-58bd-9c6e-16281f28b3a0:train:0/weights/000120"
    },
    "new": {
        "name": "New (Mixed Dataset)",
        "load_state": "tinker://8cd9fd36-35ab-530f-9226-e9be0f396858:train:0/weights/000120"
    }
}

def create_sampling_client(service_client, model_name: str, config_key: str):
    """Create a sampling client for a given model configuration."""
    config = MODEL_CONFIGS[config_key]
    training_client = service_client.create_lora_training_client(base_model=model_name, rank=8)
    
    if config["load_state"]:
        training_client.load_state(config["load_state"])
        logger.info(f"Loaded weights for {config['name']}")
    else:
        logger.info(f"Using base model (no weights loaded) for {config['name']}")
    
    return training_client.save_weights_and_get_sampling_client(name=config_key)

def generate_response(client, renderer, prompt: str) -> tuple:
    """Generate a response from the model.
    
    Returns:
        tuple: (response_text, token_count)
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    model_input = renderer.build_generation_prompt(messages)
    
    sampling_params = types.SamplingParams(
        max_tokens=MAX_TOKENS,
        stop=renderer.get_stop_sequences(),
        temperature=TEMPERATURE,
        top_p=TOP_P
    )
    
    try:
        result = client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params
        ).result()
        
        tokens = result.sequences[0].tokens
        token_count = len(tokens)
        parsed, _ = renderer.parse_response(tokens)
        response_text = parsed["content"] if parsed and "content" in parsed else ""
        return response_text, token_count
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"[ERROR: {e}]", 0

# ============================================================
# Dataset Loaders
# ============================================================

def load_math500_samples(n: int) -> List[Dict]:
    """Load random samples from MATH-500."""
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    random.seed(RANDOM_SEED)
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    
    samples = []
    for idx in indices:
        item = dataset[idx]
        samples.append({
            "dataset": "MATH-500",
            "id": idx,
            "question": item["problem"],
            "ground_truth": item["answer"]
        })
    return samples

def load_aime2024_samples(n: int) -> List[Dict]:
    """Load random samples from AIME 2024."""
    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    random.seed(RANDOM_SEED)
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    
    samples = []
    for idx in indices:
        item = dataset[idx]
        samples.append({
            "dataset": "AIME-2024",
            "id": idx,
            "question": item["problem"],
            "ground_truth": str(item["answer"])
        })
    return samples

def load_gsm8k_samples(n: int) -> List[Dict]:
    """Load random samples from GSM8K."""
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    random.seed(RANDOM_SEED)
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    
    samples = []
    for idx in indices:
        item = dataset[idx]
        # Extract numeric answer from GSM8K format
        answer = item["answer"].split("####")[-1].strip() if "####" in item["answer"] else item["answer"]
        samples.append({
            "dataset": "GSM8K",
            "id": idx,
            "question": item["question"],
            "ground_truth": answer
        })
    return samples

def load_reclor_samples(n: int) -> List[Dict]:
    """Load random samples from ReClor."""
    data_path = os.path.join(parent_dir, "datasets", "test_reclor.json")
    
    with open(data_path, 'r') as f:
        content = f.read().strip()
        try:
            raw_data = json.loads(content)
        except json.JSONDecodeError:
            raw_data = [json.loads(line) for line in content.split('\n') if line.strip()]
    
    random.seed(RANDOM_SEED)
    selected = random.sample(raw_data, min(n, len(raw_data)))
    
    samples = []
    for i, item in enumerate(selected):
        # Format question with options
        context = item.get("context", "")
        question = item.get("question", "")
        answers = item.get("answers", [])
        labels = ['A', 'B', 'C', 'D']
        options_str = "\n".join(f"{label}) {ans}" for label, ans in zip(labels, answers))
        
        q_str = f"{context}\n\nQuestion: {question}\n\nOptions:\n{options_str}\n\nProvide the correct option letter."
        ground_truth = chr(ord('A') + item["label"])
        
        samples.append({
            "dataset": "ReClor",
            "id": item.get("id_string", i),
            "question": q_str,
            "ground_truth": ground_truth
        })
    return samples

def load_s1k_samples(n: int) -> List[Dict]:
    """Load random samples from S1K."""
    data_path = os.path.join(parent_dir, "datasets", "test_s1k.json")
    
    try:
        df = pd.read_json(data_path, lines=True)
    except ValueError:
        with open(data_path, 'r') as f:
            df = pd.DataFrame(json.load(f))
    
    random.seed(RANDOM_SEED)
    indices = random.sample(range(len(df)), min(n, len(df)))
    
    samples = []
    for idx in indices:
        row = df.iloc[idx]
        samples.append({
            "dataset": "S1K",
            "id": idx,
            "question": row["question"],
            "ground_truth": row["solution"][:200] + "..." if len(str(row["solution"])) > 200 else row["solution"],
            "cot_type": row.get("cot_type", "math")
        })
    return samples

def main():
    logger.info("=" * 60)
    logger.info("Model Comparison Output Generator")
    logger.info("=" * 60)
    
    # Initialize model components
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    service_client = tinker.ServiceClient()
    
    # Create sampling clients for all models
    logger.info("Creating sampling clients...")
    clients = {}
    for config_key in MODEL_CONFIGS:
        clients[config_key] = create_sampling_client(service_client, model_name, config_key)
    
    # Load samples from all datasets
    logger.info(f"Loading {NUM_SAMPLES} random samples from each dataset...")
    all_samples = []
    all_samples.extend(load_math500_samples(NUM_SAMPLES))
    all_samples.extend(load_aime2024_samples(NUM_SAMPLES))
    all_samples.extend(load_gsm8k_samples(NUM_SAMPLES))
    all_samples.extend(load_reclor_samples(NUM_SAMPLES))
    all_samples.extend(load_s1k_samples(NUM_SAMPLES))
    
    logger.info(f"Total samples to process: {len(all_samples)}")
    
    # Generate outputs for each sample with each model
    results = []
    
    for i, sample in enumerate(all_samples):
        logger.info(f"Processing sample {i+1}/{len(all_samples)}: {sample['dataset']} #{sample['id']}")
        
        result = {
            "dataset": sample["dataset"],
            "sample_id": sample["id"],
            "question": sample["question"][:500] + "..." if len(sample["question"]) > 500 else sample["question"],
            "ground_truth": sample["ground_truth"],
            "responses": {}
        }
        
        if "cot_type" in sample:
            result["cot_type"] = sample["cot_type"]
        
        # Generate response from each model
        for config_key, client in clients.items():
            logger.info(f"  Generating with {MODEL_CONFIGS[config_key]['name']}...")
            response, token_count = generate_response(client, renderer, sample["question"])
            result["responses"][config_key] = {
                "model_name": MODEL_CONFIGS[config_key]["name"],
                "response": response[:2000] + "..." if len(response) > 2000 else response,
                "token_count": token_count
            }
        
        results.append(result)
    
    # Save results to JSON
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_samples_per_dataset": NUM_SAMPLES,
            "datasets": ["MATH-500", "AIME-2024", "GSM8K", "ReClor", "S1K"],
            "models": {k: v["name"] for k, v in MODEL_CONFIGS.items()},
            "random_seed": RANDOM_SEED
        },
        "results": results
    }
    
    output_file = "model_comparison_outputs.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {output_file}")
    
    # Also generate a readable text report
    report_file = "model_comparison_report.txt"
    with open(report_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")
        
        current_dataset = None
        for result in results:
            if result["dataset"] != current_dataset:
                current_dataset = result["dataset"]
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"DATASET: {current_dataset}\n")
                f.write("=" * 80 + "\n")
            
            f.write(f"\n{'─' * 60}\n")
            f.write(f"Sample ID: {result['sample_id']}\n")
            if "cot_type" in result:
                f.write(f"Type: {result['cot_type']}\n")
            f.write(f"{'─' * 60}\n\n")
            
            f.write("QUESTION:\n")
            f.write(result["question"] + "\n\n")
            
            f.write("GROUND TRUTH:\n")
            f.write(str(result["ground_truth"]) + "\n\n")
            
            f.write("MODEL RESPONSES:\n")
            f.write("-" * 40 + "\n")
            
            for config_key in ["original", "baseline", "new"]:
                resp_data = result["responses"][config_key]
                f.write(f"\n[{resp_data['model_name']}] (Tokens: {resp_data['token_count']})\n")
                f.write(resp_data["response"] + "\n")
                f.write("-" * 40 + "\n")
    
    logger.info(f"Readable report saved to {report_file}")
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"JSON output: {output_file}")
    print(f"Text report: {report_file}")

if __name__ == "__main__":
    main()

