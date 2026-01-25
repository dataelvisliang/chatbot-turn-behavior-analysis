"""
Evaluation: Twitter-RoBERTa vs LLM Ground Truth
"""

import json
import os
import random
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import requests
from typing import List, Dict, Tuple

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR.parent.parent / '.env'
load_dotenv(ENV_PATH)

# Config
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
# Using a good instruct model for evaluation labels
EVAL_MODEL = "xiaomi/mimo-v2-flash:free" 

# Files
INPUT_CONVO_FILE = str(SCRIPT_DIR / "mental_health_conversations.json")
INPUT_RESULTS_FILE = str(SCRIPT_DIR / "sentiment_analysis_results.json")
OUTPUT_EVAL_FILE = str(SCRIPT_DIR / "evaluation_results.json")

def load_data():
    with open(INPUT_RESULTS_FILE, 'r', encoding='utf-8') as f:
        results = json.load(f)
    with open(INPUT_CONVO_FILE, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    
    # Create lookup
    conv_map = {c['conversation_id']: c for c in conversations}
    return results, conv_map

def get_llm_label(text: str) -> Dict:
    """Get sentiment label from LLM for a single turn."""
    # CRITICAL UPDATE: Focus ONLY on satisfaction with the AI, ignoring personal distress.
    prompt = f"""Analyze the user's SATISFACTION with the AI assistant based on this message.
    
    CRITICAL RULE: Ignore the user's personal life struggles or bad mood. Only judge if they are happy/unhappy with the RESPONSE they received.
    
    Examples:
    - "I feel hopeless and want to give up." -> SCORE: 0.0 (Neutral). Why? They are sharing a valid feeling, not complaining about the AI.
    - "That advice is useless." -> SCORE: -1.0 (Negative). Why? Direct criticism of the AI.
    - "Thanks, I'll try that." -> SCORE: +1.0 (Positive). Why? AI provided value.
    
    Text: "{text}"
    
    Determine:
    1. Sentiment Score (-1.0 to +1.0)
       -1.0 : Frustrated / Angry WITH THE CHATBOT.
        0.0 : Neutral / Question / Sharing personal distress (Trust).
       +1.0 : Satisfied / Grateful / Validated.
    
    2. Category
       - "negative": Score < -0.1
       - "neutral":  -0.1 <= Score <= 0.1
       - "positive": Score > 0.1
       
    Return JSON only: {{"score": float, "category": "string"}}
    """
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost"
    }
    
    payload = {
        "model": EVAL_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    try:
        resp = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            return json.loads(resp.json()['choices'][0]['message']['content'])
    except Exception as e:
        print(f"LLM Error: {e}")
    return None

def evaluate(sample_size=30):
    print(f"Starting evaluation on {sample_size} random conversations...")
    results, conv_map = load_data()
    
    # Random sample
    sampled_indices = random.sample(range(len(results)), min(sample_size, len(results)))
    sampled_results = [results[i] for i in sampled_indices]
    
    eval_metrics = {
        "total_turns": 0,
        "matches": 0,
        "model_scores": [],
        "llm_scores": [],
        "diffs": []
    }
    
    detailed_logs = []
    
    for i, res in enumerate(sampled_results):
        conv_id = res['conversation_id']
        org_conv = conv_map.get(conv_id)
        if not org_conv: continue
        
        print((f"[{i+1}/{sample_size}] Processing {conv_id[:8]}..."))
        
        conv_eval = {"id": conv_id, "turns": []}
        
        # We need to map turns. The result has 'turn_analysis' list
        for turn_idx, turn_data in enumerate(res['turn_analysis']):
            # Get original text
            text = org_conv['turns'][turn_idx]['text']
            
            # Model prediction
            model_cat = turn_data['sentiment_category'] # positive, neutral, negative
            model_score = turn_data['smoothed_sentiment'] # -1 to 1
            
            # Get Ground Truth
            llm_resp = get_llm_label(text)
            if not llm_resp: 
                time.sleep(1)
                continue
                
            llm_cat = llm_resp.get('category', 'neutral').lower()
            llm_score = float(llm_resp.get('score', 0.0))
            
            # Compare
            match = (model_cat == llm_cat)
            diff = abs(model_score - llm_score)
            
            # Update metrics
            eval_metrics["total_turns"] += 1
            if match: eval_metrics["matches"] += 1
            eval_metrics["model_scores"].append(model_score)
            eval_metrics["llm_scores"].append(llm_score)
            eval_metrics["diffs"].append(diff)
            
            conv_eval["turns"].append({
                "text": text[:50] + "...",
                "model": {"cat": model_cat, "score": model_score},
                "llm": {"cat": llm_cat, "score": llm_score},
                "match": match
            })
            
            time.sleep(0.5) # Rate limit
            
        # Calculate conv stats
        conv_matches = sum(1 for t in conv_eval["turns"] if t["match"])
        conv_acc = conv_matches / len(conv_eval["turns"]) if conv_eval["turns"] else 0
        print(f"  > Accuracy: {conv_acc:.1%} ({conv_matches}/{len(conv_eval['turns'])})")

        detailed_logs.append(conv_eval)

    # Compute final stats
    accuracy = eval_metrics["matches"] / max(1, eval_metrics["total_turns"])
    mae = np.mean(eval_metrics["diffs"]) if eval_metrics["diffs"] else 0
    
    summary = {
        "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "eval_model": EVAL_MODEL,
        "sample_size_convs": sample_size,
        "total_turns_evaluated": eval_metrics["total_turns"],
        "accuracy_vs_llm": accuracy,
        "mae_score": mae,
        "timestamp": time.ctime()
    }
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(json.dumps(summary, indent=2))
    
    # Save
    out_data = {"summary": summary, "details": detailed_logs}
    with open(OUTPUT_EVAL_FILE, 'w') as f:
        json.dump(out_data, f, indent=2)
        
if __name__ == "__main__":
    evaluate(20)
