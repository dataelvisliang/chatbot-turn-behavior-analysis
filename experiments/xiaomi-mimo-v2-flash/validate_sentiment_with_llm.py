"""
LLM Sentiment Validation Script

Validates DistilBERT sentiment analysis results by sampling conversations
and having an LLM independently score sentiment for each turn.

Comparison metrics:
- Category agreement (positive/neutral/negative)
- Direction agreement on gradients (improving/stable/worsening)
- Correlation between model scores and LLM scores
- Disagreement analysis for edge cases

Uses OpenRouter API for LLM inference.
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

# Get script directory for file paths
SCRIPT_DIR = Path(__file__).resolve().parent

# Load environment variables from project root
env_path = SCRIPT_DIR.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded .env from: {env_path}")
else:
    load_dotenv()

# OpenRouter API configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Use a capable model for validation
VALIDATION_MODEL = "xiaomi/mimo-v2-flash:free"  # Free model for cost-effective validation

# Files
SENTIMENT_RESULTS_FILE = str(SCRIPT_DIR / "sentiment_analysis_results.json")
CONVERSATIONS_FILE = str(SCRIPT_DIR / "dummy_conversations.json")
VALIDATION_OUTPUT_FILE = str(SCRIPT_DIR / "sentiment_validation_results.json")


def load_data() -> Tuple[List[Dict], List[Dict]]:
    """Load sentiment results and original conversations."""
    with open(SENTIMENT_RESULTS_FILE, "r", encoding="utf-8") as f:
        sentiment_results = json.load(f)
    
    with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    
    # Create lookup by conversation_id
    conv_lookup = {c["conversation_id"]: c for c in conversations}
    
    return sentiment_results, conv_lookup


def random_sample(sentiment_results: List[Dict], sample_size: int = 100) -> List[Dict]:
    """
    Pure random sampling - no bias from DistilBERT classifications.
    LLM will independently evaluate sentiment without any pre-selection bias.
    """
    n = min(sample_size, len(sentiment_results))
    sampled = random.sample(sentiment_results, n)
    print(f"Randomly sampled {len(sampled)} conversations from {len(sentiment_results)} total")
    return sampled


def get_llm_sentiment_scores(turns_text: List[str]) -> Dict:
    """
    Ask LLM to score sentiment for each turn in a conversation.
    
    Returns a dict with:
    - scores: list of sentiment scores (-1 to +1) for each turn
    - categories: list of sentiment categories for each turn
    - reasoning: brief explanation
    """
    
    turns_formatted = "\n".join([f"Turn {i+1}: {text}" for i, text in enumerate(turns_text)])
    
    prompt = f"""Analyze the sentiment of each user message in this conversation.

For EACH turn, provide:
1. A sentiment score from -1.0 (very negative) to +1.0 (very positive)
2. A category: "positive", "neutral", or "negative"

Consider:
- Frustration, confusion, or dissatisfaction = negative
- Neutral questions seeking information = neutral
- Expressions of gratitude, understanding, or satisfaction = positive
- Criticism of unhelpful responses = negative

CONVERSATION:
{turns_formatted}

Output ONLY valid JSON in this exact format:
{{
  "turn_sentiments": [
    {{"turn": 1, "score": 0.2, "category": "neutral", "reason": "brief reason"}},
    {{"turn": 2, "score": -0.5, "category": "negative", "reason": "brief reason"}},
    ...
  ],
  "overall_trend": "improving" | "stable" | "worsening"
}}"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Sentiment Validation"
    }
    
    payload = {
        "model": VALIDATION_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,  # Low temperature for consistency
        "max_tokens": 2000,
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        
        # Parse JSON
        return json.loads(content)
        
    except Exception as e:
        print(f"  LLM error: {e}")
        return None


def compare_sentiments(distilbert_result: Dict, llm_result: Dict) -> Dict:
    """
    Compare DistilBERT and LLM sentiment scores.
    
    Returns comparison metrics.
    """
    if not llm_result or "turn_sentiments" not in llm_result:
        return {"error": "LLM result invalid"}
    
    turn_analysis = distilbert_result.get("turn_analysis", [])
    llm_turns = llm_result.get("turn_sentiments", [])
    
    if len(turn_analysis) != len(llm_turns):
        return {"error": f"Turn count mismatch: {len(turn_analysis)} vs {len(llm_turns)}"}
    
    # Compare scores and categories
    score_diffs = []
    category_matches = 0
    direction_matches = 0
    comparisons = []
    
    for i, (db, llm) in enumerate(zip(turn_analysis, llm_turns)):
        db_score = db.get("raw_sentiment", 0)
        db_category = db.get("sentiment_category", "neutral")
        
        llm_score = llm.get("score", 0)
        llm_category = llm.get("category", "neutral")
        
        score_diff = abs(db_score - llm_score)
        score_diffs.append(score_diff)
        
        cat_match = db_category == llm_category
        if cat_match:
            category_matches += 1
        
        # Compare gradient direction (for turns > 1)
        if i > 0:
            db_direction = turn_analysis[i].get("direction_label", "stable")
            
            # Infer LLM direction from score change
            prev_llm_score = llm_turns[i-1].get("score", 0)
            llm_diff = llm_score - prev_llm_score
            
            if llm_diff > 0.1:
                llm_direction = "improving"
            elif llm_diff < -0.1:
                llm_direction = "worsening"
            else:
                llm_direction = "stable"
            
            if db_direction == llm_direction:
                direction_matches += 1
        
        comparisons.append({
            "turn": i + 1,
            "distilbert_score": round(db_score, 4),
            "llm_score": round(llm_score, 4),
            "score_diff": round(score_diff, 4),
            "distilbert_category": db_category,
            "llm_category": llm_category,
            "category_match": cat_match,
            "llm_reason": llm.get("reason", "")
        })
    
    # Overall trend comparison
    db_slope = distilbert_result.get("session_features", {}).get("trend_slope", 0)
    if db_slope > 0.01:
        db_trend = "improving"
    elif db_slope < -0.01:
        db_trend = "worsening"
    else:
        db_trend = "stable"
    
    llm_trend = llm_result.get("overall_trend", "stable")
    
    return {
        "turn_comparisons": comparisons,
        "avg_score_diff": round(np.mean(score_diffs), 4),
        "max_score_diff": round(np.max(score_diffs), 4),
        "category_agreement": round(category_matches / len(turn_analysis), 4),
        "direction_agreement": round(direction_matches / max(1, len(turn_analysis) - 1), 4),
        "distilbert_trend": db_trend,
        "llm_trend": llm_trend,
        "trend_match": db_trend == llm_trend
    }


def run_validation(sample_size: int = 100, delay: float = 0.5):
    """
    Run the full validation pipeline.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set!")
    
    print("=" * 60)
    print("LLM SENTIMENT VALIDATION")
    print("=" * 60)
    print(f"Validation model: {VALIDATION_MODEL}")
    print(f"Sample size: {sample_size}")
    print()
    
    # Load data
    print("Loading data...")
    sentiment_results, conv_lookup = load_data()
    print(f"  Sentiment results: {len(sentiment_results)}")
    print(f"  Conversations: {len(conv_lookup)}")
    print()
    
    # Sample
    print("Random sampling...")
    sampled = random_sample(sentiment_results, sample_size)
    print()
    
    # Validate each sampled conversation
    print("Running LLM validation...")
    print("-" * 50)
    
    validation_results = []
    total_category_agreement = 0
    total_direction_agreement = 0
    total_score_diff = 0
    trend_matches = 0
    successful = 0
    
    for i, result in enumerate(sampled):
        conv_id = result["conversation_id"]
        conv = conv_lookup.get(conv_id)
        
        if not conv:
            print(f"  [{i+1}] Conversation not found: {conv_id}")
            continue
        
        turns_text = [t.get("text", "") for t in conv.get("turns", [])]
        
        print(f"  [{i+1}/{len(sampled)}] {conv_id[:8]}... ({len(turns_text)} turns)", end=" ")
        
        # Get LLM sentiment
        llm_result = get_llm_sentiment_scores(turns_text)
        
        if llm_result:
            # Compare
            comparison = compare_sentiments(result, llm_result)
            
            if "error" not in comparison:
                successful += 1
                total_category_agreement += comparison["category_agreement"]
                total_direction_agreement += comparison["direction_agreement"]
                total_score_diff += comparison["avg_score_diff"]
                if comparison["trend_match"]:
                    trend_matches += 1
                
                print(f"✓ Cat: {comparison['category_agreement']:.0%}, "
                      f"Dir: {comparison['direction_agreement']:.0%}, "
                      f"ΔScore: {comparison['avg_score_diff']:.2f}")
                
                validation_results.append({
                    "conversation_id": conv_id,
                    "sample_category": result.get("_sample_category", "unknown"),
                    "num_turns": len(turns_text),
                    "comparison": comparison,
                    "llm_result": llm_result
                })
            else:
                print(f"✗ {comparison['error']}")
        else:
            print("✗ LLM failed")
        
        # Rate limiting
        time.sleep(delay)
    
    # Save results
    print()
    print("-" * 50)
    print("Saving results...")
    
    summary = {
        "model_validated": "distilbert-base-uncased-finetuned-sst-2-english",
        "validation_model": VALIDATION_MODEL,
        "sample_size": len(sampled),
        "successful_validations": successful,
        "overall_metrics": {
            "avg_category_agreement": round(total_category_agreement / max(1, successful), 4),
            "avg_direction_agreement": round(total_direction_agreement / max(1, successful), 4),
            "avg_score_difference": round(total_score_diff / max(1, successful), 4),
            "trend_agreement": round(trend_matches / max(1, successful), 4)
        },
        "validations": validation_results
    }
    
    with open(VALIDATION_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {VALIDATION_OUTPUT_FILE}")
    print()
    
    # Print summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Successful validations: {successful}/{len(sampled)}")
    print()
    print("Agreement Metrics:")
    print(f"  Category Agreement:  {summary['overall_metrics']['avg_category_agreement']:.1%}")
    print(f"  Direction Agreement: {summary['overall_metrics']['avg_direction_agreement']:.1%}")
    print(f"  Trend Agreement:     {summary['overall_metrics']['trend_agreement']:.1%}")
    print(f"  Avg Score Diff:      {summary['overall_metrics']['avg_score_difference']:.3f}")
    print()
    
    # Analyze disagreements
    disagreements = [v for v in validation_results 
                    if v["comparison"]["category_agreement"] < 0.5]
    
    if disagreements:
        print(f"\nHigh Disagreement Cases ({len(disagreements)}):")
        for d in disagreements[:5]:
            print(f"  - {d['conversation_id'][:8]}: "
                  f"Cat agreement: {d['comparison']['category_agreement']:.0%}")
    
    return summary


if __name__ == "__main__":
    # Run validation with 100 samples (adjust as needed)
    # More samples = more accurate but more expensive/slow
    run_validation(sample_size=100, delay=0.5)
