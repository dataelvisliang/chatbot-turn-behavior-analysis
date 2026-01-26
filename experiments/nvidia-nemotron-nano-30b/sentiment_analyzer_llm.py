"""
LLM-based Sentiment Analyzer (Satisfaction Focused)
Replacing Twitter-RoBERTa with Nvidia Nemotron (via OpenRouter) 
to correctly measure Interaction Satisfaction in Mental Health.
"""

import json
import os
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import requests
from typing import List, Dict

# Config
SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR.parent.parent / '.env'
load_dotenv(ENV_PATH)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "nvidia/nemotron-3-nano-30b-a3b:free" 

INPUT_FILE = str(SCRIPT_DIR / "mental_health_conversations.json")
OUTPUT_FILE = str(SCRIPT_DIR / "sentiment_analysis_llm_results.json")

def get_llm_satisfaction_score(text: str, retries=3) -> float:
    """
    Get satisfaction score (-1.0 to 1.0) for a single turn.
    Prompt focuses STRICTLY on satisfaction with the AI, ignoring user distress.
    """
    prompt = f"""Analyze the user's SATISFACTION with the AI assistant based on this message.
    
    CRITICAL RULE: Ignore the user's personal life struggles, pain, or bad mood. 
    Only judge if they are happy/unhappy with the AI's RESPONSE.
    
    Examples:
    - "I feel hopeless and want to give up." -> SCORE: 0.0 (Neutral/Trusting).
    - "That advice is useless." -> SCORE: -1.0 (Dissatisfied).
    - "Thanks, I'll try that." -> SCORE: +1.0 (Satisfied).
    - "I am so anxious right now." -> SCORE: 0.0 (Neutral/Disclosing).
    
    Text: "{text}"
    
    Return ONLY a JSON object with a single "score" field (-1.0 to 1.0).
    Example: {{"score": 0.5}}
    """

    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost"
    }
    
    for attempt in range(retries):
        try:
            resp = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                content = data['choices'][0]['message']['content']
                # Parse JSON
                try:
                    score = json.loads(content).get('score', 0.0)
                    # Clamp
                    return max(-1.0, min(1.0, float(score)))
                except json.JSONDecodeError:
                    # Fallback parsing
                    import re
                    match = re.search(r"[-+]?\d*\.\d+|\d+", content)
                    if match:
                        return float(match.group())
                    return 0.0
            elif resp.status_code == 429:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"API Error {resp.status_code}: {resp.text}")
                return 0.0
        except Exception as e:
            print(f"Request failed: {e}")
            time.sleep(1)
            
    return 0.0 # Default to Neutral on failure

def analyze_session(conversation: Dict) -> Dict:
    """Analyze a single conversation session."""
    turns = conversation['turns']
    scores = []
    
    turn_analysis = []
    
    for i, turn in enumerate(turns):
        text = turn['text']
        # Call LLM
        sentiment = get_llm_satisfaction_score(text)
        scores.append(sentiment)
        
        # Determine category for report
        if sentiment < -0.1: cat = "negative"
        elif sentiment > 0.1: cat = "positive"
        else: cat = "neutral"
        
        # Calculate Gradient (vs previous)
        gradient = 0.0
        direction_label = "stable"
        if i > 0:
            gradient = sentiment - scores[i-1]
            if gradient > 0.1: direction_label = "improving"
            elif gradient < -0.1: direction_label = "worsening"
            
        turn_analysis.append({
            "turn": i + 1,
            "text_snippet": text[:50],
            "sentiment_score": sentiment,
            "sentiment_category": cat,
            "gradient": gradient,
            "direction_label": direction_label
        })
        
        time.sleep(0.5) # Rate limiting
        
    # Session Features
    if not scores: return {}
    
    initial = scores[0]
    final = scores[-1]
    
    # Simple linear regression for slope
    x = np.arange(len(scores))
    if len(scores) > 1:
        slope, _ = np.polyfit(x, scores, 1)
    else:
        slope = 0.0

    return {
        "conversation_id": conversation['conversation_id'],
        "num_turns": len(turns),
        "raw_scores": scores,
        "session_features": {
            "initial_sentiment": initial,
            "final_sentiment": final,
            "sentiment_delta": final - initial,
            "trend_slope": slope,
            "lowest_point": min(scores),
            "highest_point": max(scores)
        },
        "turn_analysis": turn_analysis
    }

def run_analysis(sample_size=50):
    print(f"Starting LLM Sentiment Analysis on independent sample of {sample_size} conversations...")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
        
    # User requested re-run. We'll take the first N (or random N).
    # Being deterministic is better for comparison.
    target_convos = conversations[:sample_size]
    
    results = []
    
    for i, conv in enumerate(target_convos):
        print(f"[{i+1}/{sample_size}] Analyzing {conv['conversation_id'][:8]}...", end="", flush=True)
        res = analyze_session(conv)
        results.append(res)
        
        # Log slope
        slope = res['session_features']['trend_slope']
        print(f" Slope: {slope:.3f}")
        
    # Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nAnalysis complete. Results saved to {OUTPUT_FILE}")
    
    # Generate Mini-Report statistics
    avg_slope = np.mean([r['session_features']['trend_slope'] for r in results])
    improving = sum(1 for r in results if r['session_features']['trend_slope'] > 0.01)
    worsening = sum(1 for r in results if r['session_features']['trend_slope'] < -0.01)
    
    print("\n" + "="*40)
    print("LLM SENTIMENT SUMMARY (Satisfaction)")
    print("="*40)
    print(f"Average Trend Slope: {avg_slope:.4f}")
    print(f"Improving Sessions:  {improving} ({improving/sample_size:.1%})")
    print(f"Worsening Sessions:  {worsening} ({worsening/sample_size:.1%})")
    print("="*40)

if __name__ == "__main__":
    # Run larger batch as requested
    run_analysis(500)
