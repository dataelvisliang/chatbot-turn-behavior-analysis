"""
Task 2: Turn-by-Turn Behavior Analysis

Analyzes Turn 1 vs Turn 2 and Turn 2 vs Turn 3 using:
1. Levenshtein Distance (literal similarity)
2. BGE Reranker v2 m3 (cross-encoder semantic/relevance score)

Classifies user behavior into quadrants (using median-based thresholds):
- Quadrant I: The Repeater (high semantic + high literal)
- Quadrant II: The Paraphraser (high semantic + low literal)
- Quadrant III: The Jumper (low semantic + low literal)
- Quadrant IV: The Refiner (low semantic + high literal)
"""

import json
import os
import numpy as np
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import torch
from sentence_transformers import CrossEncoder
import Levenshtein

# Load environment variables
load_dotenv()

# Configuration
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def compute_levenshtein_similarity(text1: str, text2: str) -> float:
    """
    Compute normalized Levenshtein similarity (0-1).
    1 = identical, 0 = completely different.
    """
    if not text1 or not text2:
        return 0.0
    
    distance = Levenshtein.distance(text1.lower(), text2.lower())
    max_len = max(len(text1), len(text2))
    
    if max_len == 0:
        return 1.0
    
    similarity = 1 - (distance / max_len)
    return similarity


def classify_quadrant(
    semantic_score: float, 
    literal_score: float,
    semantic_threshold: float,
    literal_threshold: float
) -> Tuple[str, str]:
    """
    Classify into behavior quadrant based on semantic and literal similarity.
    
    Returns: (quadrant_number, behavior_label)
    """
    high_semantic = semantic_score >= semantic_threshold
    high_literal = literal_score >= literal_threshold
    
    if high_semantic and high_literal:
        return ("I", "Repeater")
    elif high_semantic and not high_literal:
        return ("II", "Paraphraser")
    elif not high_semantic and not high_literal:
        return ("III", "Jumper")
    else:  # not high_semantic and high_literal
        return ("IV", "Refiner")


class TurnAnalyzer:
    def __init__(self, use_gpu: bool = True):
        """
        Initialize the Turn Analyzer.
        """
        print("Loading BGE Reranker model...")
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        self.reranker = CrossEncoder(RERANKER_MODEL, device=device)
        print("Model loaded successfully!")
    
    def analyze_turn_pair(self, turn1_text: str, turn2_text: str) -> Dict:
        """
        Analyze a pair of turns (scores only, no classification yet).
        """
        result = {}
        
        # 1. Levenshtein similarity (literal)
        result["levenshtein_similarity"] = round(
            compute_levenshtein_similarity(turn1_text, turn2_text), 4
        )
        
        # 2. Reranker score (semantic/relevance) - RAW score
        raw_score = self.reranker.predict([(turn1_text, turn2_text)])[0]
        result["reranker_score_raw"] = round(float(raw_score), 4)
        
        return result
    
    def analyze_all_conversations(
        self, 
        conversations: List[Dict],
        save_interval: int = 50,
        output_file: str = "turn_analysis_results.json"
    ) -> List[Dict]:
        """
        Analyze all conversations in two phases:
        1. Collect all scores
        2. Compute median thresholds and classify
        """
        results = []
        total = len(conversations)
        
        print(f"\n=== PHASE 1: Collecting scores from {total} conversations ===")
        print("-" * 50)
        
        # Phase 1: Collect scores
        for i, conv in enumerate(conversations):
            try:
                turns = conv.get("turns", [])
                
                analysis = {
                    "conversation_id": conv.get("conversation_id"),
                    "num_turns": len(turns),
                    "turn_analyses": []
                }
                
                if len(turns) >= 2:
                    # Turn 1 vs Turn 2
                    turn1_text = turns[0].get("text", "")
                    turn2_text = turns[1].get("text", "")
                    
                    pair_1_2 = self.analyze_turn_pair(turn1_text, turn2_text)
                    pair_1_2["pair"] = [1, 2]
                    pair_1_2["turn_1_text"] = turn1_text[:200] + "..." if len(turn1_text) > 200 else turn1_text
                    pair_1_2["turn_2_text"] = turn2_text[:200] + "..." if len(turn2_text) > 200 else turn2_text
                    analysis["turn_analyses"].append(pair_1_2)
                    
                    # Turn 2 vs Turn 3
                    if len(turns) >= 3:
                        turn3_text = turns[2].get("text", "")
                        pair_2_3 = self.analyze_turn_pair(turn2_text, turn3_text)
                        pair_2_3["pair"] = [2, 3]
                        pair_2_3["turn_2_text"] = turn2_text[:200] + "..." if len(turn2_text) > 200 else turn2_text
                        pair_2_3["turn_3_text"] = turn3_text[:200] + "..." if len(turn3_text) > 200 else turn3_text
                        analysis["turn_analyses"].append(pair_2_3)
                
                results.append(analysis)
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{total} conversations...")
                    
            except Exception as e:
                print(f"Error processing conversation {i}: {e}")
                continue
        
        # Phase 2: Compute median thresholds
        print("\n=== PHASE 2: Computing thresholds and classifying ===")
        
        all_semantic = []
        all_literal = []
        
        for result in results:
            for turn_analysis in result.get("turn_analyses", []):
                all_semantic.append(turn_analysis.get("reranker_score_raw", 0))
                all_literal.append(turn_analysis.get("levenshtein_similarity", 0))
        
        semantic_threshold = float(np.median(all_semantic))
        literal_threshold = float(np.median(all_literal))
        
        print(f"Semantic Threshold (median): {semantic_threshold:.4f}")
        print(f"Literal Threshold (median):  {literal_threshold:.4f}")
        
        # Phase 3: Classify using median thresholds
        for result in results:
            for turn_analysis in result.get("turn_analyses", []):
                semantic_score = turn_analysis.get("reranker_score_raw", 0)
                literal_score = turn_analysis.get("levenshtein_similarity", 0)
                
                quadrant, behavior = classify_quadrant(
                    semantic_score, 
                    literal_score,
                    semantic_threshold,
                    literal_threshold
                )
                turn_analysis["quadrant"] = quadrant
                turn_analysis["behavior_label"] = behavior
        
        # Save results
        self._save_results(results, output_file)
        
        return results, semantic_threshold, literal_threshold
    
    def _save_results(self, results: List[Dict], output_file: str):
        """Save results to JSON file."""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def print_summary(self, results: List[Dict], semantic_threshold: float, literal_threshold: float):
        """Print summary statistics with score distributions."""
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Collect all scores
        all_semantic = []
        all_literal = []
        quadrant_counts = {"I": 0, "II": 0, "III": 0, "IV": 0}
        total_pairs = 0
        
        for result in results:
            for turn_analysis in result.get("turn_analyses", []):
                all_semantic.append(turn_analysis.get("reranker_score_raw", 0))
                all_literal.append(turn_analysis.get("levenshtein_similarity", 0))
                quadrant = turn_analysis.get("quadrant")
                if quadrant:
                    quadrant_counts[quadrant] += 1
                    total_pairs += 1
        
        all_semantic = np.array(all_semantic)
        all_literal = np.array(all_literal)
        
        print(f"\nTotal conversations analyzed: {len(results)}")
        print(f"Total turn pairs analyzed: {total_pairs}")
        
        print("\n" + "-" * 40)
        print("THRESHOLDS USED (median-based):")
        print("-" * 40)
        print(f"Semantic (reranker_raw): {semantic_threshold:.4f}")
        print(f"Literal (levenshtein):   {literal_threshold:.4f}")
        
        print("\n" + "-" * 40)
        print("RERANKER SCORE (RAW) DISTRIBUTION:")
        print("-" * 40)
        print(f"Min:    {np.min(all_semantic):.4f}")
        print(f"Max:    {np.max(all_semantic):.4f}")
        print(f"Mean:   {np.mean(all_semantic):.4f}")
        print(f"Median: {np.median(all_semantic):.4f}")
        print(f"Std:    {np.std(all_semantic):.4f}")
        
        print("\n" + "-" * 40)
        print("LEVENSHTEIN SIMILARITY DISTRIBUTION:")
        print("-" * 40)
        print(f"Min:    {np.min(all_literal):.4f}")
        print(f"Max:    {np.max(all_literal):.4f}")
        print(f"Mean:   {np.mean(all_literal):.4f}")
        print(f"Median: {np.median(all_literal):.4f}")
        print(f"Std:    {np.std(all_literal):.4f}")
        
        print("\n" + "-" * 40)
        print("BEHAVIOR DISTRIBUTION:")
        print("-" * 40)
        print(f"Quadrant I  (Repeater):    {quadrant_counts['I']:4d} ({100*quadrant_counts['I']/max(1,total_pairs):5.1f}%)")
        print(f"Quadrant II (Paraphraser): {quadrant_counts['II']:4d} ({100*quadrant_counts['II']/max(1,total_pairs):5.1f}%)")
        print(f"Quadrant III (Jumper):     {quadrant_counts['III']:4d} ({100*quadrant_counts['III']/max(1,total_pairs):5.1f}%)")
        print(f"Quadrant IV (Refiner):     {quadrant_counts['IV']:4d} ({100*quadrant_counts['IV']/max(1,total_pairs):5.1f}%)")


def main():
    input_file = "dummy_conversations.json"
    output_file = "turn_analysis_results.json"
    
    print(f"Loading conversations from {input_file}...")
    
    with open(input_file, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    
    print(f"Loaded {len(conversations)} conversations")
    
    # Initialize analyzer
    analyzer = TurnAnalyzer(use_gpu=True)
    
    # Run analysis
    results, semantic_threshold, literal_threshold = analyzer.analyze_all_conversations(
        conversations,
        save_interval=50,
        output_file=output_file
    )
    
    # Print summary
    analyzer.print_summary(results, semantic_threshold, literal_threshold)
    
    print(f"\n✅ Results saved to {output_file}")


if __name__ == "__main__":
    main()
