"""
Task 3: Sentiment Gradient Analysis

Analyzes sentiment progression across conversation turns:
- Assigns sentiment score to each user turn (-1 to +1 scale)
- Computes sentiment gradient (delta between consecutive turns)
- Classifies gradient direction: +1 (improving), 0 (stable), -1 (worsening)
- Extracts session-level features: trend slope, max drop, recovery flag

Uses: cardiffnlp/twitter-roberta-base-sentiment-latest for sentiment analysis
Model: cardiffnlp/twitter-roberta-base-sentiment-latest (3-class: negative, neutral, positive)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from transformers import pipeline
import torch
import warnings
warnings.filterwarnings('ignore')

# Get script directory for file paths
SCRIPT_DIR = Path(__file__).resolve().parent


class SentimentGradientAnalyzer:
    def __init__(self, alpha: float = 0.3, epsilon: float = 0.05, use_gpu: bool = True):
        """
        Initialize the Sentiment Gradient Analyzer.
        
        Args:
            alpha: EMA smoothing coefficient (0-1), smaller = smoother
            epsilon: Gradient threshold, values below this are considered stable
            use_gpu: Whether to use GPU for inference
        """
        self.alpha = alpha
        self.epsilon = epsilon
        
        print("Loading sentiment analysis model...")
        device = 0 if use_gpu and torch.cuda.is_available() else -1
        print(f"Using device: {'cuda' if device == 0 else 'cpu'}")
        
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=device,
            top_k=None
        )
        print("Model loaded successfully!")
    
    def get_sentiment_score(self, text: str) -> float:
        """
        Convert sentiment to continuous score (-1 to +1).
        Score = Probability(Positive) - Probability(Negative)
        """
        # Truncate very long texts (model max is 512 tokens)
        text = text[:512]
        
        # Get all scores (top_k=None returns list of dicts for all labels)
        results = self.sentiment_analyzer(text)[0]
        
        # Parse scores
        scores = {res['label']: res['score'] for res in results}
        
        # Calculate composite score from probabilities
        # Range: -1 (100% neg) to +1 (100% pos), with 0 being neutral/ambiguous
        return scores.get('positive', 0) - scores.get('negative', 0)
    
    def score_to_100_scale(self, score: float) -> float:
        """Convert -1 to +1 score to 0-100 scale."""
        return (score + 1) * 50
    
    def _ema_smooth(self, scores: np.ndarray) -> np.ndarray:
        """Exponential Moving Average smoothing."""
        smoothed = [scores[0]]
        for s in scores[1:]:
            smoothed.append(self.alpha * s + (1 - self.alpha) * smoothed[-1])
        return np.array(smoothed)
    
    def _classify_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """
        Classify gradient direction.
        +1 = improving, 0 = stable, -1 = worsening
        """
        return np.where(
            gradient > self.epsilon, 1,
            np.where(gradient < -self.epsilon, -1, 0)
        )
    
    def _extract_session_features(
        self, 
        smoothed: np.ndarray, 
        gradient: np.ndarray, 
        gradient_label: np.ndarray
    ) -> Dict:
        """Extract conversation-level features."""
        
        # Find first negative turn (sentiment dropped)
        negative_turns = np.where(gradient_label == -1)[0]
        first_negative_turn = int(negative_turns[0]) + 1 if len(negative_turns) > 0 else None
        
        # Check for recovery (positive gradient after a negative one)
        recovery = False
        if first_negative_turn is not None and first_negative_turn < len(gradient_label):
            recovery = bool(np.any(gradient_label[first_negative_turn:] == 1))
        
        return {
            # Trend
            'trend_slope': float(np.polyfit(range(len(smoothed)), smoothed, 1)[0]) if len(smoothed) > 1 else 0.0,
            'overall_volatility': float(np.std(gradient)) if len(gradient) > 0 else 0.0,
            
            # Extremes
            'max_drop': float(np.min(gradient)) if len(gradient) > 0 else 0.0,
            'max_rise': float(np.max(gradient)) if len(gradient) > 0 else 0.0,
            'sentiment_range': float(np.ptp(smoothed)),  # peak-to-peak
            
            # Start/End
            'initial_sentiment': float(smoothed[0]),
            'final_sentiment': float(smoothed[-1]),
            'sentiment_delta': float(smoothed[-1] - smoothed[0]),
            
            # Turning points
            'first_negative_turn': first_negative_turn,
            'num_negative_turns': int(np.sum(gradient_label == -1)),
            'num_positive_turns': int(np.sum(gradient_label == 1)),
            'num_stable_turns': int(np.sum(gradient_label == 0)),
            
            # Recovery
            'recovery_detected': recovery,
            'lowest_point': float(np.min(smoothed)),
            'lowest_point_turn': int(np.argmin(smoothed)) + 1,
            'recovered_from_low': bool(smoothed[-1] > np.min(smoothed) + 0.1)
        }
    
    def analyze_conversation(self, turns: List[Dict]) -> Dict:
        """
        Analyze sentiment gradient for a single conversation.
        
        Args:
            turns: List of turn dicts with 'text' field
        
        Returns:
            Analysis dict with scores, gradients, and features
        """
        if len(turns) < 1:
            return {'error': 'No turns to analyze'}
        
        # 1. Get raw sentiment scores
        raw_scores = []
        for turn in turns:
            text = turn.get('text', '')
            score = self.get_sentiment_score(text)
            raw_scores.append(score)
        
        raw_scores = np.array(raw_scores)
        
        # 2. EMA smoothing
        smoothed = self._ema_smooth(raw_scores)
        
        # 3. Compute gradient (difference between consecutive turns)
        gradient = np.diff(smoothed) if len(smoothed) > 1 else np.array([])
        
        # 4. Classify gradient direction
        gradient_label = self._classify_gradient(gradient) if len(gradient) > 0 else np.array([])
        
        # 5. Extract session features
        features = self._extract_session_features(smoothed, gradient, gradient_label)
        
        # 6. Build turn-level analysis
        turn_analysis = []
        for i, (raw, smooth) in enumerate(zip(raw_scores, smoothed)):
            turn_info = {
                'turn': i + 1,
                'raw_sentiment': round(float(raw), 4),
                'smoothed_sentiment': round(float(smooth), 4),
                'sentiment_100_scale': round(self.score_to_100_scale(smooth), 1),
                'sentiment_category': self._categorize_sentiment(smooth)
            }
            
            if i > 0 and len(gradient) > 0:
                turn_info['gradient'] = round(float(gradient[i-1]), 4)
                turn_info['gradient_direction'] = int(gradient_label[i-1])
                turn_info['direction_label'] = {1: 'improving', 0: 'stable', -1: 'worsening'}[int(gradient_label[i-1])]
            
            turn_analysis.append(turn_info)
        
        return {
            'raw_scores': [round(float(s), 4) for s in raw_scores],
            'smoothed_scores': [round(float(s), 4) for s in smoothed],
            'gradient': [round(float(g), 4) for g in gradient],
            'gradient_labels': [int(g) for g in gradient_label],
            'session_features': features,
            'turn_analysis': turn_analysis
        }
    
    def _categorize_sentiment(self, score: float) -> str:
        """Categorize sentiment score."""
        if score > 0.3:
            return 'positive'
        elif score < -0.3:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_all_conversations(
        self,
        conversations: List[Dict],
        output_file: str = "sentiment_analysis_results.json"
    ) -> List[Dict]:
        """Analyze all conversations and save results."""
        results = []
        total = len(conversations)
        
        print(f"\nAnalyzing sentiment for {total} conversations...")
        print("-" * 50)
        
        for i, conv in enumerate(conversations):
            try:
                turns = conv.get('turns', [])
                analysis = self.analyze_conversation(turns)
                
                result = {
                    'conversation_id': conv.get('conversation_id'),
                    'num_turns': len(turns),
                    **analysis
                }
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{total} conversations...")
                    # Save progress
                    self._save_results(results, output_file)
                    
            except Exception as e:
                print(f"Error processing conversation {i}: {e}")
                continue
        
        # Final save
        self._save_results(results, output_file)
        
        return results
    
    def _save_results(self, results: List[Dict], output_file: str):
        """Save results to JSON file."""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def print_summary(self, results: List[Dict]):
        """Print summary statistics."""
        print("\n" + "=" * 60)
        print("SENTIMENT GRADIENT ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Collect metrics
        all_deltas = []
        all_slopes = []
        improving = 0
        worsening = 0
        stable = 0
        recoveries = 0
        
        for r in results:
            features = r.get('session_features', {})
            delta = features.get('sentiment_delta', 0)
            slope = features.get('trend_slope', 0)
            
            all_deltas.append(delta)
            all_slopes.append(slope)
            
            if slope > 0.01:
                improving += 1
            elif slope < -0.01:
                worsening += 1
            else:
                stable += 1
            
            if features.get('recovery_detected'):
                recoveries += 1
        
        total = len(results)
        
        print(f"\nTotal conversations analyzed: {total}")
        
        print("\n" + "-" * 40)
        print("OVERALL TRENDS:")
        print("-" * 40)
        print(f"Improving (slope > 0):  {improving:4d} ({100*improving/max(1,total):5.1f}%)")
        print(f"Stable:                 {stable:4d} ({100*stable/max(1,total):5.1f}%)")
        print(f"Worsening (slope < 0):  {worsening:4d} ({100*worsening/max(1,total):5.1f}%)")
        print(f"With recovery:          {recoveries:4d} ({100*recoveries/max(1,total):5.1f}%)")
        
        print("\n" + "-" * 40)
        print("SENTIMENT DELTA (end - start):")
        print("-" * 40)
        deltas = np.array(all_deltas)
        print(f"Min:    {np.min(deltas):.4f}")
        print(f"Max:    {np.max(deltas):.4f}")
        print(f"Mean:   {np.mean(deltas):.4f}")
        print(f"Median: {np.median(deltas):.4f}")
        
        print("\n" + "-" * 40)
        print("TREND SLOPE:")
        print("-" * 40)
        slopes = np.array(all_slopes)
        print(f"Min:    {np.min(slopes):.4f}")
        print(f"Max:    {np.max(slopes):.4f}")
        print(f"Mean:   {np.mean(slopes):.4f}")
        print(f"Median: {np.median(slopes):.4f}")


def main():
    # Use script directory for file paths
    input_file = str(SCRIPT_DIR / "dummy_conversations.json")
    output_file = str(SCRIPT_DIR / "sentiment_analysis_results.json")
    
    print(f"Loading conversations from {input_file}...")
    
    with open(input_file, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    
    print(f"Loaded {len(conversations)} conversations")
    
    # Initialize analyzer
    analyzer = SentimentGradientAnalyzer(
        alpha=0.3,      # EMA smoothing coefficient
        epsilon=0.05,   # Gradient threshold for stable classification
        use_gpu=True
    )
    
    # Run analysis
    results = analyzer.analyze_all_conversations(
        conversations,
        output_file=output_file
    )
    
    # Print summary
    analyzer.print_summary(results)
    
    print(f"\n✅ Results saved to {output_file}")


if __name__ == "__main__":
    main()
