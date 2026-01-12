You are an assistant helping to simulate and analyze multi-turn conversations for a chatbot study.

Task 1: Generate dummy data
- Generate 500 conversations.
- Each conversation should have 3–5 user turns, alternating with bot turns (average 4 user turns per conversation).
- Each turn should be short, realistic English sentences reflecting typical user and bot interactions (complaints, questions, feedback, clarifications, or help requests).

Output format:
[
  {
    "conversation_id": <unique_id>,
    "turns": [
      {"speaker": "user", "text": "..."},
      {"speaker": "bot", "text": "..."},
      ...
    ]
  },
  ...
]

---

Task 2: Analyze Turn 1 vs Turn 2 and Turn 2 vs Turn 3
- For each conversation, analyze the similarity or behavioral link between Turn 1 (user) and Turn 2 (user) using a cross-encoder model.
- Output a similarity score (0–1) or classification describing whether the second turn is a continuation, clarification, or topic shift from the first turn.
一个需要注意的“陷阱”在做用户行为分析时，你需要明确定义什么是**“相似”**：字面相似： 用户重复同样的问题（比如因为机器人没听懂）。意图延续： 用户在细化问题（比如 第一句：“推荐个餐厅”，第二句：“要西餐”）。如果是为了分析“用户是否在重复提问”：你可以同时计算 Levenshtein Distance（编辑距离） 和 Cross-Encoder 分数。如果“编辑距离”很小 + Cross-Encoder 分数很高 -> 用户在复读/复现问题。如果“编辑距离”很大 + Cross-Encoder 分数很高 -> 用户在进行深度的意图关联追问。
在分析 Chatbot 对话时，Levenshtein Distance 和你之前提到的 Cross-Encoder（语义相似度）结合使用，可以产生非常有趣的洞察：

Case A：低编辑距离 + 高语义相似度
例子： “今天天气怎么样？” vs “今天天气如何？”
结论： 用户基本在复读，或者只是微调了词组。如果第二轮对话还是这样，可能意味着机器人第一轮没回答好。
Case B：高编辑距离 + 高语义相似度
例子： “给我推荐个好吃的馆子” vs “这附近有什么评价比较高的餐厅吗？”
结论： 用户完全换了说法但表达同一个意图。这说明用户在尝试通过改变表述来让 AI 理解。
Case C：高编辑距离 + 低语义相似度
例子： “帮我订机票” vs “算了，我还是坐火车吧”
结论： 用户彻底改变了主意，或者开启了全新的话题。

构建“用户意图行为矩阵” (Behavior Matrix)
通过结合“语义相似度”和“字面相似度”，你可以将用户在第二轮（Turn 2）的行为划分为四个象限：
象限 I：高语义 + 高字面 (The Repeater)
特征： 用户几乎在复读第一句。
分析： 极大概率是因为 Chatbot 第一轮没反应、报错或回答完全牛头不对马嘴。这反映了系统可用性问题。
象限 II：高语义 + 低字面 (The Paraphraser)
特征： 意思一样，但换了完全不同的表达方式。
分析： 用户觉得 AI 没听懂，正在尝试“调教”或“顺着 AI 的思路改写”。这是模型理解力薄弱的信号。
象限 III：低语义 + 低字面 (The Jumper)
特征： 话题完全变了。
分析： 正常的用户路径切换。如果这种情况极多，说明你的 Chatbot 引导（Prompting）可能没能让用户停留在特定任务中。
象限 IV：低语义 + 高字面 (The Refiner)
特征： 词汇差不多，但意思变了（例如从“我要订票”变成“我不订票了”）。
分析： 这是一个关键信号，说明用户在做逻辑修正或否定。如果 Cross-Encoder 没识别出这种细微差别，说明模型对否定词不敏感。

---

Task 3: Sentiment Gradient
- Assign a sentiment score to each user turn (0 = very frustrated, 50 = neutral, 100 = satisfied).
- Compute the sentiment gradient across turns (Δ between consecutive turns).
- Include:
  - Direction: +1 improving, 0 stable, -1 worsening
  - Magnitude: normalized change
- Output a summary per conversation with:
  - sentiment sequence
  - gradient sequence
  - overall trend slope
  - max drop
  - recovery flag


Sentiment Gradient Analysis Example Codes:
import numpy as np
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

class SentimentGradientAnalyzer:
    def __init__(self, alpha=0.3, epsilon=0.05):
        """
        Args:
            alpha: EMA平滑系数 (0-1)，越小越平滑
            epsilon: 梯度阈值，低于此值视为无变化
        """
        self.alpha = alpha
        self.epsilon = epsilon
        # 使用更稳定的情感分析器
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # 强制使用CPU
        )
    
    def get_sentiment_score(self, text):
        """
        将情感转为-1到1的连续分数（而不是0到1）
        POSITIVE score高 -> 正分数
        NEGATIVE score高 -> 负分数
        """
        result = self.sentiment_analyzer(text)[0]
        
        if result['label'] == 'POSITIVE':
            # score 0.5-1.0 映射到 0到1
            return 2 * result['score'] - 1  
        else:  # NEGATIVE
            # score 0.5-1.0 映射到 -1到0
            return -(2 * result['score'] - 1)
    
    def analyze_conversation(self, dialogues):
        """完整的对话情感梯度分析"""
        
        # 1. 获取原始情感分数
        raw_scores = []
        for text in dialogues:
            score = self.get_sentiment_score(text)
            raw_scores.append(score)
        
        raw_scores = np.array(raw_scores)
        
        # 2. EMA平滑
        smoothed = self._ema_smooth(raw_scores)
        
        # 3. 计算梯度
        gradient = np.diff(smoothed)
        
        # 4. 梯度方向分类
        gradient_label = self._classify_gradient(gradient)
        
        # 5. 计算session级特征
        features = self._extract_session_features(smoothed, gradient, gradient_label)
        
        return {
            'raw_scores': raw_scores,
            'smoothed_scores': smoothed,
            'gradient': gradient,
            'gradient_label': gradient_label,
            'session_features': features,
            'turn_analysis': self._turn_level_analysis(smoothed, gradient, gradient_label)
        }
    
    def _ema_smooth(self, scores):
        """指数移动平均平滑"""
        smoothed = [scores[0]]
        for s in scores[1:]:
            smoothed.append(self.alpha * s + (1 - self.alpha) * smoothed[-1])
        return np.array(smoothed)
    
    def _classify_gradient(self, gradient):
        """梯度方向分类"""
        return np.where(
            gradient > self.epsilon, 1,
            np.where(gradient < -self.epsilon, -1, 0)
        )
    
    def _extract_session_features(self, smoothed, gradient, gradient_label):
        """提取对话级特征"""
        
        # 安全地找到首次负向轮次
        negative_turns = np.where(gradient_label == -1)[0]
        first_negative_turn = int(negative_turns[0]) + 1 if len(negative_turns) > 0 else None
        
        # 检测恢复（负向之后是否有正向）
        recovery = False
        if first_negative_turn is not None and first_negative_turn < len(gradient_label):
            recovery = np.any(gradient_label[first_negative_turn:] == 1)
        
        return {
            # 基础统计
            'cumulative_sentiment': np.cumsum(smoothed),
            'trend_slope': np.polyfit(range(len(smoothed)), smoothed, 1)[0],
            'overall_volatility': np.std(gradient),
            
            # 极值特征
            'max_drop': np.min(gradient) if len(gradient) > 0 else 0,
            'max_rise': np.max(gradient) if len(gradient) > 0 else 0,
            'sentiment_range': np.ptp(smoothed),  # peak-to-peak范围
            
            # 起止点特征
            'initial_sentiment': smoothed[0],
            'final_sentiment': smoothed[-1],
            'sentiment_delta': smoothed[-1] - smoothed[0],
            
            # 转折点特征
            'first_negative_turn': first_negative_turn,
            'num_negative_turns': np.sum(gradient_label == -1),
            'num_positive_turns': np.sum(gradient_label == 1),
            'num_stable_turns': np.sum(gradient_label == 0),
            
            # 恢复性特征
            'recovery_detected': recovery,
            'lowest_point': np.min(smoothed),
            'lowest_point_turn': int(np.argmin(smoothed)),
            'recovered_from_low': smoothed[-1] > np.min(smoothed) + 0.1,  # 从最低点恢复超过0.1
        }
    
    def _turn_level_analysis(self, smoothed, gradient, gradient_label):
        """逐轮详细分析"""
        turns = []
        
        for i in range(len(smoothed)):
            turn_info = {
                'turn': i + 1,
                'sentiment': float(smoothed[i]),
                'sentiment_category': self._categorize_sentiment(smoothed[i])
            }
            
            if i > 0:
                turn_info['gradient'] = float(gradient[i-1])
                turn_info['gradient_direction'] = int(gradient_label[i-1])
                turn_info['change_magnitude'] = abs(float(gradient[i-1]))
            
            turns.append(turn_info)
        
        return turns
    
    def _categorize_sentiment(self, score):
        """情感分类"""
        if score > 0.3:
            return 'positive'
        elif score < -0.3:
            return 'negative'
        else:
            return 'neutral'
    
    def print_report(self, results):
        """打印友好的分析报告"""
        print("=" * 60)
        print("对话情感梯度分析报告")
        print("=" * 60)
        
        print("\n📊 原始情感序列:")
        print(f"   {np.round(results['raw_scores'], 3)}")
        
        print("\n📈 平滑后情感序列:")
        print(f"   {np.round(results['smoothed_scores'], 3)}")
        
        print("\n📉 梯度变化:")
        print(f"   {np.round(results['gradient'], 3)}")
        
        print("\n🔄 梯度方向 (1=上升, 0=稳定, -1=下降):")
        print(f"   {results['gradient_label']}")
        
        features = results['session_features']
        
        print("\n" + "=" * 60)
        print("对话级特征")
        print("=" * 60)
        
        print(f"\n🎯 整体趋势:")
        print(f"   初始情感: {features['initial_sentiment']:.3f}")
        print(f"   最终情感: {features['final_sentiment']:.3f}")
        print(f"   情感变化: {features['sentiment_delta']:.3f}")
        print(f"   趋势斜率: {features['trend_slope']:.3f} ({'改善' if features['trend_slope'] > 0 else '恶化'})")
        
        print(f"\n📊 波动性:")
        print(f"   最大单轮下降: {features['max_drop']:.3f}")
        print(f"   最大单轮上升: {features['max_rise']:.3f}")
        print(f"   情感波动度: {features['overall_volatility']:.3f}")
        
        print(f"\n🔍 转折点:")
        print(f"   首次负向转折: 第 {features['first_negative_turn']} 轮" if features['first_negative_turn'] else "   无负向转折")
        print(f"   负向轮次数: {features['num_negative_turns']}")
        print(f"   正向轮次数: {features['num_positive_turns']}")
        print(f"   稳定轮次数: {features['num_stable_turns']}")
        
        print(f"\n💡 恢复性:")
        print(f"   最低情感点: {features['lowest_point']:.3f} (第{features['lowest_point_turn'] + 1}轮)")
        print(f"   检测到恢复: {'是' if features['recovery_detected'] else '否'}")
        print(f"   从低点恢复: {'是' if features['recovered_from_low'] else '否'}")
        
        print("\n" + "=" * 60)
        print("逐轮详细分析")
        print("=" * 60)
        
        for turn in results['turn_analysis']:
            print(f"\n轮次 {turn['turn']}:")
            print(f"  情感分数: {turn['sentiment']:.3f} ({turn['sentiment_category']})")
            if 'gradient' in turn:
                direction = {1: '↗️上升', 0: '→稳定', -1: '↘️下降'}[turn['gradient_direction']]
                print(f"  梯度: {turn['gradient']:.3f} {direction}")


# ===== 使用示例 =====
if __name__ == "__main__":
    
    # 你的示例数据
    dialogues = [
        "I love this product!", 
        "But the delivery was late.", 
        "However, the support was helpful."
    ]
    
    # 初始化分析器
    analyzer = SentimentGradientAnalyzer(alpha=0.3, epsilon=0.05)
    
    # 分析对话
    results = analyzer.analyze_conversation(dialogues)
    
    # 打印报告
    analyzer.print_report(results)
    
    print("\n" + "=" * 60)
    print("Tax RAG场景示例")
    print("=" * 60)
    
    # Tax场景的真实对话
    tax_dialogues = [
        "I'm confused about the home office deduction rules",
        "Thank you for explaining, but I'm still not sure if I qualify",
        "Oh I see! According to IRS Publication 587, I need exclusive use. That makes sense!",
        "Wait, does that mean I can't claim it if my kids use the room for homework?",
        "Got it. Thanks for the detailed explanation with the actual regulation citation!",
        "This is really helpful. I feel much more confident now."
    ]
    
    tax_results = analyzer.analyze_conversation(tax_dialogues)
    analyzer.print_report(tax_results)
    
    # 访问特定指标
    print("\n📌 关键指标快速访问:")
    print(f"情感改善了吗? {tax_results['session_features']['sentiment_delta'] > 0}")
    print(f"对话质量评分: {(tax_results['session_features']['final_sentiment'] + 1) / 2 * 100:.1f}%")
    
    # 如果你想保存结果供后续分析
    import json
    
    # 转换numpy类型为Python原生类型以便JSON序列化
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_json_serializable(tax_results)
    
    # 保存到文件
    with open('sentiment_analysis_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print("\n✅ 结果已保存到 sentiment_analysis_results.json")
    
---

Constraints:
- Keep conversations realistic and diverse.
- Ensure sentiment changes are noticeable to produce meaningful gradients.
- All output should be in JSON format.
