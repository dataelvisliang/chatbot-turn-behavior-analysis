
# Analysis Report: Mental Health Support (NVIDIA Nemotron Nano 30B)

## 1. Overview
This experiment analyzes **2000 multi-turn conversations** in the **Mental Health & Wellness** domain. The goal was to understand user behavior patterns and sentiment progression when interacting with an AI support assistant.

- **Model used**: `nvidia/nemotron-3-nano-30b-a3b`
- **Domain**: Mental Health, Anxiety, Sleep, Work-Life Balance
- **Total Conversations**: 2000
- **Total Turns**: ~12,400 (Avg 6.2 turns/conv)

---

## 2. Turn Behavior Analysis
We classified the relationship between consecutive user turns into four behavioral quadrants based on **Literal Similarity** (Levenshtein) and **Semantic Similarity** (BGE Reranker).

### Results
| Quadrant | Behavior Label | Count | Percentage | Interpretation |
|---|---|---|---|---|
| **I** | **Repeater** | **3,923** | **37.8%** | High literal + High semantic. Users often repeated the same issue, suggesting the bot's initial advice wasn't sufficient or they were venting. |
| **III** | **Jumper** | **3,138** | **30.2%** | Low literal + Low semantic. Users frequently shifted topics (e.g., from sleep issues to work stress), showing a natural, multi-threaded conversation flow. |
| **IV** | **Refiner** | **2,410** | **23.2%** | High literal + Low semantic. Users kept similar wording but changed the meaning (e.g., adding constraints like "but I don't have time"). |
| **II** | **Paraphraser** | **909** | **8.8%** | Low literal + High semantic. Users rephrased the same intent completely, likely trying to help the bot understand. |

### Key Insight
The high percentage of **Repeaters (37.8%)** is significant for mental health conversations. Unlike transactional domains where users move on after an answer, mental health users tend to **dwell on their feelings**, restating their distress even after receiving advice. This suggests chatbots in this domain need high empathy and patience, not just solution-giving.

---

## 3. Sentiment Gradient Analysis
Sentiment was tracked using **Twitter-RoBERTa-Latest** (3-class), providing a continuous score from -1 (Negative) to +1 (Positive).

### Overall Trends
| Trend Category | Count | Percentage | Description |
|---|---|---|---|
| **Improving** (Slope > 0) | **1,525** | **76.2%** | Most conversations ended better than they started. |
| **Stable** (Flat slope) | **22** | **1.1%** | Very few conversations had no emotional movement. |
| **Worsening** (Slope < 0) | **453** | **22.6%** | A significant minority left feeling worse or unresolved. |

### Volatility & Recovery
- **Average Sentiment Delta**: +0.54 (Strong improvement)
- **Recovery Rate**: **66.8%** of users who experienced a sentiment drop (e.g., frustration) eventually recovered to a positive state by the end.
- **Max Drop**: The average "worst moment" in a conversation had a gradient of -0.42.

### Sentiment Progression Example
> **User**: "I feel really drained..." (-0.87)
> **User**: "I tried that but..." (-0.45)
> **User**: "That actually sounds doable." (+0.56)

### Key Insight
The **76% improvement rate** validates the "supportive" nature of the generated conversations. However, the **22% worsening** rate is a critical area for improvement—likely cases where the bot gave generic advice ("just breathe") that frustrated the user.

---

## 4. Conclusion & Recommendations
1. **Empathy Loops**: Given the high "Repeater" behavior, the bot should be trained to acknowledge and validaterepetitive statements rather than just repeating the same advice.
2. **Handle Topic Jumps**: The 30% "Jumper" rate means the bot must be good at context switching (e.g., connecting sleep issues to the new topic of anxiety).
3. **Sentiment Monitoring**: Real-time sentiment tracking could trigger a handover or a strategy shift when the gradient slope becomes negative for >2 turns.

---
*Analysis generated on 2026-01-25 based on `mental_health_conversations.json`.*
