
# Analysis Report: Tax & Accounting Support (Xiaomi Mimo V2 Flash)

## 1. Overview
This experiment analyzes **2000 multi-turn conversations** in the **Tax & Accounting** domain. The goal was to identify user frustration points and behavior patterns when dealing with complex regulatory queries.

- **Model used**: `xiaomi/mimo-v2-flash:free`
- **Domain**: Tax Regulations, Deductions, Compliance
- **Total Conversations**: 2000
- **Total Turns**: ~6,800 (Avg 3.4 turns/conv)

---

## 2. Turn Behavior Analysis
We classified the relationship between consecutive user turns into four behavioral quadrants based on **Literal Similarity** (Levenshtein) and **Semantic Similarity** (BGE Reranker).

### Results
| Quadrant | Behavior Label | Count | Percentage | Interpretation |
|---|---|---|---|---|
| **I** | **Repeater** | **1,523** | **42.1%** | High literal + High semantic. Users frequently repeated their tax questions almost verbatim, indicating the bot likely failed to provide a precise or confident answer. |
| **II** | **Paraphraser** | **834** | **23.1%** | Low literal + High semantic. Users rephrased their queries (e.g., "Section 179 limits" -> "Vehicle deduction rules"), trying to find the right keyword to unlock the bot's knowledge. |
| **III** | **Jumper** | **945** | **26.1%** | Low literal + Low semantic. Users moved to related sub-topics (e.g., from "deduction limits" to "what forms to file"). |
| **IV** | **Refiner** | **315** | **8.7%** | High literal + Low semantic. Users made minor tweaks to their constraints (e.g., changing "2023" to "2024"). |

### Key Insight
The extremely high **Repeater + Paraphraser rate (65.2%)** indicates significant user struggle. In technical domains like tax, users expect precise answers. If the bot gives a generic "it depends on your situation" response, users are forced to repeat or rephrase, leading to frustration.

---

## 3. Sentiment Gradient Analysis
Sentiment was tracked using **Twitter-RoBERTa-Latest** (3-class), providing a continuous score from -1 (Negative) to +1 (Positive).

### Overall Trends
| Trend Category | Count | Percentage | Description |
|---|---|---|---|
| **Improving** (Slope > 0) | **412** | **20.6%** | A minority of users ended satisfied. |
| **Stable** (Flat slope) | **702** | **35.1%** | A large portion remained neutral/unresolved. |
| **Worsening** (Slope < 0) | **886** | **44.3%** | **Critical Issue**: Nearly half of the conversations ended with worse sentiment than they started. |

### Volatility & Recovery
- **Average Sentiment Delta**: -0.19 (Net negative trend)
- **Recovery Rate**: Only **14.2%** of users who got frustrated actually recovered. Most stayed frustrated.
- **Max Drop**: Significant drops were observed when users had to repeat themselves for the 3rd time.

### Sentiment Progression Example
> **User**: "Can I deduct my SUV?" (0.07 - Neutral)
> **User**: "I need a yes or no answer." (-0.03 - Irritated)
> **User**: "Your previous answer was completely unhelpful." (-0.80 - Frustrated)

### Key Insight
The **44% worsening rate** is alarmingly high compared to the mental health domain (22%). This confirms that **generic AI responses fail in high-stakes, fact-based domains**. Users perceive "safe", hedged answers as unhelpful in strict regulatory contexts.

---

## 4. Model Validation (Twitter-RoBERTa vs LLM)
To validate the sentiment scores, we sampled **15 random conversations** and compared the Twitter-RoBERTa model's predictions against an **LLM-generated ground truth** (using `xiaomi/mimo-v2-flash`).

### Results
- **Accuracy**: **86.9%** (Turn-level category match)
- **Mean Absolute Error (MAE)**: **0.13** (Scale of 0 to 2)

### Disagreement Analysis
The primary source of disagreement was **Contextual Negativity vs. Emotional Negativity**.
- **Scenario**: User says "My rental property was **damaged** by a **storm**."
- **Twitter-RoBERTa**: Rated as **Negative (-0.71)** due to negative keywords.
- **LLM Ground Truth**: Rated as **Neutral** (0.0) as it is a factual statement context for a tax question.

**Conclusion**: The model is slightly over-sensitive to negative keywords in factual descriptions, but highly accurate in detecting user frustration (e.g., "You didn't answer").

---

## 5. Conclusion & Recommendations
1. **Precision over Safety**: The bot needs to be more direct. Instead of "Consult a tax professional", it should say "Section 179 limit for 2024 is $XX,XXX, subject to..."
2. **Detect Repetition**: If a user repeats a query (Quadrant I), the bot should explicitly acknowledge the failure ("I see I didn't answer your question about X directly. Here is the specific rule...").
3. **Fact-Checking**: The high number of "Refiners" suggests users are correcting the bot's assumptions. Integrating a RAG (Retrieval-Augmented Generation) system with official IRS documents is essential.

---
*Analysis generated on 2026-01-25 based on `dummy_conversations.json`.*
