# Multi-Turn Dialogue and Sentiment Gradient Analysis

## Project Overview

This project focuses on simulating and analyzing multi-turn conversations for a chatbot study in the **tax and accounting research** domain. The goal is to generate realistic conversations between CPAs and a Q&A chatbot, then analyze user behavior patterns and sentiment progression.

---

## Tasks Completed

### Task 1: Generate Dummy Data ✅

**Script:** `generate_conversations.py`

**Purpose:** Generate 2000 realistic multi-turn tax/accounting Q&A chatbot conversations.

**Key Features:**
- Uses OpenRouter API with model `xiaomi/mimo-v2-flash:free` (configurable via `.env`)
- Batch generation (15 conversations per API call)
- Incremental saving to prevent data loss
- Resume capability from existing data

**Configuration (`.env`):**
```
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=xiaomi/mimo-v2-flash:free
NUM_CONVERSATIONS=2000
```

#### LLM Prompt Used

```
Generate {batch_size} realistic tax and accounting research Q&A chatbot conversations.

CRITICAL - CONVERSATION FLOW PATTERNS (distribute evenly across batch):
Generate approximately 4 conversations of EACH type per batch:

1. REPEATER PATTERN (~25%): User restates almost the same question because bot didn't answer well
   - Turn 2 is nearly identical to Turn 1, just slightly reworded
   - Example: Turn 1: "What's the Section 179 limit for 2024?" → Turn 2: "I asked about the Section 179 deduction limits for this year"
   - Shows user frustration that bot didn't understand

2. PARAPHRASER PATTERN (~25%): Same intent expressed completely differently
   - Turn 2 asks the same thing but with totally different words/structure
   - Example: Turn 1: "What's the Section 179 limit?" → Turn 2: "How much equipment can I write off immediately under the first-year deduction rules?"
   - User trying different approach to get answer

3. JUMPER PATTERN (~25%): Complete topic change, unrelated questions
   - Turn 2 is about a completely different tax/accounting topic
   - Example: Turn 1: "What's the Section 179 limit?" → Turn 2: "By the way, what are the rules for deducting home office expenses?"
   - User switching to new question entirely

4. DEEP DIVER PATTERN (~25%): Natural continuation with similar vocabulary
   - Turn 2 builds on Turn 1 using similar terms, going deeper into the topic
   - Example: Turn 1: "What's the Section 179 limit?" → Turn 2: "Does that Section 179 limit apply to used equipment, or only new purchases?"
   - Natural follow-up that references the same concepts

Requirements for EACH conversation:
- Each conversation should have 2-20 user messages (vary the count)
- FIRST user message: Direct question about a tax or accounting topic
- DO NOT include self-identification like "I'm a CPA" - just ask directly
- Apply the conversation flow pattern consistently throughout each conversation

QUESTION TYPES (mix across conversations):
- DIRECT: "What is the standard mileage rate for 2024?"
- HYPOTHETICAL: "If a client sells rental property at a loss after taking depreciation..."
- FACT: "What are the IRS wash sale rules?"
- DOCUMENT_GENERATION: "Help me draft a memo about S-Corp election"
- KEYWORD: "Section 179 limits" (short queries)
- KEYWORD_PLUS: "Section 179 for used equipment S-corp"
- ADVISORY: "Best approach to minimize capital gains?"

FRUSTRATION SOURCES (for Repeater patterns especially):
- Bot gave incorrect/outdated information
- Bot's answer was too vague or didn't address the specific question
- Bot asked for information already provided
- Bot gave contradictory answers

Topics: tax deductions, depreciation, capital gains, self-employment tax, estimated taxes, GAAP, revenue recognition, payroll taxes, 401k/IRA, home office, vehicle expenses, cryptocurrency, rental income, etc.

IMPORTANT: Each user message should be at least 15 words with specific details.
```

**Output Format:**
```json
[
  {
    "conversation_id": "uuid",
    "turns": [
      {"turn": 1, "speaker": "user", "text": "..."},
      {"turn": 2, "speaker": "user", "text": "..."},
      ...
    ]
  }
]
```

**Output File:** `dummy_conversations.json`

---

### Task 2: Turn-by-Turn Behavior Analysis ✅

**Script:** `turn_analyzer.py`

**Purpose:** Analyze the relationship between consecutive turns (Turn 1 vs 2, Turn 2 vs 3) to classify user behavior.

**Models Used:**
- **Levenshtein Distance:** Measures literal/surface text similarity (0-1)
- **BGE Reranker v2 m3:** Cross-encoder for semantic relevance scoring (local GPU)

#### User Behavior Matrix (4 Quadrants)

|  | **High Literal** | **Low Literal** |
|---|---|---|
| **High Semantic** | **I: Repeater** (same words, same meaning) | **II: Paraphraser** (different words, same meaning) |
| **Low Semantic** | **IV: Refiner** (same words, different meaning) | **III: Jumper** (different words, different topic) |

**Quadrant Definitions:**
- **Quadrant I - Repeater:** User restates nearly the same question (bot failed to respond properly)
- **Quadrant II - Paraphraser:** User tries different wording to get the same answer (bot didn't understand)
- **Quadrant III - Jumper:** User switches to completely different topic
- **Quadrant IV - Refiner:** User uses similar vocabulary but changes meaning (e.g., negation, correction)

**Threshold Methodology:**
- Uses **median-based thresholds** computed from actual data distribution
- Ensures natural ~25% distribution in each quadrant

**Sample Results (2000 conversations):**
```
Behavior Distribution:
----------------------------------------
Quadrant I  (Repeater):    1278 ( 32.0%)
Quadrant II (Paraphraser):  721 ( 18.0%)
Quadrant III (Jumper):     1273 ( 31.8%)
Quadrant IV (Refiner):      726 ( 18.2%)
```

**Output File:** `turn_analysis_results.json`

**Output Format:**
```json
{
  "conversation_id": "...",
  "num_turns": 5,
  "turn_analyses": [
    {
      "pair": [1, 2],
      "levenshtein_similarity": 0.2845,
      "reranker_score_raw": 0.5123,
      "quadrant": "II",
      "behavior_label": "Paraphraser",
      "turn_1_text": "...",
      "turn_2_text": "..."
    },
    ...
  ]
}
```

---

### Task 3: Sentiment Gradient Analysis 🔄 (In Progress)

**Script:** `sentiment_analyzer.py`

**Purpose:** Analyze sentiment progression across conversation turns.

**Model Used:**
- `distilbert-base-uncased-finetuned-sst-2-english` (66M params, binary sentiment)

**Key Features:**
1. **Sentiment Score** per turn (-1 to +1, or 0-100 scale)
2. **EMA Smoothing** (alpha=0.3) to reduce noise
3. **Gradient Computation** (Δ between consecutive turns)
4. **Direction Classification:** +1 (improving), 0 (stable), -1 (worsening)
5. **Session-Level Features:**
   - Trend slope
   - Max drop / Max rise
   - Sentiment range
   - Recovery flag
   - First negative turn

**Output File:** `sentiment_analysis_results.json`

---

## Project Structure

```
Multi Turn Dialogue and Sentiment Gradient/
├── .env                           # API keys and configuration
├── project_requirements.md        # Original project requirements
├── PROJECT_DOCUMENTATION.md       # This file
├── dummy_conversations.json       # Generated conversation data
├── generate_conversations.py      # Task 1: Data generation
├── turn_analyzer.py               # Task 2: Behavior analysis
├── turn_analysis_results.json     # Task 2 output
├── sentiment_analyzer.py          # Task 3: Sentiment analysis
└── sentiment_analysis_results.json # Task 3 output
```

---

## Dependencies

```bash
pip install requests python-dotenv python-Levenshtein sentence-transformers transformers torch numpy
```

---

## How to Run

### Task 1: Generate Conversations
```bash
python generate_conversations.py
```

### Task 2: Analyze Turn Behavior
```bash
python turn_analyzer.py
```

### Task 3: Analyze Sentiment Gradient
```bash
python sentiment_analyzer.py
```

---

## Technical Notes

### Why Cross-Encoder (Reranker) vs Bi-Encoder (Vector Similarity)?

| Aspect | Bi-Encoder (Embeddings) | Cross-Encoder (Reranker) |
|--------|-------------------------|--------------------------|
| How it works | Encode texts separately → cosine similarity | Encode both texts together → relevance score |
| Accuracy | Good | **Better** for pairwise comparison |
| Speed | Fast (can pre-compute) | Slower |
| Best for | Large-scale retrieval | Fine-grained comparison ✅ |

For Task 2 with ~4000 pairs, cross-encoder provides more accurate semantic relationship detection.

### Handling Raw Reranker Scores

BGE reranker outputs raw logits (can be negative):
- **Positive** = more relevant
- **Negative** = less relevant
- **Zero** = neutral

We use **median-based thresholds** rather than fixed values to ensure balanced quadrant distribution regardless of score range.

---

## Key Insights

1. **Conversation Pattern Design:** The 4 flow patterns (Repeater, Paraphraser, Jumper, Deep Diver) were specifically designed to generate data that maps to the behavior quadrants.

2. **Threshold Sensitivity:** Using fixed thresholds (like 0.5) caused 99%+ classification into one quadrant. Median-based thresholds fixed this.

3. **Sigmoid Compression:** Initial use of sigmoid to normalize reranker scores compressed variance. Using raw scores preserved natural distribution.

---

## Date Created
January 11, 2026
