import json
import os
import time
import uuid
import random
import requests
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from project root .env
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded .env from: {env_path}")
else:
    load_dotenv()  # Fallback to current dir

# OpenRouter API configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
# Hardcode model for this experiment
MODEL = "xiaomi/mimo-v2-flash:free"

# Generation settings
BATCH_SIZE = 15  # Number of conversations per API call
NUM_CONVERSATIONS = int(os.environ.get("NUM_CONVERSATIONS", "2000"))

def generate_conversations_batch(batch_size: int = 15) -> list[dict]:
    """
    Use OpenRouter LLM to generate a batch of realistic multi-turn conversations.
    Focus is on user queries; bot responses are placeholders.
    Returns a list of conversation objects.
    """
    
    prompt = f"""Generate {batch_size} realistic tax and accounting research Q&A chatbot conversations.

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

Output ONLY a JSON array of {batch_size} conversations:
[
  ["Q1 about topic A", "nearly identical restatement of Q1", "frustrated follow-up"],
  ["Q1 about topic B", "same intent totally different words", "acknowledgment"],
  ["Q1 about topic C", "completely different topic D question", "new topic follow-up"],
  ["Q1 about topic E", "deeper question using same vocabulary", "even deeper follow-up"],
  ...
]

Each inner array represents one conversation's user messages in order."""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Conversation Generator"
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.70,
        "max_tokens": 10000,
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        
        # Parse the JSON array from the response
        start_idx = content.find("[")
        end_idx = content.rfind("]") + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = content[start_idx:end_idx]
            all_conversations = json.loads(json_str)
            
            # Build conversation objects with user messages and turn numbers
            conversations = []
            for user_messages in all_conversations:
                if not isinstance(user_messages, list) or len(user_messages) < 2:
                    continue
                    
                turns = []
                for i, user_msg in enumerate(user_messages):
                    turns.append({
                        "turn": i + 1,
                        "speaker": "user",
                        "text": str(user_msg)
                    })
                
                conversations.append({
                    "conversation_id": str(uuid.uuid4()),
                    "turns": turns
                })
            
            return conversations
        else:
            print(f"Failed to parse JSON from response: {content[:300]}")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        return []

def save_conversations(conversations: list, output_file: str):
    """Save conversations to JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)

def load_existing_conversations(output_file: str) -> list:
    """Load existing conversations from file if it exists."""
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []

def generate_dummy_conversations(num_conversations: int = 500, batch_size: int = 15, delay_per_call: float = 0.5, output_file: str = "dummy_conversations.json"):
    """
    Generate conversations using LLM via OpenRouter API in batches.
    Saves incrementally after each batch to preserve progress.
    """
    
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set!")
    
    # Load existing conversations to resume from where we left off
    all_conversations = load_existing_conversations(output_file)
    starting_count = len(all_conversations)
    
    print(f"Using model: {MODEL}")
    print(f"Target: {num_conversations} conversations, {batch_size} per API call")
    print(f"Existing conversations: {starting_count}")
    print(f"Remaining to generate: {max(0, num_conversations - starting_count)}")
    print("-" * 50)
    
    if starting_count >= num_conversations:
        print("Already have enough conversations!")
        return all_conversations[:num_conversations]
    
    api_call_count = 0
    
    while len(all_conversations) < num_conversations:
        remaining = num_conversations - len(all_conversations)
        current_batch_size = min(batch_size, remaining)
        
        api_call_count += 1
        print(f"API call #{api_call_count}: Generating {current_batch_size} conversations... (Total: {len(all_conversations)}/{num_conversations})")
        
        batch = generate_conversations_batch(current_batch_size)
        
        if batch:
            all_conversations.extend(batch)
            # Save after each successful batch
            save_conversations(all_conversations, output_file)
            print(f"  ✓ Got {len(batch)} conversations, saved to {output_file}")
        else:
            print(f"  ✗ Batch failed, retrying...")
        
        # Rate limiting delay
        if len(all_conversations) < num_conversations:
            time.sleep(delay_per_call)
    
    # Trim to exact count if we got more
    all_conversations = all_conversations[:num_conversations]
    save_conversations(all_conversations, output_file)
    
    print("-" * 50)
    print(f"Generation complete!")
    print(f"  Total conversations: {len(all_conversations)}")
    print(f"  New conversations generated: {len(all_conversations) - starting_count}")
    print(f"  Total API calls: {api_call_count}")
    
    return all_conversations

if __name__ == "__main__":
    # Save output to the same directory as this script
    script_dir = Path(__file__).resolve().parent
    output_file = str(script_dir / "dummy_conversations.json")
    
    # Generate conversations using configured settings
    data = generate_dummy_conversations(
        num_conversations=NUM_CONVERSATIONS, 
        batch_size=BATCH_SIZE, 
        delay_per_call=0.5,
        output_file=output_file
    )
    
    print(f"\n✅ Final count: {len(data)} conversations in {output_file}")

