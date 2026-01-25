"""
Mental Health Q&A Conversation Generator

Generates simulated multi-turn conversations between users and a mental health 
support chatbot. These are SYNTHETIC conversations for research purposes only.

IMPORTANT ETHICAL NOTE:
- These are simulated conversations, NOT real user data
- The focus is on general wellness, coping strategies, and information-seeking
- Conversations do NOT simulate crisis situations or self-harm content
- Purpose: Research on chatbot interaction patterns and sentiment analysis

Model: nvidia/nemotron-3-nano-30b-a3b:free
Domain: Mental Health & Wellness Support
Turns: 5-25 per conversation
Target: 4000 conversations
"""

import json
import os
import time
import uuid
import random
import requests
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from project root .env
# Searches current dir, then parent, then grandparent
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded .env from: {env_path}")
else:
    load_dotenv()  # Fallback to current dir

# OpenRouter API configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
# Override model for this experiment
MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"

# Generation settings
BATCH_SIZE = 10  # Smaller batches for longer conversations
NUM_CONVERSATIONS = int(os.environ.get("NUM_CONVERSATIONS", "4000"))
MIN_TURNS = 5
MAX_TURNS = 15

def generate_conversations_batch(batch_size: int = 10) -> list[dict]:
    """
    Use OpenRouter LLM to generate a batch of realistic mental health Q&A conversations.
    Focus is on supportive interactions, NOT crisis content.
    Returns a list of conversation objects.
    """
    
    # Randomly vary turn count for this batch
    turn_range_low = random.choice([5, 6, 7, 8])
    turn_range_high = random.choice([12, 15, 18, 22, 25])
    
    prompt = f"""Generate {batch_size} realistic mental health support chatbot conversations.

CONTEXT: These are conversations between adults seeking mental health information/support 
and an AI wellness assistant. Focus on EVERYDAY challenges, NOT crisis situations.

CRITICAL - CONVERSATION FLOW PATTERNS (distribute evenly):
Generate approximately 2-3 conversations of EACH type per batch:

1. REPEATER PATTERN (~25%): User restates their concern because the bot's answer was too generic
   - Turn 2 rephrases Turn 1 with more specific details
   - Example: Turn 1: "I've been feeling really anxious lately" → Turn 2: "I mean specifically, I get anxious before work meetings to the point where I can't focus"
   - Shows user trying to get more personalized help

2. PARAPHRASER PATTERN (~25%): Same concern expressed completely differently  
   - Turn 2 describes the same issue using different framing
   - Example: Turn 1: "I can't stop worrying about everything" → Turn 2: "It's like my brain won't turn off the overthinking loop"
   - User trying metaphors or different words to be understood

3. JUMPER PATTERN (~25%): Shifts to a related but different topic
   - Turn 2 moves to a connected concern
   - Example: Turn 1: "I've been having trouble sleeping" → Turn 2: "Also, I've been stress eating a lot lately"
   - User has multiple things they want to discuss

4. DEEP DIVER PATTERN (~25%): Natural progression deeper into the same topic
   - Turn 2 asks follow-up using similar language
   - Example: Turn 1: "What are some ways to manage work stress?" → Turn 2: "That breathing technique sounds helpful - how do I remember to do it during stressful moments?"
   - User engaged and seeking more detail

CONVERSATION LENGTH: Each conversation should have {turn_range_low}-{turn_range_high} user messages.

TOPIC AREAS (mix across conversations - focus on COMMON, NON-CRISIS issues):
- Work-life balance, burnout prevention
- General anxiety management (NOT panic attacks or severe anxiety disorders)
- Sleep hygiene and relaxation techniques
- Stress from relationships, family dynamics
- Building healthy habits, motivation
- Loneliness, making social connections
- Managing negative self-talk
- Dealing with change (job, moving, life transitions)
- Setting boundaries with others
- Mindfulness and self-care practices
- Seasonal mood changes
- Time management and feeling overwhelmed
- Perfectionism and productivity anxiety

QUESTION/STATEMENT TYPES:
- INFORMATION: "What are some techniques for managing stress?"
- PERSONAL: "I've noticed I feel really drained after work calls"
- SEEKING VALIDATION: "Is it normal to feel overwhelmed when starting a new job?"
- COPING: "How do I stop myself from doom-scrolling when I'm stressed?"
- PRACTICAL: "Can you suggest a simple morning routine for better mental health?"
- REFLECTION: "I realized I never take breaks during the day"
- FRUSTRATION: "I've tried meditation apps but they don't work for me"

FRUSTRATION SOURCES (for Repeater patterns):
- Bot gave overly generic advice ("just take deep breaths")
- Bot didn't acknowledge the specific situation
- Bot's suggestions weren't realistic for user's lifestyle
- Bot repeated what user already said they tried

EMOTIONAL PROGRESSION EXAMPLES:
- Start neutral → become more open → express gratitude
- Start frustrated → feel heard → become engaged
- Start hopeful → get discouraged by generic advice → re-engage with specifics
- Start venting → gradually shift to solution-seeking

IMPORTANT GUIDELINES:
- Each user message should be 15-40 words with authentic emotional content
- Include natural speech patterns (filler words, trailing off, self-correction)
- Show realistic emotional progression across turns
- NO crisis content, self-harm mentions, or severe mental illness symptoms
- Focus on common, relatable challenges that most adults experience
- Include moments where user acknowledges helpful advice

Output ONLY a JSON array of {batch_size} conversations:
[
  ["first message about concern A", "rephrased with more detail", "follow-up question", "response to advice", "closing thought"],
  ["first message about concern B", "same issue different words", "practical question", "expressing what worked", "new related concern"],
  ...
]

Each inner array represents one conversation's user messages in order (no bot responses)."""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Mental Health Conversation Generator - Research"
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.75,  # Slightly higher for more natural variation
        # "max_tokens": 16000,  # More tokens for longer conversations
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=180)
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
                if not isinstance(user_messages, list) or len(user_messages) < MIN_TURNS:
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
                    "num_turns": len(turns),
                    "domain": "mental_health_wellness",
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

def print_stats(conversations: list):
    """Print statistics about generated conversations."""
    if not conversations:
        return
    
    turn_counts = [c.get("num_turns", len(c.get("turns", []))) for c in conversations]
    avg_turns = sum(turn_counts) / len(turn_counts)
    min_turns = min(turn_counts)
    max_turns = max(turn_counts)
    
    print(f"\nConversation Statistics:")
    print(f"  Average turns: {avg_turns:.1f}")
    print(f"  Min turns: {min_turns}")
    print(f"  Max turns: {max_turns}")
    print(f"  Turn distribution:")
    
    # Bucket distribution
    buckets = {"5-8": 0, "9-12": 0, "13-17": 0, "18-25": 0}
    for tc in turn_counts:
        if tc <= 8:
            buckets["5-8"] += 1
        elif tc <= 12:
            buckets["9-12"] += 1
        elif tc <= 17:
            buckets["13-17"] += 1
        else:
            buckets["18-25"] += 1
    
    for bucket, count in buckets.items():
        pct = 100 * count / len(turn_counts)
        print(f"    {bucket} turns: {count} ({pct:.1f}%)")

def generate_dummy_conversations(
    num_conversations: int = 4000, 
    batch_size: int = 10, 
    delay_per_call: float = 1.0,  # Slightly longer delay for rate limiting
    output_file: str = "mental_health_conversations.json"
):
    """
    Generate mental health Q&A conversations using LLM via OpenRouter API.
    Saves incrementally after each batch to preserve progress.
    """
    
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set!")
    
    # Load existing conversations to resume from where we left off
    all_conversations = load_existing_conversations(output_file)
    starting_count = len(all_conversations)
    
    print("=" * 60)
    print("Mental Health Q&A Conversation Generator")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Domain: Mental Health & Wellness (non-crisis)")
    print(f"Turn range: {MIN_TURNS}-{MAX_TURNS}")
    print(f"Target: {num_conversations} conversations")
    print(f"Batch size: {batch_size} per API call")
    print(f"Existing: {starting_count} conversations")
    print(f"Remaining: {max(0, num_conversations - starting_count)}")
    print("-" * 60)
    
    if starting_count >= num_conversations:
        print("Already have enough conversations!")
        print_stats(all_conversations[:num_conversations])
        return all_conversations[:num_conversations]
    
    api_call_count = 0
    failed_count = 0
    
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
            avg_turns = sum(len(c["turns"]) for c in batch) / len(batch)
            print(f"  Got {len(batch)} conversations (avg {avg_turns:.1f} turns), saved to {output_file}")
            failed_count = 0  # Reset failure counter on success
        else:
            failed_count += 1
            print(f"  Batch failed (attempt {failed_count}), retrying...")
            if failed_count >= 3:
                print("  Too many failures, waiting 30 seconds...")
                time.sleep(30)
                failed_count = 0
        
        # Rate limiting delay
        if len(all_conversations) < num_conversations:
            time.sleep(delay_per_call)
    
    # Trim to exact count if we got more
    all_conversations = all_conversations[:num_conversations]
    save_conversations(all_conversations, output_file)
    
    print("-" * 60)
    print("Generation complete!")
    print(f"  Total conversations: {len(all_conversations)}")
    print(f"  New conversations generated: {len(all_conversations) - starting_count}")
    print(f"  Total API calls: {api_call_count}")
    
    print_stats(all_conversations)
    
    return all_conversations

if __name__ == "__main__":
    # Save output to the same directory as this script
    script_dir = Path(__file__).resolve().parent
    output_file = str(script_dir / "mental_health_conversations.json")
    
    # Generate conversations using configured settings
    data = generate_dummy_conversations(
        num_conversations=NUM_CONVERSATIONS, 
        batch_size=BATCH_SIZE, 
        delay_per_call=1.0,
        output_file=output_file
    )
    
    print(f"\nFinal count: {len(data)} conversations in {output_file}")
