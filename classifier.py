import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.

    Your prompt needs to:
      1. Describe the task and the four valid labels
      2. Show the labeled training examples so the LLM can learn the pattern
      3. Present the new description and ask for a classification

    The LLM should return a single label from VALID_LABELS (exactly as written)
    plus a brief explanation of its reasoning.
    """
    lines = []
    lines.append("""You are classifying podcast episodes by their format.
Classify the episode into exactly one of these four labels:
- interview: a conversation between a host and one named guest
- solo: a single host speaking from memory, experience, or opinion — no guests, no assembled external sources
- panel: multiple guests with roughly equal speaking time, debating or discussing together
- narrative: a story assembled from external sources — interviews, archives, reporting — with a clear story arc

Here are labeled examples:
""")

    for ex in labeled_examples:
        # Truncate description to 150 chars to keep prompt short
        short_desc = ex['description'][:150] + "..." if len(ex['description']) > 150 else ex['description']
        lines.append(f"Title: {ex['title']}")
        lines.append(f"Description: {short_desc}")
        lines.append(f"Label: {ex['label']}")
        lines.append("---")

    lines.append(f"""
Now classify this episode:

Title: (unknown)
Description: {description}

Respond in exactly this format and nothing else:
Label: <one of: interview, solo, panel, narrative>
Reasoning: <one sentence explaining why>
""")

    return "\n".join(lines)


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.

    Steps:
      1. Call build_few_shot_prompt() to construct the prompt
      2. Send it to the LLM via _client.chat.completions.create()
      3. Parse the response to extract a label and reasoning
      4. Validate the label — if it's not in VALID_LABELS, set it to "unknown"
      5. Return a dict with "label" and "reasoning" keys

    Handle the case where the LLM returns something unparseable gracefully —
    don't let a bad response crash the whole evaluation.
    """
    try:
        prompt = build_few_shot_prompt(labeled_examples, description)

        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )

        response_text = response.choices[0].message.content
        print(response_text)  # temporary — lets you see raw output

        label = "unknown"
        reasoning = response_text.strip()

        for line in response_text.strip().splitlines():
            line_lower = line.lower()
            if line_lower.startswith("label:"):
                raw_label = line.split(":", 1)[1].strip().lower()
                if raw_label in VALID_LABELS:
                    label = raw_label
            elif line_lower.startswith("reasoning:"):
                reasoning = line.split(":", 1)[1].strip()

        return {"label": label, "reasoning": reasoning}

    except Exception as e:
        return {"label": "unknown", "reasoning": f"Error during classification: {e}"}