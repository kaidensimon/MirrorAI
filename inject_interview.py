"""
inject_interview.py
-------------------
Takes the structured dialogue saved by interview.py (interview_output.json)
and injects it into a GenerativeAgent's memory stream as observations,
exactly as the paper does when bootstrapping an agent from a life-story interview.

Each Q&A pair becomes one observation node. After all observations are added,
the agent reflects on a set of anchor topics to synthesize higher-level memories
from the raw interview content.

Usage:
    python inject_interview.py

Outputs the agent to: storage/<agent_name>/
"""

import json
import sys
import os

# Add the genagents repo root so both the genagents package and
# simulation_engine (its sibling) are importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "genagents"))

from genagents.genagents import GenerativeAgent


# ---------------------------------------------------------------------------
# Config — edit these before running
# ---------------------------------------------------------------------------

INTERVIEW_OUTPUT  = "interview_output.json"   # produced by interview.py
AGENT_SAVE_DIR    = "storage/agent_001"       # where to persist the agent
PARTICIPANT_NAME  = "John Snow"             # name used in memory content

# Topics the agent will reflect on after the interview is injected.
# These should match the broad themes covered by your questions.
REFLECTION_ANCHORS = [
    "childhood and upbringing",
    "education and career",
    "relationships and family",
    "personal values and beliefs",
    "major life events",
]

# ---------------------------------------------------------------------------


def format_observation(question: str, answer: str, name: str) -> str:
    """
    Formats a Q&A pair into the observation content string the memory stream
    expects. Matches the format found in the paper's nodes.json:
      'Interviewer: <question>\n\n<Name>: <answer>\n\n'
    """
    return f"Interviewer: {question}\n\n{name}: {answer}\n\n"


def inject(interview_path: str, save_dir: str, participant_name: str):
    with open(interview_path, encoding="utf-8") as f:
        data = json.load(f)

    dialogue = data["dialogue"]

    # Initialize a blank agent. If you want to load an existing agent instead,
    # pass agent_folder=save_dir to GenerativeAgent().
    agent = GenerativeAgent()

    # Set the participant's name in scratch so the agent knows who it is.
    agent.update_scratch({
        "first_name": participant_name.split()[0],
        "last_name":  participant_name.split()[-1] if " " in participant_name else "",
    })

    print(f"Injecting {len(dialogue)} interview turns into memory stream...")

    # time_step starts at 1; each Q&A pair advances it by 1.
    # The paper uses time_step to order memories chronologically.
    for time_step, turn in enumerate(dialogue, start=1):
        question = turn["interviewer_question"]
        answer   = turn["user_answer"]

        if not answer.strip():
            print(f"  [{time_step}] Skipping empty answer for: {question[:60]}...")
            continue

        content = format_observation(question, answer, participant_name)
        print(f"  [{time_step}] Remembering: {question[:60]}...")

        # remember() scores importance via LLM and embeds the content.
        agent.remember(content, time_step=time_step)

    # Reflect on anchor topics to generate higher-level memories that
    # synthesize patterns across the raw observations.
    next_step = len(dialogue) + 1
    print(f"\nReflecting on {len(REFLECTION_ANCHORS)} anchor topics...")
    for anchor in REFLECTION_ANCHORS:
        print(f"  Reflecting on: '{anchor}'")
        agent.reflect(anchor, time_step=next_step)
        next_step += 1

    # Persist the agent to disk.
    os.makedirs(save_dir, exist_ok=True)
    agent.save(save_dir)
    print(f"\nAgent saved to: {save_dir}")
    print(f"  Memory nodes: {len(agent.memory_stream.seq_nodes)}")


if __name__ == "__main__":
    inject(INTERVIEW_OUTPUT, AGENT_SAVE_DIR, PARTICIPANT_NAME)
