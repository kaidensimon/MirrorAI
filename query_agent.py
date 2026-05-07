"""
query_agent.py
--------------
Loads a saved GenerativeAgent and lets you have a back-and-forth conversation
with it. The agent answers as the person whose interview was injected.

Usage:
    python query_agent.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "genagents"))

from genagents.genagents import GenerativeAgent

# ---------------------------------------------------------------------------
AGENT_SAVE_DIR   = "storage/agent_001"   # must match what inject_interview.py used
PARTICIPANT_NAME = "John Snow"           # must match what inject_interview.py used
# ---------------------------------------------------------------------------


def chat(agent: GenerativeAgent):
    print(f"\nAgent loaded: {agent.get_fullname()}")
    print("Type your questions. Press Ctrl+C to quit.\n")

    dialogue = []

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not question:
            continue

        dialogue.append(["Interviewer", question])

        # utterance() retrieves relevant memories, then generates a response
        # grounded in what the agent knows about themselves.
        response = agent.utterance(dialogue, context="")

        dialogue.append([PARTICIPANT_NAME, response])
        print(f"\n{PARTICIPANT_NAME}: {response}\n")


if __name__ == "__main__":
    agent = GenerativeAgent(agent_folder=AGENT_SAVE_DIR)
    chat(agent)
