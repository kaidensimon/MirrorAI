import json
from datasets import Dataset

# Your raw data
import json

# Check if file is readable at all
with open("D:\\WeekendSideProjects\\mirrorai\\interview_output.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(type(raw_data))        # should be <class 'dict'>
print(raw_data.keys())       # should show dict_keys(['dialogue'])
print(len(raw_data["dialogue"]))  # should show 100+
# Format into chat-style training examples
def format_dialogue(dialogue):
    examples = []
    for turn in dialogue:
        question = turn["interviewer_question"]
        answer = turn["user_answer"]
        
        # Format as a conversation
        text = f"""<|im_start|>system
You are a wealthy, ruthless private equity executive from Greenwich, Connecticut. You are cold, calculating, and view everything through the lens of power and dominance. Respond exactly as this person would.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""
        
        examples.append({"text": text})
    return examples

formatted = format_dialogue(raw_data["dialogue"])
dataset = Dataset.from_list(formatted)
print(f"Total examples: {len(dataset)}")
print(dataset[0]["text"])  # verify formatting