import re
import random
import json
import os
from sklearn.model_selection import train_test_split

QUESTION_TEMPLATES = [
    "What is {}?",
    "Calculate {}.",
    "Compute {}.",
    "Find the value of {}.",
    "Evaluate {}.",
]

# Position-aware connectors for step descriptions
_CONNECTORS = {
    'only':   ["",         "We have:",    "Calculating:"],
    'first':  ["First,",   "To start,",   "Begin by computing"],
    'middle': ["Next,",    "Then,",       "After that,",  "Continuing,"],
    'last':   ["Finally,", "Lastly,",     "To finish,"],
}

def _step_connector(step_idx, total_steps):
    if total_steps == 1:
        return random.choice(_CONNECTORS['only'])
    if step_idx == 0:
        return random.choice(_CONNECTORS['first'])
    if step_idx == total_steps - 1:
        return random.choice(_CONNECTORS['last'])
    return random.choice(_CONNECTORS['middle'])


def generate_arithmetic_chain(num_steps):
    """Generate a unique multi-step arithmetic problem with varied surface forms."""
    ops = ['+', '-', '*']

    # Pre-select ops and operands so we can compute a dedup key before building strings
    chosen_ops = [random.choice(ops) for _ in range(num_steps)]
    chosen_operands = [
        random.randint(2, 9) if op == '*' else random.randint(1, 50)
        for op in chosen_ops
    ]
    start_val = random.randint(1, 200)
    dedup_key = (start_val, tuple(chosen_ops), tuple(chosen_operands))

    steps = []
    display_ops = []
    current_val = start_val

    for op, operand in zip(chosen_ops, chosen_operands):
        if op == '+':
            new_val = current_val + operand
            step_str = f"{current_val} + {operand} = {new_val}"
            display_ops.append('+')
        elif op == '-':
            new_val = current_val - operand
            step_str = f"{current_val} - {operand} = {new_val}"
            display_ops.append('-')
        else:  # '*'
            new_val = current_val * operand
            step_str = f"{current_val} × {operand} = {new_val}"
            display_ops.append('×')

        steps.append(step_str)
        current_val = new_val

    # Build expression with explicit left-to-right parenthesization.
    # Without this, "5 + 3 × 4" implies order-of-operations (= 17) but the
    # chain evaluates left-to-right (= 32), creating inconsistent labels.
    expr = str(start_val)
    for i, (op, operand) in enumerate(zip(display_ops, chosen_operands)):
        expr = f"({expr}) {op} {operand}" if i > 0 else f"{expr} {op} {operand}"

    problem = random.choice(QUESTION_TEMPLATES).format(expr)

    reasoning_lines = []
    for i, step_str in enumerate(steps):
        connector = _step_connector(i, num_steps)
        line = f"Step {i+1}: {connector} {step_str}" if connector else f"Step {i+1}: {step_str}"
        reasoning_lines.append(line)

    return dedup_key, {
        "problem": problem,
        "reasoning": "\n".join(reasoning_lines),
        "answer": str(current_val),
        "num_steps": num_steps,
        "type": "arithmetic"
    }

def generate_arithmetic_dataset(target_per_level=7500):
    """Generate 7500 *unique computations* per difficulty level.
    Dedup key is (start_val, ops, operands) — not problem string — so the same
    arithmetic never appears in both train and test even with different phrasing."""
    dataset = []
    for num_steps in [1, 2, 3, 4]:
        seen_keys = set()
        while len(seen_keys) < target_per_level:
            key, example = generate_arithmetic_chain(num_steps)
            if key not in seen_keys:
                seen_keys.add(key)
                dataset.append(example)

    random.shuffle(dataset)

    # Save
    with open("arithmetic_data.jsonl", "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    print(f"Generated {len(dataset)} arithmetic examples")

def split_jsonl_dataset(input_path, train_path, test_path, test_size=0.2, seed=42):
    """
    Reads a .jsonl file, splits the data, and saves as two .json files.
    """
    data = []
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Couldn't find the file: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line: 
                data.append(json.loads(line))

    train, test = train_test_split(data, test_size=test_size, random_state=seed)

    for dataset, path in [(train, train_path), (test, test_path)]:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=4) 

class ArithmeticDataset:
    def __init__(self, dataset, tokenizer, max_length=512, max_samples=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples if max_samples is not None else len(dataset)

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        item = self.dataset[idx]
        prefix = "<problem>" + item['problem'] + "</problem>\n"
        text = prefix + "<reasoning>\n" + item['reasoning'] + "</reasoning>\n<answer>" + item['answer'] + "</answer>"

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        prefix_len = len(self.tokenizer(prefix, add_special_tokens=False)['input_ids'])
        labels = encoded['input_ids'].clone()
        labels[0, :prefix_len] = -100  # Mask the problem prefix
        labels[0, encoded['attention_mask'][0] == 0] = -100  # Mask padding

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'labels': labels.squeeze(0)
        }

    def create_tokenizer_txt(self, output_path="arithmetic_tokenizer_corpus.txt"):
        with open(output_path, "w", encoding="utf-8") as f:
            for item in self.dataset:
                prefix = "<problem>" + item['problem'] + "</problem>\n"
                text = prefix + "<reasoning>\n" + item['reasoning'] + "</reasoning>\n<answer>" + item['answer'] + "</answer>"
                f.write(text + "\n")


class GSM8KDataset:
    def __init__(self, dataset, tokenizer, max_length=512, max_samples=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples if max_samples is not None else len(dataset['train'])
    def __len__(self):
        return self.max_samples
    def __getitem__(self, idx):
        question = self.dataset['train'][idx]['question']
        answer = self.dataset['train'][idx]['answer']
        answer_without_annotations = re.sub(r'<<.*?>>', '', answer)
        reasoning, answer = answer_without_annotations.split("####")
        prefix = "<problem>" + question + "</problem>\n"
        text = prefix + "<reasoning>\n" + reasoning + "</reasoning>\n<answer>" + answer + "</answer>"
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        prefix_len = len(self.tokenizer(prefix, add_special_tokens=False)['input_ids'])
        labels = encoded['input_ids'].clone()
        labels[0, :prefix_len] = -100  # Mask the problem prefix
        labels[0, encoded['attention_mask'][0] == 0] = -100  # Mask padding

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'labels': labels.squeeze(0)
        }
    def create_tokenizer_txt(self, output_path="tokenizer_corpus.txt"):
        with open(output_path, "w", encoding="utf-8") as f:
            for training_data in self.dataset['train']:
                question = training_data['question']
                answer = training_data['answer']
                answer_without_annotations = re.sub(r'<<.*?>>', '', answer)
                reasoning, answer = answer_without_annotations.split("####")
                prefix = "<problem>" + question + "</problem>\n"
                text = prefix + "<reasoning>\n" + reasoning + "</reasoning>\n<answer>" + answer + "</answer>"
                f.write(text + "\n")  