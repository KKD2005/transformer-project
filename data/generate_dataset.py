from datasets import load_dataset
import re

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