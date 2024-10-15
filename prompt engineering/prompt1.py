from transformers import pipeline
from datasets import load_dataset
import pandas as pd

# Load a pre-trained text classification model
classifier = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')

# Load a dataset (IMDb reviews for text classification)
dataset = load_dataset('imdb', split='test[:500]')

# Sample text from the IMDb dataset
sample_text = dataset[0]['text']

# Declarative prompt
prompt_1 = f"The sentiment of the following text is: {sample_text}"

# Question-based prompt
prompt_2 = f"What is the sentiment of the following review? {sample_text}"

# Instruction-based prompt
prompt_3 = f"Classify the sentiment of this movie review: {sample_text}"

prompts = [prompt_1, prompt_2, prompt_3]

def classify_text(prompt):
    return classifier(prompt)

results = {}

for i, prompt in enumerate(prompts):
    print(f"\nPrompt {i+1}: {prompt[:100]}...")
    result = classify_text(prompt)
    print(f"Classification result: {result}")
    results[f'Prompt {i+1}'] = result

# Create a function to apply a prompt to the dataset
def apply_prompts(dataset, prompts):
    results = []
    for sample in dataset:
        text = sample['text']
        for prompt in prompts:
            # Generate full prompt with the sample text
            prompt_text = prompt.format(text=text)
            result = classifier(prompt_text)
            results.append({'prompt': prompt, 'text': text, 'result': result})
    return pd.DataFrame(results)

# Generate multiple samples
subset = dataset.select(range(10))  # Use a subset of 10 reviews for testing
prompt_templates = [
    "The sentiment of the following text is: {text}",
    "What is the sentiment of the following review? {text}",
    "Classify the sentiment of this movie review: {text}"
]

results_df = apply_prompts(subset, prompt_templates)

from sklearn.metrics import accuracy_score, f1_score

# Convert predictions to binary labels (positive/negative sentiment)
def convert_labels(predictions):
    return [1 if p['label'] == 'POSITIVE' else 0 for p in predictions]

# Generate ground truth labels (since it's the IMDb dataset, labels are already provided)
true_labels = [sample['label'] for sample in subset]

# Evaluate prompt performance for each type of prompt
for prompt in prompt_templates:
    prompt_results = results_df[results_df['prompt'] == prompt]['result']
    pred_labels = convert_labels(prompt_results)
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    
    print(f"Prompt: {prompt}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}\n")

from transformers import pipeline

# Load a summarization model
summarizer = pipeline("summarization")

# Define different summarization prompts
summarization_prompts = [
    "Summarize the following text in a concise paragraph: {text}",
    "What is the main idea of this article? {text}",
    "Provide a brief summary of the following content: {text}"
]

# Test summarization on a sample text
sample_text = "The rise of AI has transformed numerous industries. Its impact on healthcare, finance, and education is profound."

for prompt in summarization_prompts:
    prompt_text = prompt.format(text=sample_text)
    summary = summarizer(prompt_text)
    print(f"Prompt: {prompt}\nSummary: {summary}\n")

import matplotlib.pyplot as plt

# Assuming 'accuracy_scores' is a list of accuracy values for each prompt
prompts = ['Declarative', 'Question-based', 'Instruction-based']
accuracy_scores = [0.87, 0.85, 0.89]

plt.bar(prompts, accuracy_scores, color=['blue', 'green', 'red'])
plt.xlabel('Prompt Type')
plt.ylabel('Accuracy')
plt.title('Accuracy Across Different Prompt Types')
plt.show()
