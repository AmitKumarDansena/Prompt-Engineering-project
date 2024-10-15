##from transformers import pipeline

# Load sentiment-analysis pipeline using a pre-trained model (DistilBERT in this case)
#classifier = pipeline("sentiment-analysis")

# Define input text (e.g., a movie review)
#input_text = "The movie was great, with fantastic acting and stunning visuals, but the plot was too predictable."

# Define different prompt styles
#prompts = {
#    'Declarative Prompt': f"The sentiment of the following review is: {input_text}",
#    'Question-Based Prompt': f"What is the sentiment of the following review? {input_text}",
#    'Instruction-Based Prompt': f"Classify the sentiment of the following review as either positive or negative: {input_text}"
#}

# Function to classify the sentiment based on different prompts
#def classify_sentiment(prompt_type, prompt):
#    print(f"\n{prompt_type}:")
#    print(f"Prompt: {prompt}")
#    result = classifier(prompt)
#    print(f"Classification Result: {result[0]['label']} with confidence score of {result[0]['score']:.2f}")

# Classify sentiment for each prompt style
#for prompt_type, prompt in prompts.items():
#    classify_sentiment(prompt_type, prompt)



from transformers import pipeline

# Specify the model explicitly
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Define input text (e.g., a movie review)
input_text = "The movie was great, with fantastic acting and stunning visuals, but the plot was too predictable."

# Define different prompt styles
prompts = {
    'Declarative Prompt': f"The sentiment of the following review is: {input_text}",
    'Question-Based Prompt': f"What is the sentiment of the following review? {input_text}",
    'Instruction-Based Prompt': f"Classify the sentiment of the following review as either positive or negative: {input_text}"
}

# Function to classify the sentiment based on different prompts
def classify_sentiment(prompt_type, prompt):
    print(f"\n{prompt_type}:")
    print(f"Prompt: {prompt}")
    result = classifier(prompt)
    print(f"Classification Result: {result[0]['label']} with confidence score of {result[0]['score']:.2f}")

# Classify sentiment for each prompt style
for prompt_type, prompt in prompts.items():
    classify_sentiment(prompt_type, prompt)


import os

# Disable symlink warning
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from transformers import pipeline

# Specify the model explicitly
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Define input text
input_text = "The movie was great, with fantastic acting and stunning visuals, but the plot was too predictable."

# Define prompts and classify sentiment (as shown in previous code)
