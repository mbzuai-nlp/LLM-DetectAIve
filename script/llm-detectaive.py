import json
import random
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline



# Constants
MIN_WORDS = 50
MAX_WORDS = 500
SAMPLE_JSON_PATH = Path('samples.json')

# Mapping labels from model to human-readable text
TEXT_CLASS_MAPPING = {
    'LABEL_2': 'Machine-Generated',
    'LABEL_0': 'Human-Written',
    'LABEL_3': 'Machine-Written, Machine-Humanized',
    'LABEL_1': 'Human-Written, Machine-Polished'
}

# Load model and tokenizer from Hugging Face
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline('text-classification', model=model, tokenizer=tokenizer, truncation=True, max_length=512, top_k=4)

# Load the model from Hugging Face (use the correct model name or path)
classifier = load_model("raj-tomar001/LLM-DetectAIve_deberta-base")  # Adjust model name

# Load sample essays
with open(SAMPLE_JSON_PATH, 'r') as f:
    demo_essays = json.load(f)

# Function to classify text and return the most probable label
def classify_text(text):
    if not (MIN_WORDS <= len(text.split()) <= MAX_WORDS):
        return f"Error: The input text must be between {MIN_WORDS} and {MAX_WORDS} words."

    result = classifier(text)[0]  # Get top result
    labels = [TEXT_CLASS_MAPPING[x['label']] for x in result]  # Map to human-readable labels
    scores = [x['score'] for x in result]

    final_results = dict(zip(labels, scores))
    return max(final_results, key=final_results.get)  # Return highest probability label

# Function to generate a random text sample for testing
def generate_sample_text():
    index = random.choice(range(len(demo_essays)))
    return demo_essays[index][0], index

# Function to get the correct label based on index
def get_correct_label(index):
    if 0 <= index < 20:
        return "Human-Written"
    elif 20 <= index < 40:
        return "Machine-Generated"
    elif 40 <= index < 60:
        return "Human-Written, Machine-Polished"
    elif 60 <= index < 80:
        return "Machine-Written, Machine-Humanized"

def main():
    while True:
        print("\nSelect Mode:")
        print("1. Input Text Classification")
        print("2. Challenge Yourself")
        print("3. Exit")
        mode = input("Enter your choice (1/2/3): ")

        if mode == '1':
            # Input text classification mode
            text_input = input("\nEnter your text (50 to 500 words): ")
            word_count = len(text_input.split())
            
            if MIN_WORDS <= word_count <= MAX_WORDS:
                prediction = classify_text(text_input)
                print(f"Classification Result: {prediction}")
            else:
                print(f"Error: Your input contains {word_count} words. Please enter text with 50 to 500 words.")
        
        elif mode == '2':
            # Challenge Yourself mode
            sample_text, index = generate_sample_text()
            print(f"\nSample Text:\n{sample_text}\n")

            print("Guess the label:")
            print("1. Human-Written")
            print("2. Machine-Generated")
            print("3. Human-Written, Machine-Polished")
            print("4. Machine-Written, Machine-Humanized")
            
            guess = input("Enter your guess (1/2/3/4): ")

            try:
                guess = int(guess)
                if guess not in range(1, 5):
                    raise ValueError()

                # Determine correct label and check the guess
                label_mapping = {
                    1: "Human-Written",
                    2: "Machine-Generated",
                    3: "Human-Written, Machine-Polished",
                    4: "Machine-Written, Machine-Humanized"
                }

                correct_label = get_correct_label(index)
                user_guess_label = label_mapping[guess]

                if user_guess_label == correct_label:
                    print(f"Correct! The label is: {correct_label}")
                else:
                    print(f"Incorrect! The correct label was: {correct_label}")
            except ValueError:
                print("Invalid input. Please enter a number between 1 and 4.")

        elif mode == '3':
            print("Exiting the program. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()
