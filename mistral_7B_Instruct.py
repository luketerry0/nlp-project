from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
import numpy as np
from sklearn.metrics import classification_report
import re

device = "cuda" # the device to load the model onto

# Make the LLM quantized so it can fit on the hardware
config = BitsAndBytesConfig(load_in_8bit=True)

# Load model from huggingface
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto", quantization_config=config, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# Import the raw data from the challenge
sentence_data = np.load("data/sentence_puzzle.npy", allow_pickle=True)
word_data = np.load("data/word_puzzle.npy", allow_pickle=True)

# track LLM guesses
correct_guesses = 0
guesses = []
labels = []

# example sentence question from challenge
sentence_example = """EXAMPLE QUESTION:
                   Question: A man shaves everyday, yet keeps his beard long.
                   Choices:
                   0: He is a barber.
                   1: He wants to maintain his appearance.
                   2: He wants his girlfriend to buy him a razor.
                   3: None of the above.

                   Explaination: He is not shaving his own beard, so the man must be a barber.
                   
                   Answer: 0
                   ---"""
# example word question from challenge
word_example = """EXAMPLE QUESTION:
                   Question: What part of London is in France?	
                   Choices:
                   0: The letter N.
                   1: The letter O.
                   2: The letter L.
                   3: None of the above.
                   
                   Explaination: The letter N is contained in both words London and France.
                   
                   Answer: 0
                   ---"""
# directives to keep LLM consitent and speed up result gathring
example_params = """Provided expliantion and answer MUST be within 600 tokens.
                   The answer MUST be the last line of your response and you must use the integer value associated with the answer from the choices (No Roman Numerals).
                   No other number may come after your answer choice.
                   Do not create new questions."""
# framing for the LLM to better understand what the questions is doing
context = """The following question is a puzzle where words can defy their common meanings which requires you to think carefully about your answer."""

# Loop through all questions from the data set promting the LLM for a response
for entry in word_data:

    question = f"""
Answer the following question by selecting the correct option number 0-3.

Question: {entry['question']}
Choices:
    {entry['choice_order'][0]}: {entry['choice_list'][entry['choice_order'][0]]}
    {entry['choice_order'][1]}: {entry['choice_list'][entry['choice_order'][1]]}
    {entry['choice_order'][2]}: {entry['choice_list'][entry['choice_order'][2]]}
    {entry['choice_order'][3]}: {entry['choice_list'][entry['choice_order'][3]]}
Don't provide an explaination ONLY respond with the selected number listed with you chosen answer.
No other number may come after your answer choice."""
    
    # encode the question and send it to the LLM
    encodeds = tokenizer(question, return_tensors="pt").to(device)
    # generate a response
    generated_ids = model.generate(input_ids=encodeds['input_ids'], attention_mask=encodeds['attention_mask'], max_new_tokens=10, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    labels.append(entry['label'])
    print(decoded[0])

    # capture the models answer from the sea of response text
    match = re.search(r'([0-3])(?=[^\d]*$)', decoded[0])
    print(match)
    if match:
        pred_label = int(match.group(1))
        print(pred_label)

        if(pred_label == entry['label']):
            correct_guesses += 1
        guesses.append(pred_label)
    else:
        print("no answer")
        guesses.append(-1)
        print(-1)
# Get the report of its correct guesses vs our true labels
labels = [entry['label'] for entry in word_data]
print(classification_report(labels, guesses))

