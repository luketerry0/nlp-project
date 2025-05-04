from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
import numpy as np
from sklearn.metrics import classification_report
import re

device = "cuda" # the device to load the model onto

config = BitsAndBytesConfig(load_in_4bit=True)

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto", quantization_config=config, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

sentence_data = np.load("data/sentence_puzzle.npy", allow_pickle=True)
word_data = np.load("data/word_puzzle.npy", allow_pickle=True)

correct_guesses = 0
guesses = []
labels = []

sentence_example = """EXAMPLE:
                   Question: A man shaves everyday, yet keeps his beard long.
                   Choices:
                   0: He is a barber.
                   1: He wants to maintain his appearance.
                   2: He wants his girlfriend to buy him a razor.
                   3: None of the above.

                   Explaination: He is not shaving his own beard, so the man must be a barber.
                   
                   Answer: 0"""

for entry in word_data:
    question = f"""EXAMPLE:
                   Question: What part of London is in France?	
                   Choices:
                   0: The letter N.
                   1: The letter O.
                   2: The letter L.
                   3: None of the above.
                   
                   Explaination: The letter N is contained in both words London and France.
                   
                   Answer: 0
                   ---
                   Now answer the following question by selecting the correct option number 0-3. The following question is a puzzle where words can defy their common meanings which requires you to think carefully about your answer.
                   Question: {entry['question']}
                   Choices:
                   {entry['choice_order'][0]}: {entry['choice_list'][entry['choice_order'][0]]}
                   {entry['choice_order'][1]}: {entry['choice_list'][entry['choice_order'][1]]}
                   {entry['choice_order'][2]}: {entry['choice_list'][entry['choice_order'][2]]}
                   {entry['choice_order'][3]}: {entry['choice_list'][entry['choice_order'][3]]}
                   Provided expliantion and answer MUST be within 600 tokens.
                   The answer MUST be the last line of your response and you must use the integer value associated with the answer from the choices (No Roman Numerals).
                   No other number may come after your answer choice.
                   Do not create new questions."""
    
    encodeds = tokenizer(question, return_tensors="pt").to(device)
    generated_ids = model.generate(input_ids=encodeds['input_ids'], attention_mask=encodeds['attention_mask'], max_new_tokens=600, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    labels.append(entry['label'])
    print(decoded[0])

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

labels = [entry['label'] for entry in word_data]
print(classification_report(labels, guesses))

