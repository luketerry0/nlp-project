from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
import numpy as np
from sklearn.metrics import Classification_report
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

for entry in sentence_data:
    question = f"""Answer the following question by selecting the correct option number 0-3. The following question is a trick question, so carefully consider your answer and explain your reasoning.
                   Question: {entry['question']}
                   Choices:
                   {entry['choice_order'][0]}: {entry['choice_list'][entry['choice_order'][0]]}
                   {entry['choice_order'][1]}: {entry['choice_list'][entry['choice_order'][1]]}
                   {entry['choice_order'][2]}: {entry['choice_list'][entry['choice_order'][2]]}
                   {entry['choice_order'][3]}: {entry['choice_list'][entry['choice_order'][3]]}
                   Provided expliantion and answer MUST be within 300 tokens or less.
                   The final answer MUST ONLY be the number corresponding to your answer, also it MUST be the last number in the response:"""
    
    encodeds = tokenizer(question, return_tensors="pt").to(device)
    generated_ids = model.generate(input_ids=encodeds['input_ids'], attention_mask=encodeds['attention_mask'], max_new_tokens=300, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    labels.append(entry['label'])

    match = re.search(r'\d+(?!.*\d)', decoded[0], re.DOTALL)
    if match:
        pred_label = int(match.group())
        print(pred_label)

        if(pred_label == entry['label']):
            correct_guesses += 1
        guesses.append(pred_label)
    else:
        print("no answer")
        guesses.append(-1)
        print(-1)

print(Classification_report(labels, guesses))

