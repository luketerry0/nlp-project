"""
This file fine tunes DeBERTa v3 on the task, as was done by the winning team
https://arxiv.org/pdf/2403.00809
""" 

from transformers import AutoModel, AutoModel, AutoConfig
MODEL_NAME = 'microsoft/deberta-v3-base'
model = AutoModel.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)