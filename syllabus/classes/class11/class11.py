from transformers import pipeline
import torch
'''
model_name = "bert-large-uncased" # a small version of BERT

nlp = pipeline("fill-mask", model_name) # create pipeline

sentence_to_classify = "My name is Alex"

prompt = f"""
His name is John
Her name is Clara
His name Ross
Her name is Susan
Her name is Monica
His name is Bryan
{nlp.tokenizer.mask_token} name is Karen
"""

print(prompt)
print(nlp(prompt))
'''

#more detailed approach
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
# download and load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-large-uncased")


# create prompt:
sentence_to_classify = "works as a nurse"#"works as a doctor" #"works as a nurse"
prompt = f"""
He works as a teacher
She works as a teacher
He works as a dentist
She works as a dentist
He works as a salesman
She works as a salesman
He works as a judge
She works as a judge
He works as a bartender
She works as a bartender
He works as a dishwasher
She works as a dishwasher
{tokenizer.mask_token} {sentence_to_classify} 
"""
# tokenize the input
input = tokenizer.encode(prompt, return_tensors="pt")
mask_token_index = torch.where(input == tokenizer.mask_token_id)
mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]  # record the index of the mask token
# forward pass through the model
token_logits = model(input).logits
token_logits.shape  # (batch_size, tokens, vocabulary) in this case it is (1, 30, 30522)
# extract the most likely word for the MASK
mask_token_logits = token_logits[0, mask_token_index, :]  # select the mask token
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
for token in top_5_tokens:
    print(tokenizer.decode([token]))
