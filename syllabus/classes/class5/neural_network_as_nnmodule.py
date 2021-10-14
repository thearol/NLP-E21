"""
Logistic regression implemented using the nn.module class
"""
'''
pip install spacy
pip install networkx==2.3
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
pip install torch
pip install datasets
'''

import torch
import torch.nn as nn
from sklearn import datasets

from datasets import load_dataset

from collections import Counter

import spacy
nlp = spacy.load("en_core_web_sm")

from sklearn.feature_extraction import DictVectorizer

class Model(nn.Module):
    def __init__(self, n_input_features = 10):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(n_input_features, 30)
        self.linear2 = nn.Linear(30, 30)
        self.linear3 = nn.Linear(30, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = self.linear3(x)
        y_pred = torch.sigmoid(x)
        return y_pred

def flatten(t):
    return [item for sublist in t for item in sublist]

def term_freq(tokens: 'list[str]') -> dict:
    flatten_tokens = flatten(tokens) # only if it is a list of lists
    #print()
    term_freq_list = dict(Counter(flatten_tokens))
    
    tf = {}

    # Get term frequency
    for element in term_freq_list:
        tf[str(element)] = term_freq_list[str(element)]/len(tokens)
    return tf




#print(term_freq(string_list))
'''
def term_freq(tokens) :
   #: List[str]) -> dict:
    """
    Takes in a list of tokens (str) and return a 
    dictionary of term frequency of each token
    """
    d = len(tokens)
    terms = Counter(tokens)
    for key in terms:    
      terms[key] /=  d
    return terms
'''

dataset = load_dataset("emotion")

#training dataset
train = dataset["train"]
print(train)

print("A few samples:")
for t in range(10):
    sent = train["text"][t]
    lab = train["label"][t]
    print(sent, "-", lab)

train_flat = [item for sublist in train['text'] for item in sublist]
print(train_flat)
print(train['text'])
doc = nlp(train_flat)

#term freq

tf = term_freq(train['text'])
print(tf)

v = DictVectorizer(sparse=False)
X = v.fit_transform(tf)

'''
# Create dataset
X_numpy, y_numpy = datasets.make_classification(n_samples=1000, n_features=10, random_state=7)
X = torch.tensor(X_numpy, dtype=torch.float)
y = torch.tensor(y_numpy, dtype=torch.float)
y = y.view(y.shape[0], 1)
'''

X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(train['label'], dtype=torch.float)
y = y.view(y.shape[0], 1)

# initialize model
model = Model(n_input_features=10)

# define loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters()) 

# train
epochs = 10000
for epoch in range(epochs):
    # forward
    y_hat = model(X)

    # backward
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # some print to see that it is running
    if (epoch+1) % 1000 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')