import gensim.downloader as api
import numpy as np

model = api.load("glove-wiki-gigaword-50")

''' - 1) Calculate the dot product between 
two word embedding which you believe are similar
'''
np.dot(model["dog"], model["puppy"])

'''2) Calculate the dot product between the two word and a 
word which you believe is dissimilar'''
dog = model["dog"]
puppy = model["puppy"]
sneeze = model["sneeze"]

np.dot(dog, puppy)
np.dot(puppy, sneeze)

'''
- 3) make the three words into a matrix $E$ and multiply it with its 
own transpose using matrix multiplication. So $E \cdot E^T$
  - what does the values in matric correspond to? What do you imagine 
  the dot product is? *Hint*, similarity between vectors (cosine similarity) 
  is exactly the same as the dot product assuming you normalize the lenghth 
  of each vector to 1.
'''

X = np.array([dog,puppy,sneeze])
X.shape

X_t = X.transpose()
X_t.shape

matrix = np.dot(X, X_t)
matrix.shape

print(matrix)




