"""
One-hot encoding example

"""
import torch

# for reproducibility
torch.manual_seed(8675309)

# alphabet and encoding dictionary
ALPHABET = ['a', 'b', 'c']
encode = {v: k for k,v in enumerate(ALPHABET)}
print('encode', encode)

# Produce column vector
def one_hot(position, size):
    return torch.tensor([[1] if i == position else [0] for i in range(size)])

# c is one-hot index tensor for letter c in alphabet
index_c = encode['c']
print('index_c', index_c)
c = one_hot(index_c, len(ALPHABET))
print('c', c)

# example neural network weights, one row for each letter in alphabet
weights = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]])
print('weights', weights)

# use c as index to get row of weights
row_c = (c * weights)[index_c]
print('row_c', row_c)

# ////////////////
# this time with Embedding

num_embeddings = len(ALPHABET) 
embedding_dim = len(ALPHABET) # for bigrams in the same alphabet

embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
print('weights', embedding.weight)

row_c = embedding(torch.tensor(index_c)))
print('row_c', row_c)
