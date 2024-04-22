"""
BigramModule
  neural network that learns probabilities of one token following another in a corpus
  size is number of distinct tokens we are using
  input_tensor is matrix of token numbers, Batch x Block

"""
import torch

class BigramModule(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.embedding = torch.nn.Embedding(size, size)

    def forward(self, input_tensor, target_tensor = None):
        """input_tensor is batch_size x block_size of token numbers"""

        # logits is BxTxC, batch x time x channel
        logits = self.embedding(input_tensor) 

        # we want to predict what comes next given single token

        # give names to dimensions and view tensors along channel dimension
        loss = None
        if target_tensor != None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target_tensor = target_tensor.view(B * T)
            # compute negative log likelihood
            loss = torch.nn.functional.cross_entropy(logits, target_tensor)

        return logits, loss

    def generate(self, source_tensor, max_new_tokens):
        # source_tensor is BxT of indexes
        for _ in range(max_new_tokens):
            # get the predictions; ignore the loss
            logits, _ = self(source_tensor)

            # look at last time step--the last element in the time dimension
            logits = logits[:, -1, :] # BxTxC becomes BxC

            # apply softmax to get probabilities that sum to 1
            probs = torch.nn.functional.softmax(logits, dim=-1) # probs is BxC

            # use random value to pick the next token
            next_index = torch.multinomial(probs, num_samples=1) # Bx1

            # append to source_tensor
            source_tensor = torch.cat((source_tensor, next_index), dim=1) # BxT+1

        return source_tensor
