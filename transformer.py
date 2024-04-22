"""
Transformer
  open corpus
  read batches of training input from corpus
    reserve 20% of corpus for validation and test
      index(item) % 5 == 0 (every fifth one)
  forward input through the machine
  backward and adjust gradients

Example conversation chunk
  c
   ANDY
    Man has a complex.
   BUFFY
    He's got a...  What do you call it? A Napoleonic Code.

"""
import sys
sys.path.insert(1, '../markup-tools/')
import atomicml
import torch
import tokenizer
import bigram

# reproduce results
torch.manual_seed(8675309)

class Transformer:

    def __init__(self, source):
        # capture the generator
        self.node_generator = atomicml.parse_node(source)

        # which conversation will be used next
        self.which_chunk = 0

        # maximum number of characters presented to the transformer
        self.block_size = 18

        # number of simultaneous blocks presented to the transformer
        self.batch_size = 4
    
        # convert string into list of integer
        self.tokenizer = tokenizer.Tokenizer()


    def next_chunk(self):
        self.which_chunk += 1
        node = next(self.node_generator, None)
        while not self.which_chunk % 5:
            # skip 1 in 5 chunks so we can use those chunks to validate later
            self.which_chunk += 1
            node = next(self.node_generator, None)
        return node

    def next_tokens(self):
        """Create tokens from a single sample from the corpus"""
        node = self.next_chunk()
        text = '\n'.join([str(child) for child in node.children])
        # print('text', text)
        tokens = self.tokenizer.encode(text)[:self.block_size + 1]
        tokens.extend(0 for _ in range(self.block_size - len(tokens) + 1))
        # print('tokens', tokens)
        return torch.tensor(tokens)

    def next_batch(self):
        """Create input and target tensors, offset by +1"""
        samples = [self.next_tokens() for _ in range(self.batch_size)] 
        source_tensor = torch.stack([line[0:self.block_size] for line in samples])
        target_tensor = torch.stack([line[1:] for line in samples])
        return source_tensor, target_tensor

    def train_batch(self):
        source_tensor, target_tensor = self.next_batch()
        logits, loss = self.bigram(source_tensor, target_tensor)
        # print('logits', logits.shape) # 4 x 18 x 28
        # print('loss', loss)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss

    def train_batches(self):
        self.bigram = bigram.BigramModule(transformer.tokenizer.size())
        # adjust gradients
        self.optimizer = torch.optim.AdamW(self.bigram.parameters(), lr=1e-3)
        for _ in range(20000):
            try:
                loss = self.train_batch()
            except Exception as e:
                break
        print(loss.item())
        
    def generate(self):
        # run on an untrained model; get garbage
        source_tensor = torch.zeros((1,1), dtype=torch.long)
        out = self.bigram.generate(source_tensor, 500)[0].tolist()
        print(out)
        print(self.tokenizer.decode(out))

if __name__ == '__main__':
    fp = open('conversations.at')
    transformer = Transformer(fp)
    input_tensor, target_tensor = transformer.next_batch()
    print('input_tensor', input_tensor)
    print('target_tensor', target_tensor)

    transformer.train_batches()
    transformer.generate()
