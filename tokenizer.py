"""
Tokenizer
  convert incoming training text into vectors of integers
    downcase A-Z to a-z
    remove punctuation, etc., keeping A-Za-z, space, and newline
    collapse multiple whitespace into single whitespace
  convert vectors of integers back into text
  
"""

class Tokenizer:
    def __init__(self):
        self.tokens = ' abcdefghijklmnopqrstuvwxyz\n'
        self.int2string = { k : v for k, v in enumerate(self.tokens) }
        self.string2int = { v : k for k, v in enumerate(self.tokens) }

    def size(self):
        return len(self.tokens)

    def encode(self, text):
        text = ''.join(c for c in text.lower() if c in self.tokens)
        words = text.split()
        return [self.string2int[c] for c in ' '.join(words)]

    def decode(self, vector):
        return ''.join(self.int2string[i] for i in vector)

if __name__ == '__main__':
    m = Tokenizer()
    s = 'Hello,  World!'
    v = m.encode(s)
    print(s, v, m.decode(v))
