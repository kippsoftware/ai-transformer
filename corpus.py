"""
Convert the Cornell Movie-Dialogs Corpus into AtomicML for GPT

https://convokit.cornell.edu/documentation/movie.html

Example conversation chunk:

  c
   ANDY
    Man has a complex.
   BUFFY
    He's got a...  What do you call it? A Napoleonic Code.

We will use these lines to train a GPT to write more conversations.

"""
import json
from io import StringIO

class Conversation:
    """An exchange of spoken lines"""
    def __init__(self):
        self.id = None
        self.movie = None
        self.lines = []
    def __str__(self):
        out = ['c']
        out.extend([f' {line[0]}\n  {line[1]}' for line in self.lines])
        return '\n'.join(out)

class Corpus:
    """Knows JSON schema for the corpus"""
    def __init__(self):
        self.conversations = None
        self.speakers = None
        self.utterances = None
        self.movies = {}
        self.count = 0

    def __str__(self):
        out = ['corpus']
        out.append(f'  movies {len(self.movies)}')
        out.append(f'  speakers {len(self.speakers)}')
        out.append(f'  conversations {len(self.conversations)}')
        out.append(f'  parsed {self.count}')
        return '\n'.join(out)

    def parse_speakers(self, filename):
        """Slurp speakers"""
        self.speakers = json.load(open(filename)) # fast enough
        for speaker in self.speakers.values():
            self.movies[speaker['meta']['movie_idx']] = speaker['meta']['movie_name']
        # [print(k,v) for k,v in self.movies.items()]

    def parse_conversations(self, filename):
        """Slurp conversations"""
        self.conversations = json.load(open(filename)) # fast enough

    def parse_conversation(self, filename):
        """Generator to parse utterances into conversation objects"""
        fp = open(filename)
        conversation = Conversation()
        line = fp.readline()
        while line:
            line = json.load(StringIO(line))
            speaker = self.speakers[line['speaker']]['meta']['character_name']
            text = line['text']
            conversation.lines.insert(0, (speaker, text))
            if line['conversation_id'] == line['id'] :
                conversation.id = line['id']
                conversation.movie = self.conversations[conversation.id]['meta']['movie_idx']
                yield conversation
                self.count += 1
                conversation = Conversation()
            line = fp.readline()

if __name__ == '__main__' :
    SPEAKERS_FILENAME = 'movie-corpus/speakers.json'
    CONVERSATIONS_FILENAME = 'movie-corpus/conversations.json'
    UTTERANCES_FILENAME = 'movie-corpus/utterances.jsonl'

    corpus = Corpus()
    corpus.parse_speakers(SPEAKERS_FILENAME)
    corpus.parse_conversations(CONVERSATIONS_FILENAME)
    fp = open('conversations.at', 'w')
    for conversation in corpus.parse_conversation(UTTERANCES_FILENAME):
        fp.write(str(conversation))
        fp.write('\n')
    fp.close()
    print(corpus)
