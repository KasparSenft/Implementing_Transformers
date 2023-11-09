from transformers import AutoTokenizer
from tokenizers import CharBPETokenizer
from torch import nn
from collections import defaultdict, Counter

from torch import nn
from collections import defaultdict

class BPE_Tokenizer():
    def __init__(self):
        #Preprocessor used to get words from corpus
        self.preprocess = AutoTokenizer.from_pretrained('gpt2')
        self.merges = defaultdict()


    def train(self, corpus_pth, vocab_size):
        #Intialize vocab, word freqs, and splits
        self.init_word_freqs(corpus_pth)
        self.init_vocab()
        
        #train loop until vocab size reached
        while len(self.vocab)< vocab_size:
            self.compute_pair_freq()
            pair = self.get_most_freq_pair()
            self.merge(pair)
            self.vocab.append(pair[0]+pair[1])

    def init_word_freqs(self,corpus_pth):
        #Get words from corpus and save their frequencies

        #Load corpus from file
        with open(corpus_pth, 'r') as f:
            corpus = f.read().splitlines()

        self.word_freqs = defaultdict(int)
        for text in corpus:
            words_and_offsets = self.preprocess.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text) #gets the word and their offset positionn
            words = [word for word, _ in words_and_offsets]
            for word in words:
                self.word_freqs[word] +=1

    def init_vocab(self):
        #Get vocab from corpus
        self.vocab = list(set(''.join(list(self.word_freqs.keys())))) + ["<|endoftext|>"]
        self.vocab.sort()
        self.init_splits() 

    def init_splits(self):
        #Initialise the splits as every character in the word
        self.splits = {word:list(word) for word in self.word_freqs.keys()}
        
    def compute_pair_freq(self):
        #computes the frequency of pairs given the current splits
        self.pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            for x,y in zip(split,split[1:]):
                self.pair_freqs[(x,y)] += freq

    def merge(self, pair):
        #merges the occurences of a given pair within the splits
        for word in self.word_freqs:
            split = self.splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [pair[0] + pair[1]] + split[i + 2 :]
                else:
                    i += 1
            self.splits[word] = split

        #Save the merge
        self.merges[pair] = pair[0]+pair[1]

    def get_most_freq_pair(self):
        #Get the most frequent pair of vocab entries
        return max(self.pair_freqs, key=self.pair_freqs.get)
    
    def __call__(self,text):
        #Tokenize an input string

        #Get the words from the pretokenizer
        pre_tokenize = self.preprocess._tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pre_tokenized_text = [word for word, _ in pre_tokenize]
        
        #get initial splits as characters
        splits = [list(word) for word in pre_tokenized_text]

        #Iterate through merges and apply them to splits
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2 :]
                    else:
                        i += 1
                splits[idx] = split

        return sum(splits, [])

BPE = BPE_Tokenizer()
BPE.train('data/corpus.txt', 64)
print(BPE('Machine learning is a subset of artificial intelligence.'))


hf_BPE = CharBPETokenizer()
hf_BPE.train('data/corpus.txt',vocab_size=256)
encoded = hf_BPE.encode('Machine learning is a subset of artificial intelligence.')
print(encoded.tokens)
