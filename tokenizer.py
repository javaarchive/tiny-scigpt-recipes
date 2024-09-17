import tiktoken

tiktoken_encoding = tiktoken.get_encoding("cl100k_base") # this used in gpt-4 amd 3.5-turbo
# old:
#.get_encoding("o200k_base") # this is used for gpt-4o apparently
vocab_size = tiktoken_encoding.n_vocab
print("vocab_size updated to",vocab_size)

def encode(text):
    return tiktoken_encoding.encode(text)

def decode(tokens):
    return tiktoken_encoding.decode(tokens)

