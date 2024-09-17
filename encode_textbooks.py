import pandas as pd
import random
import time
# we use direct tiktoken encoding for perf
from tokenizer import encode, decode, tiktoken_encoding

dataset = pd.read_parquet('test-00000-of-00001.parquet')
print("Dataset size", len(dataset))

SEED = 123456789
random.seed(SEED)
N_TEXTBOOKS = 5000
MIN_SIZE = 128 + 5 # match with train value plus a bot
KEY = "textbook"

canidates = dataset[KEY].sample(n = N_TEXTBOOKS)
print(repr(canidates))
#list(range(len(dataset)))
#random.shuffle(canidates)
#canidates = canidates[:N_TEXTBOOKS]

start_time = time.time()
encoded = tiktoken_encoding.encode_batch(canidates)
print(len(encoded))
print(time.time() - start_time, "s")

# no ragged tensor in torch
# so this fail
# import torch
#encoded_tensor = torch.tensor(encoded, dtype = torch.long)
#torch.save(encoded_tensor, "encoded_textbooks.bin")
#print("Saved encoded/tokenized textbooks")
import numpy
encoded_array = numpy.array([encoded_textbook for encoded_textbook in encoded if len(encoded_textbook) > MIN_SIZE], dtype = object) # rip effiency 
numpy.save("textbooks.npy", encoded_array)