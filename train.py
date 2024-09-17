import torch
import torch.nn as nn
from torch.nn import functional as F

from model import *

import sys
import time
import random

import collections # for defaultdict

from tokenizer import encode, decode, tiktoken_encoding, vocab_size

from tqdm import tqdm, trange

torch.manual_seed(1337)
random.seed(1337)

# old: wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('merged.md.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# old tokenizer
# here are all the unique characters that occur in this text
# chars = sorted(list(set(text)))
# vocab_size = len(chars)
# we upgrade the encode and decode funcs to use actual tokens
# referencing
# https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken



# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# textbook loader
import numpy
textbooks = []
if TRAIN:
    for textbook_tokenized in numpy.load("textbooks.npy", allow_pickle = True):
        if len(textbook_tokenized) < block_size + 4:
            print("WARNING: small textbook found")
        textbooks.append(torch.tensor(textbook_tokenized, dtype = torch.long))
    print("loaded",len(textbooks), " textbooks")

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    if split == "pretrain":
        ix = torch.randint(len(textbooks), (batch_size,))
        x_prestack = []
        y_prestack = []
        for textbook_id in ix:
            textbook_tensor = textbooks[textbook_id]
            textbook_len = len(textbook_tensor)
            upper_bound = textbook_len - block_size - 2 # TODO; remove -2 not needed?
            offset = random.randint(0, upper_bound)
            x_prestack.append(textbook_tensor[offset: offset + block_size])
            y_prestack.append(textbook_tensor[offset + 1: offset + block_size + 1])
        x = torch.stack(x_prestack)
        y = torch.stack(y_prestack)
        x = x.to(device)
        y = y.to(device)
        return x, y

    else:
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['pretrain', 'train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = TokenBasedLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

if TRAIN and __name__ == "__main__":
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = 0.1)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": estimate_loss(),
    }, "improved-v5-init.bin")

    iterator = trange(max_iters) # tqdm(range(max_iters))
    for iter in iterator:

        # sample a batch of data
        split = "train"
        if iter < (PRETRAIN_PERCENTAGE * max_iters):
            split = "pretrain"

        #if iter == (PRETRAIN_PERCENTAGE * max_iters):
            # https://stackoverflow.com/questions/48324152/how-to-change-the-learning-rate-of-an-optimizer-at-any-given-moment-no-lr-sched
            #for g in optimizer.param_groups:
            #    g["lr"] = 1e-5

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            iterator.set_description(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, pretrain loss {losses['pretrain']:.4f}, split: {split}")

        
        xb, yb = get_batch(split)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": estimate_loss(),
    }, "improved-v5.bin")

else:
    print("Loading checkpoint from file")
    checkpoint = torch.load("improved-v5.bin")
    model.load_state_dict(checkpoint["model_state_dict"])
    print("State restored")
    # optimizer state not needed for inference

if __name__ == "__main__":
    # generate from the model
    prompt = "[Prologue](../Text/part0020.xhtml#c_pro){.calibre2}\n\n"
    prompt_encoded = encode(prompt) # trigger book 2 intro
    #encode("[1]{.ePub-B}\n") # trigger first chapter
    context = torch.tensor(prompt_encoded, dtype = torch.long, device = device).view(1, len(prompt_encoded))
    # torch.zeros((1, 1), dtype=torch.long, device=device)

    # print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
    output = prompt[:]
    print("# begin generation")

    start_time = time.time()

    token_count = 0

    for token in m.generate(context, max_new_tokens=2000, stream = True):
        chunk = decode([token])
        print(chunk, end = "")
        sys.stdout.flush()
        output += chunk
        token_count += 1
    with open("sample_output.txt", "w") as f:
        f.write(output)
    print("# end generation")
    print("average tokens/s", token_count / (time.time() - start_time))