import gradio as gr
import numpy as np
import time

from tokenizer import encode, decode, vocab_size
from model import *

model = TokenBasedLanguageModel()
m = model.to(device)

print("Loading checkpoint from file")
checkpoint = torch.load("improved-v5.bin")
model.load_state_dict(checkpoint["model_state_dict"])
print("State restored")

def generate_llm(prompt, max_tokens = 512, analyze_probs = False):
    prompt_encoded = encode(prompt) # trigger book 2 intro
    #encode("[1]{.ePub-B}\n") # trigger first chapter
    context = torch.tensor(prompt_encoded, dtype = torch.long, device = device).view(1, len(prompt_encoded))
    output = prompt[:]
    start_time = time.time()
    token_count = 0
    probtext = ""
    for encoded_token_pair in model.generate(context, max_new_tokens=max_tokens, stream = True, stream_probs = analyze_probs):
        probtext = ""
        encoded_token = encoded_token_pair
        if analyze_probs:
            [encoded_token, probs] = encoded_token_pair
            prob_list = []
            for token_id in range(vocab_size):
                prob_list.append([token_id, probs[token_id]])
            prob_list.sort(key = lambda x: x[1], reverse = True)
            for prob_pair in prob_list[:25]:
                probtext += f'"{decode([prob_pair[0]])}": {prob_pair[1]}\n'
        else:
            probtext = "Feature disabled."
        part = decode([encoded_token])
        output += part
        token_count += 1
        yield [output, str(token_count / (time.time() - start_time)) + "tok/s " + str(token_count) + " tokens generated.", probtext]
    return [output, str(token_count / (time.time() - start_time)) + "tok/s " + str(token_count) + " tokens generated.", probtext]

demo = gr.Interface(generate_llm,
                    inputs=[gr.TextArea(placeholder = "In the midst of chaos."), gr.Number(value = 512, maximum = 2048, minimum = 1, step = 1, label = "Max tokens"), gr.Checkbox(label = "Show probs, 10x slower")],
                    outputs=[gr.TextArea(label = "Output"), gr.Text(placeholder = "tok/s and other stats", label = "Stats"), gr.TextArea(label = "Probability stats")])

demo.launch(share = True)