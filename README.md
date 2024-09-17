# tiny-scigpt-recipes
training and model arch tweaks to a tiny model that is some weird mix of llama and gpt. 
## usage
This training script is a bit more weird in which it has two phases. See `model.py` for training config. Phase one pretrains the LLM on a set of "textbooks" from [this huggingface dataset](https://huggingface.co/datasets/nampdn-ai/tiny-textbooks). The second phase trains from those weights on the contents of `merged.md.txt`. Textbooks are processed before training to save memory, run `python encode_textbooks.py` to do that. Finally to train, `python train.py` and you get a neat tqdm progress bar and estimate (note: evaling is slow).
