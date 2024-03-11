# LLaMA

A simple llama2 demo. This repo is just for learning and backup. 

Caution: I haven't do any test on llama2 model, This demo was only written by code without any test. In fact, I do not like LLaMA impletement for the next reasons:

1, vocab-size=32000, that mean the multi-language will be supported not so well.
2, The vocab embedding size module and output project module are redundent(about 260m parameters here.) for me. 
3, The model-size was too large(7B) for me.

Please forget all aka code here. It's a sample proxy to torch:

    aka.nn --> torch.nn
    aka.numpy --> torch + torch.nn.F

## Requirements

    python
    torch
    torchvision
    transformer

## Prepare

Download llama files from: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

to folder like:

    data/Llama-2-7b-chat-hf

## Run

> python LLaMA.py
