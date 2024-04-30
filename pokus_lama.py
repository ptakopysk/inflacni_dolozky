#!/usr/bin/env python3
#coding: utf-8

import sys

import logging
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_opts = [
        "BUT-FIT/CSTinyLlama-1.2B",
        ]
max_lengths = [
        2048,
        ]
output_len = 512

device = "cuda:0" if torch.cuda.is_available() else "cpu"

prompt_opts = ["Velikost inflační doložky za rok 2017?", "Co se v textu píše o inflační doložce?", "Vypiš z textu informace týkající se inflační doložky."]

for model_index in range(len(model_opts)):
    model_name = model_opts[model_index]
    logging.info(f"Load model {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_max_len = max_lengths[model_index]
    tokenizer.model_max_length = model_max_len
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)

    logging.info("Process files")
    for i in range(1,6):
        with open(f'smlouva{i}.txt') as infile:
            text = infile.read()
            for prompt_index in range(len(prompt_opts)):
                prompt = prompt_opts[prompt_index]
                full_prompt = prompt + "\n\n" + text + "\n\n" + prompt
                logging.info(full_prompt)
                tokenized_prompt = tokenizer.encode(full_prompt,
                        return_tensors='pt',
                        max_length=model_max_len-output_len,
                        truncation=True).to(device)
                logging.info(tokenized_prompt)
                out = model.generate(
                        tokenized_prompt,
                        max_new_tokens=output_len,
                        pad_token_id= tokenizer.pad_token_id,
                        eos_token_id = tokenizer.eos_token_id)
                decoded = tokenizer.decode(out[0][tokenized_prompt.shape[1]:], skip_special_tokens=True)
                with open(f'result_lama_{i}_{prompt_index}.txt', 'w') as outfile:
                    print(decoded, file=outfile)





