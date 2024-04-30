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

model_opts = [
        "lchaloupsky/czech-gpt2-oscar",
        "BUT-FIT/Czech-GPT-2-XL-133k",
        "BUT-FIT/CSTinyLlama-1.2B",
        "simecek/cswikimistral_0.1"
        ]
for model_index in range(len(model_opts)):
    model_name = model_opts[model_index]
    logging.info(f"Load model {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    full_prompt = "Jak se Napoleon zove?"
    logging.info(full_prompt)
    tokenized_prompt = tokenizer.encode(full_prompt, return_tensors='pt')
    logging.info(tokenized_prompt)
    
    out = model.generate(
            tokenized_prompt,
            max_new_tokens=512,
            pad_token_id= tokenizer.pad_token_id,
            eos_token_id = tokenizer.eos_token_id)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    logging.info("Writing output")
    with open(f'napoleon_{model_index}.txt', 'w') as outfile:
        print(decoded, file=outfile)
    logging.info("Done with this model")


