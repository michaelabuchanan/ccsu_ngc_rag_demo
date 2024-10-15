from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM
import sys
import os
from sentence_transformers.util import semantic_search, dot_score
from sentence_transformers import SentenceTransformer
import pandas as pd

RAG = int(sys.argv[1]) # if 1 then do RAG, otherwise no RAG

if RAG:
    print("\nUsing RAG in this run!\n")
else:
    print("\nNo RAG this run!\n")

llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

tokenizer.pad_token = tokenizer.eos_token
llama_model.generation_config.pad_token_id = tokenizer.pad_token_id

if RAG:
    df = pd.read_csv("hf://datasets/Shengtao/recipe/recipe.csv")
    df = df.head(500)

    text = []
    for i, row in df.iterrows():
        text.append(row['title'] + ": " + row['directions'])

    print("Made recipe list")

    st_model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = []

    for line in text:
        embedding = st_model.encode(line, convert_to_tensor=True)
        embeddings.append(embedding)

    print("Made recipe embeddings")

prompt_task = "Question: "
prompt_end = "\nAnswer: "

prompts = [
    "What two ingredients do I need for two-ingredient pizza dough?",
    "What do I need for a quick tartar sauce?",
    "What should I preheat the oven to to make blueberry muffins?"
]

for prompt in prompts:
    prompt = prompt_task + prompt + prompt_end
    if RAG:
        prompt_embed = st_model.encode(prompt, convert_to_tensor=True)
        hits = semantic_search(prompt_embed, embeddings, top_k=2)

        result_line1 = hits[0][0]['corpus_id']
        result_line2 = hits[0][1]['corpus_id']

        prompt = text[result_line1] + "\n" + text[result_line2] + "\n\n" + prompt
    
    inputs = tokenizer(prompt, return_tensors="pt")

    generate_ids = llama_model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=25)

    print("\n")
    print(tokenizer.decode(generate_ids[0]))
    print("\n------------------------------------------------------------")

print("\n\nAll done")