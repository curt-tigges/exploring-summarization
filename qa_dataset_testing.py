# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import einops
from functools import partial
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from jaxtyping import Float, Int, Bool
from typing import Dict, Iterable, List, Tuple, Union
from transformer_lens import HookedTransformer
from transformer_lens.utils import (
    get_dataset,
    tokenize_and_concatenate,
    get_act_name,
    test_prompt,
)
from transformer_lens.hook_points import HookPoint
from tqdm.notebook import tqdm
import pandas as pd
from circuitsvis.activations import text_neuron_activations
from circuitsvis.topk_samples import topk_samples
from IPython.display import HTML, display
from utils.circuit_analysis import get_logit_diff

from utils.tokenwise_ablation import (
    compute_ablation_modified_loss,
    load_directions,
    get_random_directions,
    get_zeroed_dir_vector,
    get_layerwise_token_mean_activations,
)
from utils.datasets import OWTData, PileFullData, PileSplittedData
from utils.neuroscope import plot_top_onesided

# %% [markdown]
# ### Load Dataset

# %%
from datasets import load_dataset

# %%
ds = load_dataset("tasksource/ruletaker")

# %%
ds['train'][0]['context']

# %%
QA_TEST = ds['train'][1]['context'] + '\nQ: ' + ds['train'][1]['question'] + ' True/false?' + '\nA:'
print(QA_TEST)


# %% [markdown]
# ### Data Preparation

# %%
def format_questions(example):
    return {
        "prompt": example["context"] + "\nQ: " + example["question"] + " True/false?" + "\nA:",
        "answer": ' True' if example["label"]=='entailment' else ' False',
    }

dataset = ds.map(format_questions)

# %%
from transformers import AutoTokenizer

model_checkpoint = "EleutherAI/pythia-410m"  # or any other appropriate small model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.pad_token = tokenizer.decode([1])
torch.set_grad_enabled(True)

def tokenize_and_format(examples):
    # Tokenize the prompts
    tokenized_inputs = tokenizer(examples["prompt"], return_tensors="np", padding="max_length", truncation=True, max_length=512)

    # Tokenize just the answers
    tokenized_answers = tokenizer(examples["answer"], return_tensors="np", add_special_tokens=False)["input_ids"]

    # Create labels with -100 for ignored positions
    labels = np.full(tokenized_inputs["input_ids"].shape, -100)

    # Check for nesting in tokenized_answers
    # if any(isinstance(i, (list, np.ndarray)) for i in tokenized_answers):
    #     raise ValueError("Nesting detected in tokenized_answers")

    # Place the answer token ID immediately after the end of the prompt
    for i, answer_ids in enumerate(tokenized_answers):
        end_of_prompt_index = np.where(tokenized_inputs["input_ids"][i] == tokenizer.pad_token_id)[0][0]
        labels[i, end_of_prompt_index-1:end_of_prompt_index+len(answer_ids)-1] = answer_ids

    # Check if shapes of labels and input_ids differ
    if labels.shape != tokenized_inputs["input_ids"].shape:
        raise ValueError("Shape mismatch between labels and input_ids")

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# %%
# sanity check
res = tokenize_and_format(dataset['train'][0:2])
idx = np.where(res['input_ids'][0] == tokenizer.pad_token_id)[0][0]
res['input_ids'][0][idx-1:idx+10], res['labels'][0][idx-1:idx+10]

# %%
tokenized_datasets = dataset.map(tokenize_and_format, batched=True)

# remove all columns except input_ids and labels
tokenized_datasets.set_format(type="torch", columns=["input_ids", "labels"])

# sanity check
tokenized_datasets['train'][0]['input_ids'].shape, tokenized_datasets['train'][0]['labels'].shape


# %%
from transformers import AutoModelForCausalLM
torch.set_grad_enabled(True)
#load model from results folder
model = AutoModelForCausalLM.from_pretrained('results')

# %%
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=40,
    per_device_eval_batch_size=40,
    num_train_epochs=7,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].select(range(80000)),
    eval_dataset=tokenized_datasets["test"].select(range(1000)),
    tokenizer=tokenizer,
)

trainer.train()

# %%
from datasets import load_metric

# Load your model and tokenizer
#model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
#tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Load the test dataset
test_dataset = dataset['test'].select(list(range(1000))) #dataset["test"]

# Define the evaluation function
def evaluate(model, tokenizer, dataset):
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_predictions = 0

    for example in dataset:
        inputs = tokenizer(example["prompt"], return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            # Get the position of the last token
            last_token_position = (inputs["input_ids"][0] == tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0] - 1

            # Get the logits for the last token position
            last_token_logits = logits[0, last_token_position]

            # Get the logits for the correct and incorrect answers
            correct_answer_token = tokenizer.encode(example["answer"], add_special_tokens=False)[0]
            incorrect_answer_token = tokenizer.encode(" True" if example["answer"] == " False" else " False", add_special_tokens=False)[0]
            correct_answer_logit = last_token_logits[correct_answer_token]
            incorrect_answer_logit = last_token_logits[incorrect_answer_token]

            # Check if the correct answer logit is greater than the incorrect answer logit
            if correct_answer_logit > incorrect_answer_logit:
                # Check if the correct answer logit is in the top ten logits
                top_ten_logits = torch.topk(last_token_logits, 10).indices
                if correct_answer_token in top_ten_logits:
                    correct_predictions += 1

            total_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy

# Evaluate the model
accuracy = evaluate(model, tokenizer, test_dataset)
print(f"Accuracy: {accuracy}")

# %%
trainer.evaluate(tokenized_datasets["test"])

# %%
trainer.save_model("results")

# %%
