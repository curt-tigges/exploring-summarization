# %%
from typing import List, Tuple
from datasets import load_dataset, DatasetDict, Dataset
from jaxtyping import Float, Int
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name, test_prompt
from summarization_utils.patching_metrics import get_final_token_logits

# %%
BATCH_SIZE = 8
TOPK = 10
SEED = 0
torch.set_grad_enabled(False)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA"] = "1"
# %%
model = HookedTransformer.from_pretrained(
    "google/gemma-2b",
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    device="cuda",
    dtype="bfloat16",
)
LAYER = model.cfg.n_layers // 2
# %%
ds = load_dataset("NeelNanda/counterfact-tracing")
assert isinstance(ds, DatasetDict)
# %%
french_data = ds["train"].filter(lambda x: x["target_true"] == " French")
non_french_data = ds["train"].filter(lambda x: x["target_false"] == " French")
print(len(french_data), len(non_french_data))
# # %%
# for i in range(5):
#     prompt = french_data["prompt"][i]
#     target_true = french_data["target_true"][i]
#     test_prompt(prompt, target_true, model)
# %%
# for i in range(5):
#     prompt = non_french_data["prompt"][i]
#     target_false = non_french_data["target_true"][i]
#     test_prompt(prompt, target_false, model)


# %%
def filter_by_output_probability(batch: Dataset) -> List[bool]:
    batch_tokens = model.to_tokens(batch["prompt"])
    batch_size = batch_tokens.shape[0]
    batch_answers = model.to_tokens(
        batch["target_true"], prepend_bos=False, move_to_device=False
    )[:, 0]
    batch_wrong_answers = model.to_tokens(
        batch["target_false"], prepend_bos=False, move_to_device=False
    )[:, 0]
    batch_logits = model(batch_tokens, return_type="logits")
    batch_logits = get_final_token_logits(
        batch_logits, tokenizer=model.tokenizer, tokens=batch_tokens
    )
    answer_logits = batch_logits[torch.arange(batch_size), batch_answers]
    wrong_answer_logits = batch_logits[torch.arange(batch_size), batch_wrong_answers]
    ld_mask = answer_logits > wrong_answer_logits
    topk_threshold = batch_logits.topk(TOPK, dim=1).values.min(dim=1).values
    topk_mask = answer_logits > topk_threshold
    mask = ld_mask & topk_mask
    return mask.tolist()


# %%
def batched_to_str_tokens(
    prompts: List[str], model: HookedTransformer
) -> List[List[str]]:
    tokens: Int[Tensor, "batch pos"] = model.to_tokens(prompts)
    return [model.to_str_tokens(t) for t in tokens]  # type: ignore


# %%
french_data = french_data.filter(
    filter_by_output_probability, batch_size=BATCH_SIZE, batched=True
)
non_french_data = non_french_data.filter(
    filter_by_output_probability, batch_size=BATCH_SIZE, batched=True
)
half_length = min(len(french_data), len(non_french_data))
french_data = french_data.select(range(half_length))
non_french_data = non_french_data.select(range(half_length))
assert len(french_data) == len(non_french_data)
print(len(french_data))
# %%
french_data["prompt"][:5]
# %%
ACT_NAME = get_act_name("resid_post", LAYER)
print(ACT_NAME)


# %%
def create_probing_dataset(
    data, model, act_name, batch_size, device="cpu"
) -> Tuple[Float[Tensor, "batch d_model"], Float[Tensor, "batch"]]:
    vectors = torch.zeros(
        (len(data), model.cfg.d_model), dtype=torch.float32, device=device
    )
    logit_diffs = torch.zeros(len(data), dtype=torch.float32, device=device)
    for batch_idx in range(0, len(data), batch_size):
        batch = data[batch_idx : batch_idx + batch_size]
        batch_tokens = model.to_tokens(batch["prompt"])
        batch_size = batch_tokens.shape[0]
        prompt_str_tokens = batched_to_str_tokens(batch["prompt"], model)
        subject_str_tokens = model.to_str_tokens(batch["subject"])
        subject_positions = [
            prompt_str_tokens[i].index(s[-1]) for i, s in enumerate(subject_str_tokens)
        ]
        answer_true = model.to_tokens(batch["target_true"], prepend_bos=False)[:, 0]
        answer_false = model.to_tokens(batch["target_false"], prepend_bos=False)[:, 0]
        batch_logits, batch_cache = model.run_with_cache(
            batch_tokens,
            return_type="logits",
            names_filter=lambda name: name == act_name,
        )
        batch_logits = get_final_token_logits(
            batch_logits, tokenizer=model.tokenizer, tokens=batch_tokens
        )
        assert batch_cache[act_name].shape == (
            batch_size,
            batch_tokens.shape[1],
            model.cfg.d_model,
        ), (
            f"Expected shape {(batch_size, batch_tokens.shape[1], model.cfg.d_model)} "
            f"but got {batch_cache[act_name].shape}. "
            f"Input shape {batch_tokens.shape}."
        )
        assert answer_true.shape == answer_false.shape == (batch_size,)
        vectors[batch_idx : batch_idx + batch_size] = batch_cache[act_name][
            torch.arange(batch_size), subject_positions
        ]
        logit_diffs[batch_idx : batch_idx + batch_size] = (
            batch_logits[torch.arange(batch_size), answer_true]
            - batch_logits[torch.arange(batch_size), answer_false]
        )
    return vectors, logit_diffs


# %%
french_vectors, french_logit_diffs = create_probing_dataset(
    french_data, model, ACT_NAME, BATCH_SIZE
)
non_french_vectors, non_french_logit_diffs = create_probing_dataset(
    non_french_data, model, ACT_NAME, BATCH_SIZE
)
is_french = torch.cat(
    [torch.ones(len(french_vectors)), torch.zeros(len(non_french_vectors))]
)
probing_vectors = torch.cat([french_vectors, non_french_vectors])
# %%
logit_diffs_cat = torch.cat([french_logit_diffs, non_french_logit_diffs])
print(logit_diffs_cat.mean().item(), (logit_diffs_cat > 0).float().mean().item())
# %%
X_train, X_test, y_train, y_test = train_test_split(
    probing_vectors, is_french, test_size=0.2
)
# %%
lr = LogisticRegression(random_state=SEED)
lr.fit(X_train, y_train)


# %%
def get_accuracy_mean_and_std_error(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    is_correct = (y_true == y_pred).astype(float)
    accuracy = is_correct.mean()
    std_error = is_correct.std() / (len(is_correct) ** 0.5)
    return accuracy, std_error


# %%
insample_pred = lr.predict(X_train)
get_accuracy_mean_and_std_error(y_train, insample_pred)
# %%
y_pred = lr.predict(X_test)
get_accuracy_mean_and_std_error(y_test, y_pred)
# %%
print(lr.coef_.shape)
# %%
NAMES = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Hannah"]
toy_prompts = (
    # positive examples
    [f"{name} lives in France. {name}" for name in NAMES]
    + [f"{name} is French. {name}" for name in NAMES]
    + [f"{name} was born in France. {name}" for name in NAMES]
    + [f"{name} lives in Paris. {name}" for name in NAMES]
    + [f"{name} is from Paris. {name}" for name in NAMES]
    # negative examples
    + [f"{name} lives in England. {name}" for name in NAMES]
    + [f"{name} is English. {name}" for name in NAMES]
    + [f"{name} was born in England. {name}" for name in NAMES]
    + [f"{name} lives in London. {name}" for name in NAMES]
    + [f"{name} is from London. {name}" for name in NAMES]
)
toy_activations = torch.zeros(
    (len(toy_prompts), model.cfg.d_model), dtype=torch.float32, device="cpu"
)
for batch_idx in range(0, len(toy_prompts), BATCH_SIZE):
    batch = toy_prompts[batch_idx : batch_idx + BATCH_SIZE]
    batch_tokens = model.to_tokens(batch)
    _, batch_cache = model.run_with_cache(
        batch_tokens, return_type=None, names_filter=lambda name: name == ACT_NAME
    )
    toy_activations[batch_idx : batch_idx + BATCH_SIZE] = batch_cache[ACT_NAME][:, -1]
# %%
toy_is_french = torch.cat(
    [torch.ones(len(toy_prompts) // 2), torch.zeros(len(toy_prompts) // 2)]
)
toy_pred = lr.predict(toy_activations)
get_accuracy_mean_and_std_error(toy_is_french, toy_pred)

# %%
