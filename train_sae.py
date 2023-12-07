# %%
import os
from pathlib import Path

import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_device

from sparse_autoencoder import (
    ActivationResampler,
    AdamWithReset,
    L2ReconstructionLoss,
    LearnedActivationsL1Loss,
    LossReducer,
    Pipeline,
    PreTokenizedDataset,
    SparseAutoencoder,
)
import wandb

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.random.manual_seed(49)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_NOTEBOOK_NAME"] = "train_sae.py"
# %%
SOURCE_DATA_BATCH_SIZE = 32
TRAIN_BATCH_SIZE = 8192
hyperparameters = {
    # Expansion factor is the number of features in the sparse representation, relative to the
    # number of features in the original MLP layer. The original paper experimented with 1x to 256x,
    # and we have found that 4x is a good starting point.
    "expansion_factor": 4,
    # L1 coefficient is the coefficient of the L1 regularization term (used to encourage sparsity).
    "l1_coefficient": 3e-4,
    # Adam parameters (set to the default ones here)
    "lr": 1e-4,
    "adam_beta_1": 0.9,
    "adam_beta_2": 0.999,
    "adam_epsilon": 1e-8,
    "adam_weight_decay": 0.0,
    # Batch sizes
    "train_batch_size": TRAIN_BATCH_SIZE,
    "context_size": 128,
    # Source model hook point
    "source_model_name": "gelu-2l",
    "source_model_dtype": "float32",
    "source_model_hook_point": "blocks.0.hook_mlp_out",
    "source_model_hook_point_layer": 0,
    # Train pipeline parameters
    "max_store_size": 384 * TRAIN_BATCH_SIZE * 2,
    "max_activations": 2_000_000_000,
    "resample_frequency": 122_880_000,
    "checkpoint_frequency": 100_000_000,
    "validation_frequency": 384 * TRAIN_BATCH_SIZE * 2 * 100,  # Every 100 generations
}
# %%
# Source model setup with TransformerLens
src_model = HookedTransformer.from_pretrained(
    str(hyperparameters["source_model_name"]),
    dtype=str(hyperparameters["source_model_dtype"]),
)

# Details about the activations we'll train the sparse autoencoder on
autoencoder_input_dim: int = src_model.cfg.d_model  # type: ignore (TransformerLens typing is currently broken)

f"Source: {hyperparameters['source_model_name']}, \
    Hook: {hyperparameters['source_model_hook_point']}, \
    Features: {autoencoder_input_dim}"
# %%
expansion_factor = hyperparameters["expansion_factor"]
autoencoder = SparseAutoencoder(
    n_input_features=autoencoder_input_dim,  # size of the activations we are autoencoding
    n_learned_features=int(autoencoder_input_dim * expansion_factor),  # size of SAE
).to(device)
autoencoder
# %%
# We use a loss reducer, which simply adds up the losses from the underlying loss functions.
loss = LossReducer(
    LearnedActivationsL1Loss(
        l1_coefficient=float(hyperparameters["l1_coefficient"]),
    ),
    L2ReconstructionLoss(),
)
loss
# %%
optimizer = AdamWithReset(
    params=autoencoder.parameters(),
    named_parameters=autoencoder.named_parameters(),
    lr=float(hyperparameters["lr"]),
    betas=(
        float(hyperparameters["adam_beta_1"]),
        float(hyperparameters["adam_beta_2"]),
    ),
    eps=float(hyperparameters["adam_epsilon"]),
    weight_decay=float(hyperparameters["adam_weight_decay"]),
)
optimizer
# %%
activation_resampler = ActivationResampler(
    resample_interval=10_000, n_steps_collate=10_000, max_resamples=5
)
source_data = PreTokenizedDataset(
    dataset_path="NeelNanda/c4-code-tokenized-2b",
    context_size=int(hyperparameters["context_size"]),
)
# %%
checkpoint_path = Path("../../.checkpoints")
checkpoint_path.mkdir(exist_ok=True)
Path(".cache/").mkdir(exist_ok=True)
wandb.init(
    project="sparse-autoencoder",
    dir=".cache",
    config=hyperparameters,
)
pipeline = Pipeline(
    activation_resampler=activation_resampler,
    autoencoder=autoencoder,
    cache_name=str(hyperparameters["source_model_hook_point"]),
    checkpoint_directory=checkpoint_path,
    layer=int(hyperparameters["source_model_hook_point_layer"]),
    loss=loss,
    optimizer=optimizer,
    source_data_batch_size=SOURCE_DATA_BATCH_SIZE,
    source_dataset=source_data,
    source_model=src_model,
)

pipeline.run_pipeline(
    train_batch_size=int(hyperparameters["train_batch_size"]),
    max_store_size=int(hyperparameters["max_store_size"]),
    max_activations=int(hyperparameters["max_activations"]),
    checkpoint_frequency=int(hyperparameters["checkpoint_frequency"]),
    validate_frequency=int(hyperparameters["validation_frequency"]),
)
wandb.finish()
# %%
