from omegaconf import DictConfig, OmegaConf
import hydra
import random
import torch
import os
import plotly.express as px
from summarization_utils.store import ResultsFile
from summarization_utils.toy_datasets import CounterfactualDataset
from summarization_utils.counterfactual_patching import (
    patch_by_position_group,
    patch_by_layer,
    plot_layer_results_per_batch,
)
from summarization_utils.models import TokenSafeTransformer


@hydra.main(version_base=None, config_path="conf", config_name="patch_toy_dataset")
def main(
    cfg: DictConfig,
):
    print(OmegaConf.to_yaml(cfg))
    model_cfg = cfg.model
    dataset_cfg = cfg.dataset
    os.environ["TOKENIZERS_PARALLELISM"] = cfg.tokenizers_parallelism
    os.environ["CUDA_LAUNCH_BLOCKING"] = cfg.cuda_launch_blocking
    os.environ["TORCH_USE_CUDA_DSA"] = cfg.torch_use_cuda_dsa
    torch.set_grad_enabled(False)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    pos_results_file = ResultsFile(
        name="pos_patch_toy_dataset",
        extension="html",
        result_type="plots",
        model_name=model_cfg.model_name,
        dataset_name=dataset_cfg.dataset_name,
        dataset_size=dataset_cfg.dataset_size,
        seed=cfg.seed,
    )
    layer_results_file = ResultsFile(
        name="layer_patch_toy_dataset",
        extension="html",
        result_type="plots",
        model_name=model_cfg.model_name,
        dataset_name=dataset_cfg.dataset_name,
        dataset_size=dataset_cfg.dataset_size,
        seed=cfg.seed,
    )
    if cfg.skip_if_exists and pos_results_file.exists() and layer_results_file.exists():
        print(
            f"Results already exist at {pos_results_file.path} and {layer_results_file.path}."
        )
        return

    model = TokenSafeTransformer.from_pretrained(
        model_cfg.model_name,
        fold_ln=model_cfg.fold_ln,
        center_writing_weights=model_cfg.center_writing_weights,
        center_unembed=model_cfg.center_unembed,
        device=model_cfg.device,
        dtype=model_cfg.torch_dtype,
    )
    assert model.tokenizer is not None

    dataset = CounterfactualDataset.from_name(
        dataset_cfg.dataset_name, model, dataset_size=dataset_cfg.dataset_size
    )
    dataset.check_lengths_match()
    dataset.test_prompts(max_prompts=cfg.prompts, top_k=cfg.top_k)
    all_logit_diffs, cf_logit_diffs = dataset.compute_logit_diffs()
    print(f"Original mean: {all_logit_diffs.mean():.2f}")
    print(f"Counterfactual mean: {cf_logit_diffs.mean():.2f}")

    # Check accuracy is 100%
    assert (all_logit_diffs > 0).all(), (
        "Negative logit diff for "
        f"prompt={dataset.prompts[torch.where(all_logit_diffs < 0)[0][0]]}, "
        f"answer={dataset.answers[torch.where(all_logit_diffs < 0)[0][0]]}, "
        f"cf_answer={dataset.cf_answers[torch.where(all_logit_diffs < 0)[0][0]]}, "
        f"logit_diff={all_logit_diffs[torch.where(all_logit_diffs < 0)[0][0]]}"
    )
    assert (cf_logit_diffs < 0).all(), (
        "Positive logit diff for "
        f"prompt={dataset.cf_prompts[torch.where(cf_logit_diffs > 0)[0][0]]}, "
        f"answer={dataset.cf_answers[torch.where(cf_logit_diffs > 0)[0][0]]}, "
        f"cf_answer={dataset.cf_answers[torch.where(cf_logit_diffs > 0)[0][0]]}, "
        f"logit_diff={cf_logit_diffs[torch.where(cf_logit_diffs > 0)[0][0]]}"
    )

    if dataset_cfg.sep is not None and not (
        pos_results_file.exists() and cfg.skip_if_exists
    ):
        print("Patching by position...")
        results_pd = patch_by_position_group(dataset, sep=dataset_cfg.sep)
        bar = px.bar(
            results_pd.mean(axis=0),
            labels={"index": "Position", "value": "Patching metric"},
            title=f"Patching by position (model={model_cfg.model_name}, dataset={dataset_cfg.dataset_name})",
        )
        bar.update_layout(showlegend=False)
        pos_results_file.save(bar)
        print(f"...saved to {pos_results_file.path}.")

    if cfg.patch_by_layer and not (cfg.skip_if_exists and layer_results_file.exists()):
        print("Patching by layer...")
        pos_layer_results = patch_by_layer(dataset)
        layer_fig = plot_layer_results_per_batch(dataset, pos_layer_results)
        layer_results_file.save(layer_fig)
        print(f"...saved to {layer_results_file.path}.")
    print("Done.")


if __name__ == "__main__":
    main()
