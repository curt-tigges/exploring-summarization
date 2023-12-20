import einops
from jaxtyping import Float, Int
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.utils import test_prompt, get_attention_mask
from summarization_utils.circuit_analysis import get_logit_diff
from summarization_utils.path_patching import Node, IterNode, act_patch


KNOWN_FOR_TUPLES = [
    (
        "Known for being the first to walk on the moon, Neil",
        " Armstrong",
        "Known for being the star of the movie Jazz Singer, Neil",
        " Diamond",
    ),
    (
        "Known for being the first to cross Antarctica, Sir",
        " Ernest",
        "Known for being the first to summit Everest, Sir",
        " Edmund",
    ),
    (
        "Known for being the fastest production car in the world, the",
        " McL",
        "Known for being the best selling car in the world, the",
        " Ford",
    ),
    (
        "Known for being the most popular fruit in the world, the humble",
        " apple",
        "Known for being the most popular vegetable in the world, the humble",
        " potato",
    ),
    (
        "Known for being a wonder of the world, located in Australia, the",
        " Great",
        "Known for being a wonder of the world, located in India, the",
        " Taj",
    ),
    (
        "Known for being the most popular sport in Brazil, the game of",
        " soccer",
        "Known for being the most popular sport in India, the game of",
        " cricket",
    ),
]

OF_COURSE_TUPLES = [
    (
        "The first to walk on the moon is of course, Neil",
        " Armstrong",
        "The star of the movie Jazz Singer is of course, Neil",
        " Diamond",
    ),
    (
        "The first to cross Antarctica was of course, Sir",
        " Ernest",
        "The first to summit Everest was of course, Sir",
        " Edmund",
    ),
    (
        "The fastest production car in the world is of course, the",
        " McL",
        "The best selling car in the world is of course, the",
        " Ford",
    ),
    (
        "The most popular fruit in the world is of course, the humble",
        " apple",
        "The most popular vegetable in the world is of course, the humble",
        " potato",
    ),
    (
        "The most popular sport in Brazil is of course, the game of",
        " soccer",
        "The most popular sport in India is of course, the game of",
        " cricket",
    ),
]


def patch_prompt_by_layer(
    prompt: str,
    answer: str,
    cf_prompt: str,
    cf_answer: str,
    model: HookedTransformer,
    prepend_bos: bool = True,
) -> Float[np.ndarray, "layer pos"]:
    prompt_tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    cf_tokens = model.to_tokens(cf_prompt, prepend_bos=prepend_bos)
    answer_id = model.to_single_token(answer)
    cf_answer_id = model.to_single_token(cf_answer)
    answer_tokens = torch.tensor(
        [answer_id, cf_answer_id], dtype=torch.int64, device=model.cfg.device
    ).unsqueeze(0)
    assert prompt_tokens.shape == cf_tokens.shape, (
        f"Prompt and counterfactual prompt must have the same shape, "
        f"for prompt {prompt} "
        f"got {prompt_tokens.shape} and {cf_tokens.shape}"
    )
    model.reset_hooks(including_permanent=True)
    base_logits_by_pos: Float[Tensor, "1 seq_len d_vocab"] = model(
        prompt_tokens,
        prepend_bos=False,
        return_type="logits",
    )
    base_logits: Float[Tensor, "... d_vocab"] = base_logits_by_pos[:, -1, :]
    base_ldiff = get_logit_diff(base_logits, answer_tokens=answer_tokens)
    cf_logits, cf_cache = model.run_with_cache(
        cf_tokens, prepend_bos=False, return_type="logits"
    )
    assert isinstance(cf_logits, Tensor)
    cf_ldiff = get_logit_diff(cf_logits, answer_tokens=answer_tokens)
    nodes = IterNode(node_names=["resid_pre"], seq_pos="each")
    metric = lambda logits: (
        get_logit_diff(logits, answer_tokens=answer_tokens) - base_ldiff
    ) / (cf_ldiff - base_ldiff)
    results = act_patch(
        model, prompt_tokens, nodes, metric, new_cache=cf_cache, verbose=True
    )[
        "resid_pre"
    ]  # type: ignore
    results = torch.stack(results, dim=0)  # type: ignore
    results = einops.rearrange(
        results,
        "(pos layer) -> layer pos",
        layer=model.cfg.n_layers,
        pos=prompt_tokens.shape[1],
    )
    results = results.cpu().numpy()
    return results


def get_position_dict(
    tokens: Float[Tensor, "batch seq_len"], model: HookedTransformer, sep=","
) -> Dict[str, List[List[int]]]:
    """
    Returns a dictionary of positions of clauses and separators
    e.g. {"clause_0": [[0, 1, 2]], "sep_0": [[3]], "clause_1": [[4, 5, 6]]}
    """
    sep_id = model.to_single_token(sep)
    batch_size, seq_len = tokens.shape
    full_pos_dict = dict()
    for b in range(batch_size):
        batch_pos_dict = {}
        sep_count = 0
        current_clause = []
        for s in range(seq_len):
            if tokens[b, s] == sep_id:
                batch_pos_dict[f"clause_{sep_count}"] = current_clause
                batch_pos_dict[f"sep_{sep_count}"] = [s]
                sep_count += 1
                current_clause = []
                continue
            else:
                current_clause.append(s)
        if current_clause:
            batch_pos_dict[f"clause_{sep_count}"] = current_clause
        for k, v in batch_pos_dict.items():
            if k in full_pos_dict:
                full_pos_dict[k].append(v)
            else:
                full_pos_dict[k] = [v]
    return full_pos_dict


class CounterfactualDataset:
    def __init__(
        self,
        prompts: List[str],
        answers: List[str],
        cf_prompts: List[str],
        cf_answers: List[str],
        model: HookedTransformer,
    ) -> None:
        self.prompts = prompts
        self.answers = answers
        self.cf_prompts = cf_prompts
        self.cf_answers = cf_answers
        self.model = model
        self.device = self.model.cfg.device
        self._prompt_tokens = None
        self._cf_tokens = None
        self._mask = None
        self._cf_mask = None
        self._answer_tokens = None
        self._base_ldiff = None
        self._cf_ldiff = None

    @property
    def prompt_tokens(self) -> Int[Tensor, "batch seq_len"]:
        if self._prompt_tokens is None:
            self._prompt_tokens = self.model.to_tokens(self.prompts, prepend_bos=True)
        return self._prompt_tokens

    @property
    def cf_tokens(self) -> Int[Tensor, "batch seq_len"]:
        if self._cf_tokens is None:
            self._cf_tokens = self.model.to_tokens(self.cf_prompts, prepend_bos=True)
        return self._cf_tokens

    @property
    def mask(self) -> Int[Tensor, "batch seq_len"]:
        if self._mask is None:
            self._mask = get_attention_mask(
                self.model.tokenizer, self.prompt_tokens, prepend_bos=False
            )
        return self._mask

    @property
    def cf_mask(self) -> Int[Tensor, "batch seq_len"]:
        if self._cf_mask is None:
            self._cf_mask = get_attention_mask(
                self.model.tokenizer, self.cf_tokens, prepend_bos=False
            )
        return self._cf_mask

    @property
    def answer_tokens(self) -> Int[Tensor, "batch 2"]:
        if self._answer_tokens is None:
            self._answer_tokens = torch.tensor(
                [
                    (self.model.to_single_token(d[1]), self.model.to_single_token(d[3]))
                    for d in self
                ],
                device=self.device,
            )
        assert (self._answer_tokens[:, 0] != self._answer_tokens[:, 1]).all(), (
            f"Base answer {self._answer_tokens[:, 0]} and cf answer {self._answer_tokens[:, 1]} "
            f"must be different"
        )
        return self._answer_tokens

    @property
    def base_ldiff(self) -> Float[Tensor, "batch"]:
        if self._base_ldiff is None:
            base_logits = self.model(
                self.prompt_tokens, prepend_bos=False, return_type="logits"
            )
            self._base_ldiff = get_logit_diff(
                base_logits,
                answer_tokens=self.answer_tokens,
                per_prompt=True,
                mask=self.mask,
            )
        return self._base_ldiff

    @property
    def cf_ldiff(self) -> Float[Tensor, "batch"]:
        if self._cf_ldiff is None:
            cf_logits = self.model(
                self.cf_tokens, prepend_bos=False, return_type="logits"
            )
            assert isinstance(cf_logits, Tensor)
            self._cf_ldiff = get_logit_diff(
                cf_logits,
                answer_tokens=self.answer_tokens,
                per_prompt=True,
                mask=self.cf_mask,
            )
        return self._cf_ldiff

    @classmethod
    def from_tuples(
        cls, tuples: List[Tuple[str, str, str, str]], model: HookedTransformer
    ):
        """
        Accepts data in the form [
            (
                "Known for being the first to walk on the moon, Neil",
                " Armstrong",
                "Known for being the star of the movie Jazz Singer, Neil",
                " Diamond",
            ),
            ...
        ]
        """
        prompts = []
        answers = []
        cf_prompts = []
        cf_answers = []
        for prompt, answer, cf_prompt, cf_answer in tuples:
            assert prompt != cf_prompt, (
                f"Prompt {prompt} and counterfactual prompt {cf_prompt} "
                f"must be different"
            )
            assert answer != cf_answer, (
                f"Answer {answer} and counterfactual answer {cf_answer} "
                f"must be different"
            )
            prompts.append(prompt)
            answers.append(answer)
            cf_prompts.append(cf_prompt)
            cf_answers.append(cf_answer)
        return cls(
            prompts=prompts,
            answers=answers,
            cf_prompts=cf_prompts,
            cf_answers=cf_answers,
            model=model,
        )

    def __iter__(self):
        return iter(zip(self.prompts, self.answers, self.cf_prompts, self.cf_answers))

    def check_lengths_match(self):
        for prompt, _, cf_prompt, _ in self:
            prompt_str_tokens = self.model.to_str_tokens(prompt)
            cf_str_tokens = self.model.to_str_tokens(cf_prompt)
            assert len(prompt_str_tokens) == len(cf_str_tokens), (
                f"Prompt and counterfactual prompt must have the same length, "
                f"for prompt \n{prompt_str_tokens} \n and counterfactual\n{cf_str_tokens} \n"
                f"got {len(prompt_str_tokens)} and {len(cf_str_tokens)}"
            )

    def test_prompts(
        self,
        max_prompts: int = 4,
        top_k: int = 10,
        prepend_space_to_answer: bool | None = False,
        **kwargs,
    ):
        for i, (prompt, answer, cf_prompt, cf_answer) in enumerate(self):
            if i * 2 >= max_prompts:
                break
            test_prompt(
                prompt,
                answer,
                model=self.model,
                top_k=top_k,
                prepend_space_to_answer=prepend_space_to_answer,
                **kwargs,
            )
            test_prompt(
                cf_prompt,
                cf_answer,
                model=self.model,
                top_k=top_k,
                prepend_space_to_answer=prepend_space_to_answer,
                **kwargs,
            )

    def _compute_logit_diffs_loop(
        self,
    ) -> Tuple[Float[Tensor, "batch"], Float[Tensor, "batch"]]:
        all_logit_diffs = []
        cf_logit_diffs = []
        for prompt, answer, cf_prompt, cf_answer in self:
            prompt_tokens = self.model.to_tokens(prompt, prepend_bos=True)
            cf_tokens = self.model.to_tokens(cf_prompt, prepend_bos=True)
            answer_id = self.model.to_single_token(answer)
            cf_answer_id = self.model.to_single_token(cf_answer)
            answer_tokens = torch.tensor(
                [answer_id, cf_answer_id], dtype=torch.int64, device=self.device
            ).unsqueeze(0)
            assert prompt_tokens.shape == cf_tokens.shape, (
                f"Prompt and counterfactual prompt must have the same shape, "
                f"for prompt {prompt} "
                f"got {prompt_tokens.shape} and {cf_tokens.shape}"
            )
            self.model.reset_hooks()
            base_logits = self.model(
                prompt_tokens, prepend_bos=False, return_type="logits"
            )
            base_ldiff = get_logit_diff(base_logits, answer_tokens=answer_tokens)
            cf_logits = self.model(cf_tokens, prepend_bos=False, return_type="logits")
            assert isinstance(cf_logits, Tensor)
            cf_ldiff = get_logit_diff(cf_logits, answer_tokens=answer_tokens)
            all_logit_diffs.append(base_ldiff)
            cf_logit_diffs.append(cf_ldiff)
        all_logit_diffs = torch.stack(all_logit_diffs, dim=0)
        cf_logit_diffs = torch.stack(cf_logit_diffs, dim=0)
        return all_logit_diffs, cf_logit_diffs

    def _compute_logit_diffs_vectorized(
        self,
    ) -> Tuple[Float[Tensor, "batch"], Float[Tensor, "batch"]]:
        return self.base_ldiff, self.cf_ldiff

    def compute_logit_diffs(
        self,
        vectorized: bool = True,
    ) -> Tuple[Float[Tensor, "batch"], Float[Tensor, "batch"]]:
        if vectorized:
            return self._compute_logit_diffs_vectorized()
        else:
            return self._compute_logit_diffs_loop()

    def patch_by_layer(self) -> List[Float[np.ndarray, "layer pos"]]:
        results_list = []
        for prompt, answer, cf_prompt, cf_answer in self:
            prompt_results = patch_prompt_by_layer(
                prompt,
                answer,
                cf_prompt,
                cf_answer,
                model=self.model,
                prepend_bos=True,
            )
            results_list.append(prompt_results)
        return results_list

    def plot_layer_results_per_batch(
        self, results: List[Float[np.ndarray, "layer pos"]]
    ) -> go.Figure:
        fig = make_subplots(rows=len(results), cols=1)
        for row, (prompt, result) in enumerate(zip(self.prompts, results)):
            prompt_str_tokens = self.model.to_str_tokens(prompt)
            hm = go.Heatmap(
                z=result,
                x=[f"{i}: {t}" for i, t in enumerate(prompt_str_tokens)],
                y=[f"{i}" for i in range(self.model.cfg.n_layers)],
                colorscale="RdBu",
                zmin=0,
                zmax=1,
                hovertemplate="Layer %{y}<br>Position %{x}<br>Logit diff %{z}<extra></extra>",
            )
            fig.add_trace(hm, row=row, col=1)
        fig.update_layout(
            title_x=0.5,
            title=f"Patching metric by layer and position, {self.model.cfg.model_name}",
            xaxis_title="Position",
            yaxis_title="Layer",
            width=800,
            height=400 * len(results),
        )
        return fig

    def patch_by_position_group(self, sep=",") -> pd.Series:
        if self.base_ldiff.shape[0] == 0:
            self.compute_logit_diffs(vectorized=True)
        assert self.base_ldiff != self.cf_ldiff, (
            f"Base logit diff {self.base_ldiff} and cf logit diff {self.cf_ldiff} "
            f"must be different"
        )
        metric = lambda logits: (
            get_logit_diff(logits, answer_tokens=self.answer_tokens, mask=self.mask)
            - self.base_ldiff
        ) / (self.cf_ldiff - self.base_ldiff)
        pos_dict = get_position_dict(self.prompt_tokens, model=self.model, sep=sep)
        results_dict = dict()
        for pos_label, positions in pos_dict.items():
            positions = pos_dict[pos_label]
            nodes = [
                Node(node_name="resid_pre", layer=layer, seq_pos=positions)
                for layer in range(self.model.cfg.n_layers)
            ]
            pos_results = act_patch(
                self.model, self.prompt_tokens, nodes, metric, new_input=self.cf_tokens, verbose=True  # type: ignore
            ).item()
            results_dict[pos_label] = pos_results
        return pd.Series(results_dict)

    def patch_at_position(
        self, positions: List[int] | List[List[int]], verbose: bool = True
    ) -> Float[Tensor, "batch"]:
        results = []
        for batch, (prompt, answer, cf_prompt, cf_answer) in enumerate(self):
            prompt_tokens = self.model.to_tokens(prompt, prepend_bos=True)
            cf_tokens = self.model.to_tokens(cf_prompt, prepend_bos=True)
            answer_id = self.model.to_single_token(answer)
            cf_answer_id = self.model.to_single_token(cf_answer)
            answer_tokens = torch.tensor(
                [answer_id, cf_answer_id], dtype=torch.int64, device=self.device
            ).unsqueeze(0)
            assert prompt_tokens.shape == cf_tokens.shape, (
                f"Prompt and counterfactual prompt must have the same shape, "
                f"for prompt {prompt} "
                f"got {prompt_tokens.shape} and {cf_tokens.shape}"
            )
            self.model.reset_hooks()
            base_logits = self.model(
                prompt_tokens, prepend_bos=False, return_type="logits"
            )
            base_ldiff = get_logit_diff(base_logits, answer_tokens=answer_tokens)
            cf_logits, cf_cache = self.model.run_with_cache(
                cf_tokens, prepend_bos=False, return_type="logits"
            )
            assert isinstance(cf_logits, Tensor)
            cf_ldiff = get_logit_diff(cf_logits, answer_tokens=answer_tokens)
            metric = lambda logits: (
                get_logit_diff(logits, answer_tokens=answer_tokens) - base_ldiff
            ) / (cf_ldiff - base_ldiff)
            nodes = [
                Node(node_name="resid_pre", layer=layer, seq_pos=positions[batch])
                for layer in range(self.model.cfg.n_layers)
            ]
            result = act_patch(
                self.model,
                prompt_tokens,
                nodes,
                metric,
                new_cache=cf_cache,
                verbose=verbose,
            )
            results.append(result)
        results = torch.stack(results, dim=0)
        return results


class KnownForDataset(CounterfactualDataset):
    def __init__(self, model: HookedTransformer) -> None:
        super().from_tuples(KNOWN_FOR_TUPLES, model)


class OfCourseDataset(CounterfactualDataset):
    def __init__(self, model: HookedTransformer) -> None:
        super().from_tuples(OF_COURSE_TUPLES, model)
