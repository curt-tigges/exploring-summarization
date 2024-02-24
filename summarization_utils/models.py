import torch
from typing import List, Literal
from transformer_lens import HookedTransformer
from transformer_lens.utils import USE_DEFAULT_VALUE


class TokenSafeTransformer(HookedTransformer):
    def to_tokens(
        self,
        string: str | List[str],
        prepend_bos=USE_DEFAULT_VALUE,
        padding_side: Literal["left", "right"] | None = USE_DEFAULT_VALUE,
        move_to_device: bool | None = True,
        truncate: bool | None = True,
        prepend_blank: bool = True,
        verbose: bool = False,
        blank_char: str = "|",
    ):
        """Map a string to the ids for that string.

        Handles the case of mistral which has an extra BOS-like token prepended to the start
        of the output of tokenizer.encode().
        """
        if "mistral" not in self.cfg.model_name.lower() or not prepend_blank:
            return super().to_tokens(
                string,
                prepend_bos=prepend_bos,
                padding_side=padding_side,
                move_to_device=move_to_device,
                truncate=truncate,
            )
        if prepend_bos is USE_DEFAULT_VALUE:
            prepend_bos = self.cfg.default_prepend_bos
        if isinstance(string, str):
            string = blank_char + string
        else:
            string = [blank_char + s for s in string]

        tokens = super().to_tokens(
            string,
            prepend_bos=True,
            padding_side=padding_side,
            move_to_device=move_to_device,
            truncate=truncate,
        )
        if verbose:
            print(f"Tokenizing string={string}")
            print(f"Tokens={tokens}")
            print(f"verbose={verbose}")
        # Remove the artificial token
        if prepend_bos:
            return torch.cat([tokens[:, :1], tokens[:, 2:]], dim=1)
        else:
            return tokens[:, 2:]
