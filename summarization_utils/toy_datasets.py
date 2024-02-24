import itertools
import random
from jaxtyping import Float, Int
from typing import List, Optional, Tuple
import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.utils import test_prompt, get_attention_mask
from summarization_utils.patching_metrics import get_logit_diff
from abc import ABC, abstractmethod


PYTHIA_KNOWN_FOR = [
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
        "Known for being a wonder of the world located in Australia, the",
        " Great",
        "Known for being a wonder of the world located in India, the",
        " Taj",
    ),
    (
        "Known for being the most popular sport in Brazil, the game of",
        " soccer",
        "Known for being the most popular sport in India, the game of",
        " cricket",
    ),
]


MISTRAL_KNOWN_FOR = [
    (
        "Known for being the first to walk on the moon, Neil",
        " Arm",
        "Known for being the star of movie Jazz Singer, Neil",
        " Diamond",
    ),
    (
        "Known as first to cross Antarctica, Sir",
        " Ernest",
        "Known for being the first to summit Everest, Sir",
        " Ed",
    ),
    (
        "Known for being the fastest production car in the world, the",
        " Mc",
        "Known for being the best selling car in the world, the",
        " Ford",
    ),
    (
        "Known for being the most popular fruit in the world, the humble",
        " apple",
        "Known for being the most popular crop in the world, the humble",
        " pot",
    ),
    (
        "Known for being a wonder of the world located in Australia, the",
        " Great",
        "Known for being a wonder of the world located in India, the",
        " T",
    ),
    (
        "Known for being the most popular sport in Brazil, the game of",
        " soccer",
        "Known for being the most popular sport in India, the game of",
        " cricket",
    ),
]

PYTHIA_OF_COURSE = [
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


MISTRAL_OF_COURSE = [
    (
        "The first to walk on the moon is of course, Neil",
        " Arm",
        "The star of movie Jazz Singer is of course, Neil",
        " Diamond",
    ),
    (
        "First across Antarctica was of course, Sir",
        " Ernest",
        "The first to summit Everest was of course, Sir",
        " Ed",
    ),
    (
        "The fastest production car in the world is of course, the",
        " Mc",
        "The best selling car in the world is of course, the",
        " Ford",
    ),
    (
        "The most popular fruit in the world is of course, the humble",
        " apple",
        "The most popular crop in the world is of course, the humble",
        " pot",
    ),
    (
        "The most popular sport in Brazil is of course, the game of",
        " soccer",
        "The most popular sport in India is of course, the game of",
        " cricket",
    ),
]

SANTACODER_CODE = [
    (
        "x = 0\nprint(x) # ",
        "0",
        "x = 1\nprint(x) # ",
        "1",
    ),
    (
        "x = 0\n x += 1\nprint(x) # ",
        "1",
        "x = 1\n x += 1\nprint(x) # ",
        "2",
    ),
    (
        "x = 'Hello World'\nprint(x) #",
        " Hello",
        "x = 'Hi Sys'\nprint(x) #",
        " Hi",
    ),
    (
        "x = 'Hello World'\nx = x.upper()\nx = x.lower\nprint(x) #",
        " hello",
        "x = 'Hi Sys'\nx = x.upper()\nx = x.lower\nprint(x) #",
        " hi",
    ),
    (
        "x = 'Hello World'\nprint(x) # Hello World\nx = x.upper()\nprint(x) #",
        " HEL",
        "x = 'Hi Sys'\nprint(x) # Hi Sys\nx = x.upper()\nprint(x) #",
        " H",
    ),
    (
        "x = 'Hello World'\nprint(x) # Hello World\nx = x.upper()\nprint(x) # HELLO WORLD\nx = x.lower()\nprint(x) #",
        " hello",
        "x = 'Hi Sys'\nprint(x) # Hi Sys\nx = x.upper()\nprint(x) # HI SYS\nx = x.lower()\nprint(x) #",
        " hi",
    ),
    (
        "x = 'Hello World'\nprint(x) # Hello World\nx = x.upper()\nprint(x) # HELLO WORLD\nx = x.lower()\nprint(x) # hello world\nx *= 2\nprint(x) # hello worldhello world\nx = x.split()[0]\nprint(x) #",
        " hello",
        "x = 'Hi Sys'\nprint(x) # Hi Sys\nx = x.upper()\nprint(x) # HI SYS\nx = x.lower()\nprint(x) # hi sys\nx *= 2\nprint(x) # hi syshi sys\nx = x.split()[0]\nprint(x) #",
        " hi",
    ),
    (
        "def print_first_n_even_numbers(n: int) -> None:\n    for num in range(1, n + 1):\n        if num % 2 == ",
        "0",
        "def print_first_n_odd_numbers(n: int) -> None:\n    for num in range(1, n + 1):\n        if num % 2 == ",
        "1",
    ),
    (
        "def print_first_n_factorial_inorder(n: int) -> None:\n    x = 1\n    for num in range(1, n + 1):\n        x = x",
        " *",
        "def print_first_n_triangular_numbers(n: int) -> None:\n    x = 0\n    for num in range(1, n + 1):\n        x = x",
        " +",
    ),
    (
        "def print_first_n_multiples_of_3(n: int) -> None:\n    for num in range(1, n):\n        print(num * ",
        "3",
        "def print_first_n_multiples_of_5(n: int) -> None:\n    for num in range(1, n):\n        print(num * ",
        "5",
    ),
    (
        "def print_first_n_composites(n: int) -> None:\n    for num in range(2, n):\n        if num > 1:\n            for i in range(2, num):\n                if (num % i) == 0:\n                    ",
        " print",
        "def print_first_n_prime_numbers(n: int) -> None:\n    for num in range(2, n):\n        if num > 1:\n            for i in range(2, num):\n                if (num % i) == 0:\n                    ",
        " break",
    ),
    (
        "def count_words(string: str) -> int:\n    return len(string.",
        "split",
        "def count_lines(string: str) -> int:\n    return len(string.",
        "splitlines",
    ),
    (
        "def reverseorder_string(string: str) -> str:\n    return string",
        "[::-",
        "def halve_string(string: str) -> str:\n    return string",
        "[:",
    ),
    (
        "def is_uppercase(string: str) -> bool:\n    return string.is",
        "upper",
        "def is_lowercase(string: str) -> bool:\n    return string.is",
        "lower",
    ),
    (
        "def is_uppercase(string: str) -> bool:\n    # Check if string is in all caps using python's builtin isupper() method\n    return string.is",
        "upper",
        "def is_lowercase(string: str) -> bool:\n    # Check if string is in lower case using python's builtin islower() method\n    return string.is",
        "lower",
    ),
    (
        "def is_right_case(string: str) -> bool:\n    # Check if string is in all caps using python's builtin isupper() method\n    # This function will be useful later\n    return string.is",
        "upper",
        "def is_right_case(string: str) -> bool:\n    # Check if string is in lower case using python's builtin islower() method\n    # This function will be useful later\n    return string.is",
        "lower",
    ),
    (
        "def convert_to_celsius(temp: float) -> float:\n    return (temp",
        " -",
        "def convert_to_fahrenheit(temp: float) -> float:\n    return (temp",
        " *",
    ),
    (
        "def Factorial(n: int) -> int\n    if n < 2:\n        return 1\n    else:\n        return",
        " n",
        "def fibonacci(n: int) -> int\n    if n < 2:\n        return 1\n    else:\n        return",
        " fib",
    ),
    (
        "def find_min(array: List[int]) -> int:\n    return",
        " min",
        "def find_max(array: List[int]) -> int:\n    return",
        " max",
    ),
    (
        "def calculate_mean(array: List[int]) -> float:\n    return",
        " sum",
        "def calculate_mode(array: List[int]) -> float:\n    return",
        " max",
    ),
]


def wrap_instruction(instruction: str, model: HookedTransformer):
    if "mistral-7b-instruct" in model.cfg.model_name.lower():
        return f"[INST] {instruction} [/INST]"
    elif "qwen" in model.cfg.model_name.lower() and "chat" in model.cfg.model_name:
        return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    elif "instruct" in model.cfg.model_name or "chat" in model.cfg.model_name:
        raise NotImplementedError(
            f"Model {model.cfg.model_name} does not support instructions"
        )
    else:
        return instruction


class TemplaticDataset(ABC):
    def __init__(
        self,
        template: str,
        prompt_tuples: List[Tuple[str, ...]],
        model: HookedTransformer,
        dataset_size: Optional[int] = None,
    ) -> None:
        self.template = template
        self.prompt_tuples = prompt_tuples[:dataset_size]
        self._cf_tuples = []
        self.model = model

    @abstractmethod
    def get_counterfactual_tuples(self) -> List[Tuple[str, ...]]:
        pass

    @classmethod
    @abstractmethod
    def get_answers(cls, prompt_tuples: List[Tuple[str, ...]]) -> List[str]:
        pass

    @staticmethod
    @abstractmethod
    def format_prompts(prompts: List[Tuple[str, ...]]) -> List[str]:
        pass

    @property
    def cf_tuples(self) -> List[Tuple[str, ...]]:
        if not self._cf_tuples:
            self._cf_tuples = self.get_counterfactual_tuples()
        return self._cf_tuples

    @property
    def prompts(self) -> List[str]:
        return self.format_prompts(self.prompt_tuples)

    @property
    def answers(self) -> List[str]:
        return self.get_answers(self.prompt_tuples)

    @property
    def cf_prompts(self) -> List[str]:
        return self.format_prompts(self.cf_tuples)

    @property
    def cf_answers(self) -> List[str]:
        return self.get_answers(self.cf_tuples)

    def to_counterfactual(self):
        return CounterfactualDataset(
            prompts=self.prompts,
            answers=self.answers,
            cf_prompts=self.cf_prompts,
            cf_answers=self.cf_answers,
            model=self.model,
        )


class BooleanNegatorDataset(TemplaticDataset):
    NAMES = [
        "Anne",
        "Bob",
        "Carol",
        "David",
        "Emma",
        "Mike",
        "Sarah",
        "John",
        "Linda",
        "Peter",
        "Grace",
        "Oliver",
        "Sophie",
        "Josh",
        "Tom",
        "Rachel",
        "Henry",
        "Alice",
        "George",
    ]
    POSITIVE_ATTRIBUTES = [
        "loud",
        "fast",
        "tall",
        "fat",
        "young",
        "strong",
        "smart",
        "happy",
        "kind",
        "funny",
        "curious",
        "calm",
        "pretty",
    ]
    NEGATIVE_ATTRIBUTES = [
        "quiet",
        "slow",
        "short",
        "thin",
        "old",
        "weak",
        "dumb",
        "sad",
        "mean",
        "serious",
        "dull",
        "nervous",
        "ugly",
    ]

    @classmethod
    def get_attribute_sign_and_index(cls, attr: str) -> Tuple[bool, int]:
        if attr in cls.POSITIVE_ATTRIBUTES:
            return True, cls.POSITIVE_ATTRIBUTES.index(attr)
        elif attr in cls.NEGATIVE_ATTRIBUTES:
            return False, cls.NEGATIVE_ATTRIBUTES.index(attr)
        else:
            raise ValueError(f"Unknown attribute {attr}")

    def __init__(
        self,
        model: HookedTransformer,
        dataset_size: int = 100,
        seed: int = 0,
    ) -> None:
        template = wrap_instruction(
            "Question: "
            "{NAME} is {ATTR1}. {NAME} is {ATTR2}. {NAME} is {ATTR3}. Is {NAME} {ATTR_R}?"
            " Answer (Yes/No):",
            model,
        )
        prompt_tuples = [
            (
                name,
                attr1_list[attr1_idx],
                attr2_list[attr2_idx],
                attr3_list[attr3_idx],
                attr_r,
            )
            for name in self.NAMES
            for attr1_idx, attr2_idx, attr3_idx in itertools.combinations(
                range(len(self.POSITIVE_ATTRIBUTES)), 3
            )
            for attr1_list in [self.POSITIVE_ATTRIBUTES, self.NEGATIVE_ATTRIBUTES]
            for attr2_list in [self.POSITIVE_ATTRIBUTES, self.NEGATIVE_ATTRIBUTES]
            for attr3_list in [self.POSITIVE_ATTRIBUTES, self.NEGATIVE_ATTRIBUTES]
            for attr_r in [
                self.POSITIVE_ATTRIBUTES[attr1_idx],
                self.POSITIVE_ATTRIBUTES[attr2_idx],
                self.POSITIVE_ATTRIBUTES[attr3_idx],
            ]
        ]
        super().__init__(template, prompt_tuples, model, dataset_size=dataset_size)
        self.seed = seed

    def get_counterfactual_tuples(self) -> List[Tuple[str, ...]]:
        random.seed(self.seed)
        cf_tuples = []
        for name, attr1, attr2, attr3, attr_r in self.prompt_tuples:
            # Flip the sign of the one of the three attributes which matches attr_r
            _, attr_r_idx = self.get_attribute_sign_and_index(attr_r)
            idx_to_change, attr_to_change = [
                (i, attr)
                for i, attr in enumerate((attr1, attr2, attr3))
                if self.get_attribute_sign_and_index(attr) == attr_r_idx
            ][0]
            attr_idx, attr_sign = self.get_attribute_sign_and_index(attr_to_change)
            cf_attr = (
                self.POSITIVE_ATTRIBUTES[attr_idx]
                if not attr_sign
                else self.NEGATIVE_ATTRIBUTES[attr_idx]
            )
            cf_attr1, cf_attr2, cf_attr3 = (
                cf_attr if idx_to_change == 0 else attr1,
                cf_attr if idx_to_change == 1 else attr2,
                cf_attr if idx_to_change == 2 else attr3,
            )
            cf_tuples.append((name, cf_attr1, cf_attr2, cf_attr3, attr_r))
        return cf_tuples

    @classmethod
    def get_answers(cls, prompt_tuples: List[Tuple[str, ...]]) -> List[str]:
        answers = []
        for _, attr1, attr2, attr3, attr_r in prompt_tuples:
            attr1_sign, attr1_idx = cls.get_attribute_sign_and_index(attr1)
            attr2_sign, attr2_idx = cls.get_attribute_sign_and_index(attr2)
            attr3_sign, attr3_idx = cls.get_attribute_sign_and_index(attr3)
            _, attr_r_idx = cls.get_attribute_sign_and_index(attr_r)
            if attr_r_idx == attr2_idx:
                answer = attr2_sign
            elif attr_r_idx == attr1_idx:
                answer = attr1_sign
            elif attr_r_idx == attr3_idx:
                answer = attr3_sign
            else:
                raise ValueError(f"Unknown attribute {attr_r}")
            answers.append(" Yes" if answer else " No")
        return answers

    def format_prompts(self, prompt_tuples: List[Tuple[str, ...]]) -> List[str]:
        return [
            self.template.format(
                NAME=name,
                ATTR1=attr1,
                ATTR2=attr2,
                ATTR3=attr3,
                ATTR_R=attr_r,
            )
            for name, attr1, attr2, attr3, attr_r in prompt_tuples
        ]


class BooleanOperatorDataset(TemplaticDataset):
    NAMES = [
        "Anne",
        "Bob",
        "Carol",
        "David",
        "Emma",
        "Mike",
        "Sarah",
        "John",
        "Linda",
        "Peter",
        "Grace",
        "Oliver",
        "Sophie",
        "Josh",
        "Tom",
        "Rachel",
        "Henry",
        "Alice",
        "George",
    ]
    POSITIVE_ATTRIBUTES = [
        "loud",
        "fast",
        "tall",
        "fat",
        "young",
        "strong",
        "smart",
        "happy",
        "kind",
        "funny",
        "curious",
        "calm",
        "pretty",
    ]
    NEGATIVE_ATTRIBUTES = [
        "quiet",
        "slow",
        "short",
        "thin",
        "old",
        "weak",
        "dumb",
        "sad",
        "mean",
        "serious",
        "dull",
        "nervous",
        "ugly",
    ]
    OPERATORS = [
        "and",
        "or",
    ]

    @classmethod
    def get_attribute_sign_and_index(cls, attr: str) -> Tuple[bool, int]:
        if attr in cls.POSITIVE_ATTRIBUTES:
            return True, cls.POSITIVE_ATTRIBUTES.index(attr)
        elif attr in cls.NEGATIVE_ATTRIBUTES:
            return False, cls.NEGATIVE_ATTRIBUTES.index(attr)
        else:
            raise ValueError(f"Unknown attribute {attr}")

    def __init__(
        self,
        model: HookedTransformer,
        dataset_size: int = 100,
        seed: int = 0,
    ) -> None:
        template = wrap_instruction(
            "Question: "
            "{NAME} is {ATTR1}. {NAME} is {ATTR2}. {NAME} is {ATTR3}. Is {NAME} {ATTR_L} {OPERATOR} {ATTR_R}?"
            " Answer (Yes/No):",
            model,
        )
        prompt_tuples = [
            (
                name,
                attr1_list[attr1_idx],
                attr2_list[attr2_idx],
                attr3_list[attr3_idx],
                attr_l,
                operator,
                attr_r,
            )
            for name in self.NAMES
            for operator in self.OPERATORS
            for attr1_idx, attr2_idx, attr3_idx in itertools.combinations(
                range(len(self.POSITIVE_ATTRIBUTES)), 3
            )
            for attr1_list in [self.POSITIVE_ATTRIBUTES, self.NEGATIVE_ATTRIBUTES]
            for attr2_list in [self.POSITIVE_ATTRIBUTES, self.NEGATIVE_ATTRIBUTES]
            for attr3_list in [self.POSITIVE_ATTRIBUTES, self.NEGATIVE_ATTRIBUTES]
            for attr_l, attr_r in [
                (
                    self.POSITIVE_ATTRIBUTES[attr1_idx],
                    self.POSITIVE_ATTRIBUTES[attr2_idx],
                ),
                (
                    self.POSITIVE_ATTRIBUTES[attr2_idx],
                    self.POSITIVE_ATTRIBUTES[attr1_idx],
                ),
                (
                    self.POSITIVE_ATTRIBUTES[attr1_idx],
                    self.POSITIVE_ATTRIBUTES[attr3_idx],
                ),
                (
                    self.POSITIVE_ATTRIBUTES[attr3_idx],
                    self.POSITIVE_ATTRIBUTES[attr1_idx],
                ),
                (
                    self.POSITIVE_ATTRIBUTES[attr2_idx],
                    self.POSITIVE_ATTRIBUTES[attr3_idx],
                ),
                (
                    self.POSITIVE_ATTRIBUTES[attr3_idx],
                    self.POSITIVE_ATTRIBUTES[attr2_idx],
                ),
            ]
        ]
        super().__init__(template, prompt_tuples, model, dataset_size=dataset_size)
        self.seed = seed

    @classmethod
    def get_answer(cls, attr1, attr2, attr3, attr_l, operator, attr_r) -> bool:
        attr1_sign, attr1_idx = cls.get_attribute_sign_and_index(attr1)
        attr2_sign, attr2_idx = cls.get_attribute_sign_and_index(attr2)
        attr3_sign, attr3_idx = cls.get_attribute_sign_and_index(attr3)
        _, attr_l_idx = cls.get_attribute_sign_and_index(attr_l)
        _, attr_r_idx = cls.get_attribute_sign_and_index(attr_r)
        if operator == "and":
            if attr_l_idx == attr1_idx and attr_r_idx == attr2_idx:
                answer = attr1_sign and attr2_sign
            elif attr_l_idx == attr2_idx and attr_r_idx == attr1_idx:
                answer = attr1_sign and attr2_sign
            elif attr_l_idx == attr1_idx and attr_r_idx == attr3_idx:
                answer = attr1_sign and attr3_sign
            elif attr_l_idx == attr3_idx and attr_r_idx == attr1_idx:
                answer = attr1_sign and attr3_sign
            elif attr_l_idx == attr2_idx and attr_r_idx == attr3_idx:
                answer = attr2_sign and attr3_sign
            elif attr_l_idx == attr3_idx and attr_r_idx == attr2_idx:
                answer = attr2_sign and attr3_sign
            else:
                raise ValueError(
                    f"Invalid combination of attributes {attr_l} and {attr_r}"
                )
        elif operator == "or":
            if attr_l_idx == attr1_idx and attr_r_idx == attr2_idx:
                answer = attr1_sign or attr2_sign
            elif attr_l_idx == attr2_idx and attr_r_idx == attr1_idx:
                answer = attr1_sign or attr2_sign
            elif attr_l_idx == attr1_idx and attr_r_idx == attr3_idx:
                answer = attr1_sign or attr3_sign
            elif attr_l_idx == attr3_idx and attr_r_idx == attr1_idx:
                answer = attr1_sign or attr3_sign
            elif attr_l_idx == attr2_idx and attr_r_idx == attr3_idx:
                answer = attr2_sign or attr3_sign
            elif attr_l_idx == attr3_idx and attr_r_idx == attr2_idx:
                answer = attr2_sign or attr3_sign
            else:
                raise ValueError(
                    f"Invalid combination of attributes {attr_l} and {attr_r}"
                )
        else:
            raise ValueError(f"Unknown operator {operator}")
        return answer

    def get_counterfactual_tuples(self) -> List[Tuple[str, ...]]:
        # We try flipping the sign of each of the three attributes until we find one
        # which flips the answer
        random.seed(self.seed)
        cf_tuples = []
        for name, attr1, attr2, attr3, attr_l, operator, attr_r in self.prompt_tuples:
            orig_answer = self.get_answer(attr1, attr2, attr3, attr_l, operator, attr_r)
            indices = [0, 1, 2]
            random.shuffle(indices)
            for idx_to_change in indices:
                attr_sign, attr_idx = self.get_attribute_sign_and_index(
                    [attr1, attr2, attr3][idx_to_change]
                )
                cf_attr = (
                    self.POSITIVE_ATTRIBUTES[attr_idx]
                    if not attr_sign
                    else self.NEGATIVE_ATTRIBUTES[attr_idx]
                )
                cf_attr1, cf_attr2, cf_attr3 = (
                    cf_attr if idx_to_change == 0 else attr1,
                    cf_attr if idx_to_change == 1 else attr2,
                    cf_attr if idx_to_change == 2 else attr3,
                )
                cf_answer = self.get_answer(
                    cf_attr1, cf_attr2, cf_attr3, attr_l, operator, attr_r
                )
                if orig_answer != cf_answer:
                    break
            else:
                raise ValueError(
                    f"Could not find a counterfactual for {name} {attr1} {attr2} {attr3} {attr_l} {operator} {attr_r}"
                )
            cf_tuples.append(
                (name, cf_attr1, cf_attr2, cf_attr3, attr_l, operator, attr_r)
            )
        return cf_tuples

    @classmethod
    def get_answers(cls, prompt_tuples: List[Tuple[str, ...]]) -> List[str]:
        answers = []
        for _, attr1, attr2, attr3, attr_l, operator, attr_r in prompt_tuples:
            answer = cls.get_answer(attr1, attr2, attr3, attr_l, operator, attr_r)
            answers.append(" Yes" if answer else " No")
        return answers

    def format_prompts(self, prompt_tuples: List[Tuple[str, ...]]) -> List[str]:
        return [
            self.template.format(
                NAME=name,
                ATTR1=attr1,
                ATTR2=attr2,
                ATTR3=attr3,
                ATTR_L=attr_l,
                OPERATOR=operator,
                ATTR_R=attr_r,
            )
            for name, attr1, attr2, attr3, attr_l, operator, attr_r in prompt_tuples
        ]


class ToyBindingTemplate(TemplaticDataset):
    NAMES = [
        "Anne",
        "Bob",
        "Carol",
        "David",
        "Emma",
        "Mike",
        "Sarah",
        "John",
        "Linda",
        "Peter",
        "Grace",
        "Oliver",
        "Sophie",
        "Josh",
        "Tom",
        "Rachel",
        "Henry",
        "Alice",
        "George",
    ]
    OBJECTS = [
        "car",
        "bike",
        "house",
        "boat",
        "plane",
        "computer",
        "phone",
        "book",
        "pen",
        "ball",
        "toy",
        "game",
        "shirt",
    ]

    def __init__(
        self,
        model: HookedTransformer,
        dataset_size: int = 100,
        seed: int = 0,
    ) -> None:
        template = (
            wrap_instruction(
                "{NAME_L} likes {OBJECT_L}. {NAME_R} likes {OBJECT_R}. Who does the {OBJECT_Q} belong to?",
                model,
            )
            + "The {OBJECT_Q} belongs to"
        )
        prompt_tuples = [
            (name_l, object_l, name_r, object_r, object_q)
            for name_l, name_r in itertools.combinations(self.NAMES, 2)
            for object_l, object_r in itertools.combinations(self.OBJECTS, 2)
            for object_q in (object_l, object_r)
        ]
        super().__init__(template, prompt_tuples, model, dataset_size=dataset_size)
        self.seed = seed

    def get_counterfactual_tuples(self) -> List[Tuple[str, ...]]:
        # Just swap the left and right objects
        return [
            (name_l, object_r, name_r, object_l, object_q)
            for name_l, object_l, name_r, object_r, object_q in self.prompt_tuples
        ]

    @classmethod
    def get_answers(cls, prompt_tuples: List[Tuple[str, ...]]):
        return [
            " " + (name_l if object_l == object_q else name_r)
            for name_l, object_l, name_r, object_r, object_q in prompt_tuples
        ]

    def format_prompts(self, prompt_tuples: List[Tuple[str, ...]]):
        return [
            self.template.format(
                NAME_L=name_l,
                OBJECT_L=object_l + "s",
                NAME_R=name_r,
                OBJECT_R=object_r + "s",
                OBJECT_Q=object_q,
            )
            for name_l, object_l, name_r, object_r, object_q in prompt_tuples
        ]


class ToyDeductionTemplate(TemplaticDataset):
    NAMES = [
        "Anne",
        "Bob",
        "Carol",
        "David",
        "Emma",
        "Mike",
        "Sarah",
        "John",
        "Linda",
        "Peter",
        "Grace",
        "Oliver",
        "Sophie",
        "Josh",
        "Mia",
        "Tom",
        "Rachel",
        "Henry",
        "Alice",
        "George",
    ]
    GROUPS = [
        "dog",
        "cat",
        "bird",
        "hamster",
        "rabbit",
        "Capricorn",
        "Scorpio",
        "Leo",
        "Cancer",
        "Gemini",
    ]
    ATTRIBUTES = [
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "black",
        "white",
        "brown",
        "grey",
        "tall",
        "short",
        "young",
        "old",
        "smart",
        "dumb",
        "rich",
        "poor",
        "happy",
        "sad",
    ]

    def __init__(
        self,
        model: HookedTransformer,
        dataset_size: int = 100,
        seed: int = 0,
    ) -> None:
        template = (
            wrap_instruction(
                "{NAME} is a {GROUP}. {CAPITAL_GROUP}s are {ATTR}. Conclusion?", model
            )
            + "Therefore, {NAME} is"
        )
        prompt_tuples = list(
            itertools.product(
                self.NAMES,
                self.GROUPS,
                self.ATTRIBUTES,
            )
        )
        random.seed(seed)
        random.shuffle(prompt_tuples)
        super().__init__(template, prompt_tuples, model, dataset_size=dataset_size)
        self.seed = seed

    def get_counterfactual_tuples(self) -> List[Tuple[str, ...]]:
        random.seed(self.seed)
        cf_tuples = []
        for name, group, attr in self.prompt_tuples:
            attr_idx = self.ATTRIBUTES.index(attr)
            new_attr = self.ATTRIBUTES[(attr_idx + 1) % len(self.ATTRIBUTES)]
            cf_tuples.append((name, group, new_attr))
        return cf_tuples

    @classmethod
    def get_answers(cls, prompt_tuples: List[Tuple[str, ...]]):
        return [" " + attr for _, _, attr in prompt_tuples]

    def format_prompts(self, prompt_tuples: List[Tuple[str, ...]]):
        return [
            self.template.format(
                NAME=name,
                ATTR=attr,
                GROUP=group,
                CAPITAL_GROUP=group.capitalize(),
            )
            for name, group, attr in prompt_tuples
        ]


class ToyProfilesTemplate(TemplaticDataset):
    QUERIES = [
        "Nationality",
        "Occupation",
    ]
    CONJUGATIONS = [
        "is from the country of",
        "is a",
    ]
    QUERY_TO_CONJ = dict(zip(QUERIES, CONJUGATIONS))
    NAMES = [
        "Anne",
        "Bob",
        "Carol",
        "David",
        "Emma",
        "Mike",
        "Sarah",
        "John",
        "Linda",
        "Peter",
        "Grace",
        "Oliver",
        "Sophie",
        "Josh",
        "Tom",
        "Rachel",
        "Henry",
        "Alice",
        "George",
    ]
    CITIES = [
        "Paris",
        "London",
        "Berlin",
        "Tokyo",
        "Moscow",
        "Beijing",
        "Sydney",
        "Madrid",
        "Rome",
    ]
    COUNTRIES = [
        "France",
        "England",
        "Germany",
        "Japan",
        "Russia",
        "China",
        "Australia",
        "Spain",
        "Italy",
    ]
    CITY_TO_NATIONALITY = dict(zip(CITIES, COUNTRIES))
    JOBS = [
        "teacher",
        "doctor",
        "lawyer",
        "scientist",
        "writer",
        "singer",
        "dancer",
        "dentist",
        "pilot",
    ]

    def __init__(
        self,
        model: HookedTransformer,
        dataset_size: int = 100,
        seed: int = 0,
    ) -> None:
        template = (
            wrap_instruction(
                "Profile: {NAME} was born in {CITY}. {NAME} works as a {JOB}. What is their {QUERY}?",
                model,
            )
            + "{QUERY}: {NAME} {CONJ}"
        )
        prompt_tuples = list(
            itertools.product(self.NAMES, self.CITIES, self.JOBS, self.QUERIES)
        )
        random.seed(seed)
        random.shuffle(prompt_tuples)
        super().__init__(template, prompt_tuples, model, dataset_size=dataset_size)
        self.seed = seed

    def get_counterfactual_tuples(self) -> List[Tuple[str, ...]]:
        cf_tuples = []
        for _, (name, city, job, query) in enumerate(self.prompt_tuples):
            if query == "Nationality":
                new_city = random.choice([c for c in self.CITIES if c != city])
                cf_tuples.append((name, new_city, job, query))
            elif query == "Occupation":
                new_job = random.choice([j for j in self.JOBS if j != job])
                cf_tuples.append((name, city, new_job, query))
            else:
                raise ValueError(f"Unknown query {query}")
        return cf_tuples

    @classmethod
    def get_answers(cls, prompt_tuples: List[Tuple[str, ...]]):
        answers = []
        for _, city, job, query in prompt_tuples:
            if query == "Nationality":
                answers.append(" " + cls.CITY_TO_NATIONALITY[city])
            elif query == "Occupation":
                answers.append(" " + job)
        return answers

    def format_prompts(self, prompt_tuples: List[Tuple[str, ...]]):
        return [
            self.template.format(
                NAME=name,
                CITY=city,
                JOB=job,
                QUERY=query,
                CONJ=self.QUERY_TO_CONJ[query],
            )
            for name, city, job, query in prompt_tuples
        ]


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

    def __add__(self, other: "CounterfactualDataset"):
        return CounterfactualDataset(
            prompts=self.prompts + other.prompts,
            answers=self.answers + other.answers,
            cf_prompts=self.cf_prompts + other.cf_prompts,
            cf_answers=self.cf_answers + other.cf_answers,
            model=self.model,
        )

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # If the index is a slice, return a new CounterfactualDataset with the sliced data
            return CounterfactualDataset(
                prompts=self.prompts[idx],
                answers=self.answers[idx],
                cf_prompts=self.cf_prompts[idx],
                cf_answers=self.cf_answers[idx],
                model=self.model,
            )
        else:
            # If the index is an integer, return a tuple with the data at that index
            return (
                self.prompts[idx],
                self.answers[idx],
                self.cf_prompts[idx],
                self.cf_answers[idx],
            )

    def shuffle(self, seed: int = 0) -> "CounterfactualDataset":
        random.seed(seed)
        index = list(range(len(self)))
        random.shuffle(index)
        return CounterfactualDataset(
            prompts=[self.prompts[i] for i in index],
            answers=[self.answers[i] for i in index],
            cf_prompts=[self.cf_prompts[i] for i in index],
            cf_answers=[self.cf_answers[i] for i in index],
            model=self.model,
        )

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

    @classmethod
    def empty(cls, model: HookedTransformer):
        return cls(
            prompts=[],
            answers=[],
            cf_prompts=[],
            cf_answers=[],
            model=model,
        )

    @classmethod
    def from_name(cls, name: str, model: HookedTransformer, **kwargs):
        assert model.tokenizer is not None
        is_pythia = "pythia" in model.tokenizer.name_or_path
        is_mistral = "mistral" in model.tokenizer.name_or_path
        is_santacoder = "santacoder" in model.tokenizer.name_or_path
        if name == "KnownFor" and is_pythia:
            return cls.from_tuples(PYTHIA_KNOWN_FOR, model)
        elif name == "KnownFor" and is_mistral:
            return cls.from_tuples(MISTRAL_KNOWN_FOR, model)
        elif name == "OfCourse" and is_pythia:
            return cls.from_tuples(PYTHIA_OF_COURSE, model)
        elif name == "OfCourse" and is_mistral:
            return cls.from_tuples(PYTHIA_OF_COURSE, model)
        elif name == "Code" and is_santacoder:
            return cls.from_tuples(SANTACODER_CODE, model)
        elif name == "BooleanNegator":
            return BooleanNegatorDataset(model, **kwargs).to_counterfactual()
        elif name == "BooleanOperator":
            return BooleanOperatorDataset(model, **kwargs).to_counterfactual()
        elif name == "ToyBinding":
            return ToyBindingTemplate(model, **kwargs).to_counterfactual()
        elif name == "ToyDeduction":
            return ToyDeductionTemplate(model, **kwargs).to_counterfactual()
        elif name == "ToyProfiles":
            return ToyProfilesTemplate(model, **kwargs).to_counterfactual()
        else:
            raise ValueError(f"Unknown dataset name {name}")

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
