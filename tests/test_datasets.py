import torch
import unittest
from unittest.mock import MagicMock, patch
from utils.datasets import ExperimentData, OWTData


class TestOWTData(unittest.TestCase):
    def setUp(self):
        # Mocks
        self.mocked_dataset = {
            "train": MagicMock(),
            "validation": MagicMock(),
            "test": MagicMock(),
        }
        self.mocked_model = MagicMock()
        self.mocked_model.tokenizer = MagicMock()
        self.text_column = "text"

    @patch("utils.datasets.tokenize_and_concatenate")
    def test_tokenize(self, mocked_tokenize_and_concatenate):
        outputs = {
            "train": MagicMock(),
            "validation": MagicMock(),
            "test": MagicMock(),
        }
        mocked_tokenize_and_concatenate.side_effect = outputs.values()
        owt_data = OWTData(self.mocked_dataset, self.mocked_model)

        owt_data._tokenize()
        for split in self.mocked_dataset.keys():
            assert self.mocked_dataset[split] == outputs[split]

    def test_create_attention_mask(self):
        example = {"tokens": torch.tensor([1, 2, 3, 4], dtype=torch.long)}
        owt_data = OWTData(self.mocked_dataset, self.mocked_model)
        result = owt_data._create_attention_mask(example)
        self.assertEqual(result["attention_mask"].tolist(), [1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
