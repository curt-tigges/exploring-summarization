import torch
from summarization_utils.path_patching import get_batch_and_seq_pos_indices
import unittest


class TestBatchPosIndices(unittest.TestCase):
    def testIntPos(self):
        seq_pos = 3
        batch_size = 2
        seq_len = 5
        batch_indices, seq_pos_indices = get_batch_and_seq_pos_indices(
            seq_pos, batch_size, seq_len
        )
        self.assertTrue((torch.tensor([[0], [1]]) == batch_indices).all())
        self.assertTrue((torch.tensor([[3], [3]]) == seq_pos_indices).all())

    def testNonePos(self):
        seq_pos = None
        batch_size = 2
        seq_len = 5
        batch_indices, seq_pos_indices = get_batch_and_seq_pos_indices(
            seq_pos, batch_size, seq_len
        )
        self.assertEqual(batch_indices, slice(None))
        self.assertEqual(seq_pos_indices, slice(None))

    def testBatchList(self):
        seq_pos = [3, 2]
        batch_size = 2
        seq_len = 5
        batch_indices, seq_pos_indices = get_batch_and_seq_pos_indices(
            seq_pos, batch_size, seq_len
        )
        self.assertTrue((torch.tensor([[0, 0], [1, 1]]) == batch_indices).all())
        self.assertTrue((torch.tensor([[3, 2], [3, 2]]) == seq_pos_indices).all())

    def testNestedList(self):
        seq_pos = [[3, 2], [1, 0]]
        batch_size = 2
        seq_len = 5
        batch_indices, seq_pos_indices = get_batch_and_seq_pos_indices(
            seq_pos, batch_size, seq_len
        )
        self.assertTrue((torch.tensor([[0, 0], [1, 1]]) == batch_indices).all())
        self.assertTrue((torch.tensor([[3, 2], [1, 0]]) == seq_pos_indices).all())

    def testNonHomogeneousList(self):
        seq_pos = [[3, 2], [1]]
        batch_size = 2
        seq_len = 5
        batch_indices, seq_pos_indices = get_batch_and_seq_pos_indices(
            seq_pos, batch_size, seq_len
        )
        self.assertTrue((torch.tensor([0, 0, 1]) == batch_indices).all())
        self.assertTrue((torch.tensor([3, 2, 1]) == seq_pos_indices).all())

    def testTensor(self):
        seq_pos = torch.tensor([[3, 2], [1, 0]])
        batch_size = 2
        seq_len = 5
        batch_indices, seq_pos_indices = get_batch_and_seq_pos_indices(
            seq_pos, batch_size, seq_len
        )
        self.assertTrue((torch.tensor([[0, 0], [1, 1]]) == batch_indices).all())
        self.assertTrue((torch.tensor([[3, 2], [1, 0]]) == seq_pos_indices).all())
