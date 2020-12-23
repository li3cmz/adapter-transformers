import os
import copy
import unittest

import torch

from transformers import (
    OpenAIGPTLMHeadModel,
    BertTokenizer,
    AdamW
)
from transformers.testing_utils import require_torch


def filter_parameters(model, filter_string):
    return {k: v for (k, v) in model.named_parameters() if filter_string in k}

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        
    def train(self, input_ids, token_type_ids, lm_labels):
        self.model.train()

        (lm_loss), *_ = self.model(input_ids, labels=lm_labels, token_type_ids=token_type_ids)
        loss = lm_loss / 64
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


@require_torch
class AdapterTrainingTest(unittest.TestCase):

    model_names = ['../tmp/CDial-GPT/CDial-GPT_LCCC-large/']

    def test_train_single_adapter(self):
        for model_name in self.model_names:
            with self.subTest(model_name=model_name):
                model_class = OpenAIGPTLMHeadModel
                tokenizer_class = BertTokenizer
                tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=True)
                model = model_class.from_pretrained(model_name)

                # add two adapters: one will be trained and the other should be frozen
                model.add_adapter(adapter_name="test1", adapter_type="text_task")
                model.add_adapter(adapter_name="test2", adapter_type="text_task")

                self.assertIn("test1", model.config.adapters.adapters)
                self.assertIn("test2", model.config.adapters.adapters)

                # train the mrpc adapter -> should be activated & unfreezed
                model.train_adapter("test1")
                self.assertEqual([["test1"]], model.active_adapters)

                # all weights of the adapter should be activated
                for k, v in filter_parameters(model, "text_task_adapters.test1").items():
                    self.assertTrue(v.requires_grad, k)
                # all weights of the adapter not used for training should be freezed
                for k, v in filter_parameters(model, "text_task_adapters.test2").items():
                    self.assertFalse(v.requires_grad, k)
                # weights of the model should be freezed (check on some examples)
                for k, v in filter_parameters(model, "layer.0.attn").items():
                    self.assertFalse(v.requires_grad, k)

                state_dict_pre = copy.deepcopy(model.state_dict())

                input_ids = torch.ones(2, 128).long()
                token_type_ids = input_ids
                lm_labels = torch.zeros(2, 128).long()

                optimizer = AdamW([{'params': model.parameters(), 'initial_lr': 5e-4}], lr=5e-4, correct_bias=True)
                trainer = Trainer(
                    model=model,
                    optimizer=optimizer
                )
                trainer.train(input_ids, token_type_ids, lm_labels)

                for ((k1, v1), (k2, v2)) in zip(state_dict_pre.items(), model.state_dict().items()):
                    if "test1" in k1:
                        self.assertFalse(torch.equal(v1, v2))
                    else:
                        self.assertTrue(torch.equal(v1, v2))
                
                save_path="adapters/text-task/test1/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                model.save_adapter(save_path, "test1")

if __name__ == "__main__":
    unittest.main()