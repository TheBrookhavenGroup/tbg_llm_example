import unittest
from src.tbg_llm_example.llmmodels import get_models, get_model_output


class LLMTests(unittest.TestCase):
    def setUp(self):
        model_name = "EleutherAI/pythia-410m"
        self.models = get_models(model_names=[model_name, model_name])

    def test_fox(self):
        input_string = "The quick brown fox"
        for model, tokenizer in self.models:
            input_ids, output, logits, next_token_id, next_token_string = (
                get_model_output(model, tokenizer, input_string))
            self.assertEqual(" jumps", next_token_string)
