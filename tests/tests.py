import unittest
from src.tbg_llm_example.llmmodels import Algorithm


class LLMTests(unittest.TestCase):
    def setUp(self):
        self.algorithm = Algorithm()

    def test_fox(self):
        input_string = "The quick brown fox"
        result = self.algorithm.run(input_string)
        self.assertTrue(result['status'])
        self.assertEqual('over', result['output'].split()[-1])
