import torch

# transformers comes from huggingface-hub install.
from transformers import AutoModelForCausalLM, AutoTokenizer


class Algorithm:
    def __init__(self, logger=None, model_names=None):
        self.logger = logger
        if model_names is None:
            model_names = ["EleutherAI/pythia-410m", "EleutherAI/pythia-410m"]
        self.models = [self.get_model(model_name)
                       for model_name in model_names]

    @staticmethod
    def get_model(model_name):
        # Load the model and tokenizer.  This will be memoized.
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     trust_remote_code=False,
                                                     low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return model, tokenizer

    def one(self, model, tokenizer, input_text="The quick brown fox"):
        # Run a model one time

        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        # Generate -- or predict the next token in the sequence
        output = model(input_ids)

        # logits are probabilitys of each token in the vocabulary
        # at each location in the sequence given all the tokens before it.
        logits = output["logits"]

        # Most likely next token
        next_token_id = logits.argmax(-1)[0, -1]

        # Decode the token to a string
        next_token_string = tokenizer.decode(next_token_id)

        logger = self.logger
        if logger:
            logger.info(f"input_string: {input_text}")
            logger.info(f"input_ids (tokenized input): {input_ids}")
            logger.info(f"output keys: {output.keys()}")
            logger.info(f"output logits shape: {logits.shape}")
            logger.info(f"next_token_id: {next_token_id}")
            logger.info(f"next_token_string: {next_token_string}")

        return input_ids, output, logits, next_token_id, next_token_string

    def run(self, input_text="The quick brown fox"):
        result = []
        for model, tokenizer in self.models:
            input_ids, output, logits, next_token_id, next_token_string = (
                self.one(model, tokenizer, input_text))
            result.append((input_ids, output, logits,
                           next_token_id, next_token_string))
            input_text += next_token_string

        status = next_token_string == " over"
        return {'status': status, 'output': input_text}
