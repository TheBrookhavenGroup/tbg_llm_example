import torch

# transformers comes from huggingface-hub install.
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_one_models(model_name):

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 trust_remote_code=False,
                                                 low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def get_models(model_names=["EleutherAI/pythia-410m"]):
    # "EleutherAI/pythia-410m" is small enough to run on a CPU.
    return [get_one_models(model_name) for model_name in model_names]


def get_model_output(model, tokenizer, input_string):
    # A model is a function that takes a tensor of token indices and returns a
    # dictionary of model outputs

    # Tokenize -- or convert string to a list of indices from the vocabulary
    input_ids = tokenizer.encode(input_string, return_tensors="pt")

    # Generate -- or predict the next token in the sequence
    output = model(input_ids)

    # logits are probabilitys of each token in the vocabulary at each location
    # in the sequence given
    # all the tokens before it
    logits = output["logits"]

    # Most likely next token
    next_token_id = logits.argmax(-1)[0, -1]

    # Decode the token to a string
    next_token_string = tokenizer.decode(next_token_id)

    return input_ids, output, logits, next_token_id, next_token_string
