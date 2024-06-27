import torch
from llmmodels import get_models, get_model_output


def run(input_string="The quick brown fox"):
    for model, tokenizer in get_models():
        input_ids, output, logits, next_token_id, next_token_string = (
            get_model_output(model, tokenizer, input_string))

        print(f"input_string: {input_string}")
        print(f"input_ids (tokenized input): {input_ids}")
        print(f"output keys: {output.keys()}")
        print(f"output logits shape: {logits.shape}")
        print(f"next_token_id: {next_token_id}")
        print(f"next_token_string: {next_token_string}")
        result = tokenizer.decode(torch.cat((input_ids[0],
                                             next_token_id.unsqueeze(0))))
        print(f"The entire output string: {result}")

        # We can also generate lots of output by predicting the next token in a loop,
        # but this is wrapped for us in a method called generate()

        # Generate -- or predict the next token in the sequence
        output = model.generate(input_ids, max_length=20)

        # Decode the output to a string
        output_string = tokenizer.decode(output[0])

        print(f"output_string: {output_string}")

        print("\n\ndone.\n")
