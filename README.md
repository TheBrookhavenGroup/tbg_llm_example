# tbg_llm_example
Learning how to use LLMs

## Try it

```shell
$ pip install git+https://github.com/TheBrookhavenGroup/tbg_llm_example.git
$ python src/tbg_llm_example/example.py
```

## A note about how PyTorch manages hardware

PyTorch will use GPUs if available or fallback to using the CPU.  If we
intantiate more than one model at a time it will allocate the first one
to the first gpu and the second to the second gpu and so on.

There are option in PyTorch like `device_map` to control this manually.

