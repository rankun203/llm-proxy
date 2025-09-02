from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")
params = SamplingParams(logprobs=5)  # Get top 5 logprobs per token
outputs = llm.generate(["You can call me "], params)

# Access probabilities for each token
for output in outputs:
    logprobs = output.outputs[0].logprobs
    for i, token_logprobs in enumerate(logprobs):
        print(f"Step {i}: {token_logprobs}")  # Dict of token_id -> logprob
