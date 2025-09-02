"""
Example usage:
1. Install vLLM: pip install vllm
2. Run the script:
   - Default: python infer_w_conf.py
   - Custom message: python infer_w_conf.py "Hello, I am "
   - Other examples: python infer_w_conf.py "The countries using Mandarin as the official language include "
"""

from vllm import LLM, SamplingParams
import math
import sys

# Load the model
model_name = "Qwen/Qwen3-8B"
llm = LLM(model=model_name)

# Get input message from command line argument, default to "My name is "
if len(sys.argv) > 1:
    input_message = sys.argv[1]
else:
    input_message = "My name is "

# Create prompt using string template (based on Qwen3 template)
# This follows the pattern: <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
prompt = f"""<|im_start|>user
Hello, what is your name?<|im_end|>
<|im_start|>assistant
<think>

</think>

{input_message}"""

print(f"Starting generation from: '{input_message}'")

# Set sampling parameters (recommended for non-thinking mode)
params = SamplingParams(
    logprobs=5,  # Get top 5 logprobs per token
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
    max_tokens=5  # Only generate 5 tokens to see the name prediction
)

outputs = llm.generate([prompt], params)

# Access probabilities for each token
for output in outputs:
    logprobs = output.outputs[0].logprobs
    for i, token_logprobs in enumerate(logprobs):
        # print(f"\nStep {i} - Alternative tokens and probabilities:")
        # Sort by rank (or by probability descending)
        sorted_tokens = sorted(token_logprobs.items(), key=lambda x: x[1].rank)
        
        # Calculate probabilities - vLLM returns log(P) where P is already softmax over full vocab
        total_prob_shown = 0
        for token_id, logprob_obj in sorted_tokens:
            # exp(log(P)) = P, where P is the true softmax probability over all vocabulary
            probability = math.exp(logprob_obj.logprob)
            decoded_token = logprob_obj.decoded_token
            rank = logprob_obj.rank
            total_prob_shown += probability
            
            if rank == 1:
                print(f"{decoded_token}| {probability:.4f}")
            else:
                print(f"\t{decoded_token}| {probability:.4f}")

        print(f"  Top 5 probability\t: {total_prob_shown:.6f}")
        print(f"  Remaining probability\t: {1.0 - total_prob_shown:.6f}")
