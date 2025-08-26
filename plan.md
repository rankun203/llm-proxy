implement a command line tool that has a python program that imports vllm and fastapi server

and it implements all APIs for vllm openai compatible server with fastapi paths and proxy to vllm

every time the fastapi receives a inference request on actual inference (chat/completion), it will update "last_access_time" to current time. Then there is a separate process that checks once certain amount of time has passed, it will turn off the vllm server.

the server runs on a serve_config, which defines how long to wait, the total size of models to download (vllm automatically downloads).

Since our server uses slurm, so we need to use slurm to launch the vllm service and proxy requests to it. However our main program will not run in slurm.

We run program in slurm using something like `srun --mem=250G --gres=gpu:1 uv run python -m pyserini.index.merge_faiss_indexes       --prefix ~/downloads/faiss-flat.msmarco-v2.1-doc-segmented-shard       --shard-num 1       --dimension 1024`

In case there is a slurm python library that we can use to launch the vllm, use it.

for adding packages use uv add

srun --mem=30G --gres=gpu:1 uv run --with vllm vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8040 \
    --model tiiuae/falcon3-10b-instruct \
    --api-key apikey \
    --tensor-parallel-size 0 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048 \
    --distributed-executor-backend mp \
    --gpu-memory-utilization 0.95

Recall that in slurm, it has a srun command that will launch our program and assign memory, gpu to it

---

Create a python package with a command llm-proxy, when launches, it starts a FastAPI server right away and have a /health api, as well as all APIs for vllm openai compatible server with fastapi paths and proxy to vllm, like /v1/chat/completions and /v1/models (and if there are others).

llm-proxy accepts parameters like --port xxxx and --target-port. port is what current FastAPI will listen on, --target-port is what vllm will listen on, if user provided vllm command specifies --port, this --target-port should override that, and anything that's that's not its own parameters as a command, usually the full vllm command. It doesn't run the vllm command directly, but only when received a request on chat or models api calls.

llm-proxy has another parameter --use-slurm, which when set, it will require more parameters --loopback-user and --loopback-host, which the slurm command will later use to establish a reverse port forwarding back to this server, (let's say vllm listens on 8040 on slurm execution server), llm-proxy will run another ssh -R command on execution server to forward current server port 8040 to execution server 8040. So the execution server will need to know current server host name to connect to. Also remind user to use ssh current-host and/or manually fixing known hosts or auth keys issues. If --use-slurm is not set, it will not use srun to run vllm, but instead run the vllm command directly and forward requests to that port.

if using slurm, run the final command this way:

```bash
srun --mem=30G --gres=gpu:1 bash -c "
# Start reverse SSH tunnel in background
ssh -v -N -f -R [target-port]:localhost:[target-port] [loopback-user]@[loopback-host]

# Start vLLM server
execute the full vllm command (but replaced --host with 0.0.0.0 and --port with [target-port], to properly connect network)

# the command could look like this:
# uv run --with vllm --with numpy==1.26.4 --with flashinfer-python==0.2.2 python -m vllm.entrypoints.openai.api_server \
#     --host 0.0.0.0 \
#     --port [target-port] \
#     --model tiiuae/falcon3-10b-instruct \
#     --api-key apikey \
#     --tensor-parallel-size 1 \
#     --enable-chunked-prefill \
#     --max-num-batched-tokens 2048 \
#     --distributed-executor-backend mp \
#     --gpu-memory-utilization 0.95
"
```

if not using slurm, of course just run the vllm command directly (but also replace --host and --port with 0.0.0.0 and [target-port] for proper network connectivity)

This package will be published as a pypi python package, when I use it, I do:

llm-proxy --port 8085 --target-port 8084 --use-slurm --loopback-user e123456 --loopback-host 10.10.10.5 uv run --with vllm --with numpy==1.26.4 --with flashinfer-python==0.2.2 python -m vllm.entrypoints.openai.api_server \
     --host 0.0.0.0 \
     --port 8084 \
     --model tiiuae/falcon3-10b-instruct \
     --api-key apikey \
     --tensor-parallel-size 1 \
     --enable-chunked-prefill \
     --max-num-batched-tokens 2048 \
     --distributed-executor-backend mp \
     --gpu-memory-utilization 0.95

once this launches, my local FastAPI server will be running and proxying requests to the vLLM server.

Notice llm-proxy doesn't do much changes to the vllm running command, that means, if it's launching another server like SGLang, it will still launch successfully and serving requests.

Once the FastAPI server haven't received a new request for 30 minutes (configurable with --idle-timeout), it will shutdown the vllm server (if running with slurm, it will gracefully shutdown the srun command).
