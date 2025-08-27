# llm-proxy-ondemand

A FastAPI proxy server for vLLM with SLURM support and automatic process management.

## Features

- **On-demand vLLM server startup**: The vLLM server is only started when the first request is received
- **SLURM integration**: Run vLLM on SLURM clusters with SSH reverse tunneling
- **Automatic port management**: Auto-select available ports if not specified
- **Idle timeout**: Automatically shutdown vLLM server after a configurable period of inactivity
- **OpenAI-compatible API**: Proxy all vLLM OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/models`, etc.)
- **Health monitoring**: Built-in health endpoint for monitoring
- **Flexible command handling**: Works with any vLLM-compatible server command (as long as target run command has --host and --port)

## Installation

```bash
pip install llm-proxy-ondemand
```

Or install from source:

```bash
git clone <repository-url>
cd llm-proxy-ondemand
pip install -e .
```

## Usage

### Basic Usage

Start a proxy server that will launch vLLM on-demand:

```bash
llm-proxy-ondemand -- uv run --with vllm python -m vllm.entrypoints.openai.api_server --model tiiuae/falcon3-10b-instruct
```

**Note:** Use `--` to separate llm-proxy-ondemand options from the vLLM command.

This will:

- Start FastAPI server on port 8100 (default)
- Auto-select an available port for vLLM (starting from 8101)
- Launch vLLM when the first request is received
- Shutdown vLLM after 30 minutes of inactivity

### Custom Ports

```bash
llm-proxy-ondemand --port 8085 --target-port 8084 -- uv run --with vllm python -m vllm.entrypoints.openai.api_server --model some-model
```

### SLURM Integration

For running on SLURM clusters with SSH reverse tunneling:

```bash
llm-proxy-ondemand --use-slurm --loopback-user e123456 --loopback-host 10.10.10.5 -- \
  uv run --with vllm python -m vllm.entrypoints.openai.api_server \
  --model tiiuae/falcon3-10b-instruct \
  --api-key apikey \
  --tensor-parallel-size 1
```

### Custom SLURM Resources

```bash
llm-proxy-ondemand --use-slurm --srun-cmd "srun --mem=60G --gres=gpu:2 --time=4:00:00" \
  --loopback-user e123456 --loopback-host 10.10.10.5 -- \
  uv run --with vllm python -m vllm.entrypoints.openai.api_server --model some-model

# A working example
uv run --with vllm --with numpy==1.26.4 --with flashinfer-python==0.2.2 llm-proxy-ondemand --use-slurm --loopback-user e128356 --loopback-host 10.205.51.153 --api-key password -- python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-0.6B
```

Issues that are hard to debug:

- Wrong ssh username or password, check the logging output files.

### Custom Idle Timeout

```bash
llm-proxy-ondemand --idle-timeout 3600 -- uv run --with vllm python -m vllm.entrypoints.openai.api_server --model some-model
```

## Command Line Options

- `--port`: Port for the FastAPI proxy server (default: 8100)
- `--target-port`: Port for the vLLM server (default: auto-select from 8101+)
- `--use-slurm`: Use SLURM to run the vLLM server
- `--srun-cmd`: SLURM srun command (default: "srun --mem=30G --gres=gpu:1")
- `--loopback-user`: SSH user for reverse tunneling (required with --use-slurm)
- `--loopback-host`: SSH host for reverse tunneling (required with --use-slurm)
- `--idle-timeout`: Idle timeout in seconds before shutting down vLLM server (default: 1800)
- `--ping-path`: Path to ping the target server to check if it's ready (default: /ping)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR, default: INFO)
- `--api-key`: API key for Bearer token authentication (optional)

## API Endpoints

### Health Check

```bash
curl http://localhost:8100/health
```

Response:

```json
{
  "status": "healthy",
  "llm_running": false,
  "target_port": 8101,
  "last_request": 1640995200.0
}
```

### OpenAI-Compatible Endpoints

All vLLM OpenAI-compatible endpoints are proxied:

- `POST /v1/chat/completions` - Chat completions
- `GET /v1/models` - List available models
- And any other `/v1/*` endpoints supported by vLLM

Example:

```bash
curl -X POST http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tiiuae/falcon3-10b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## How It Works

1. **Startup**: llm-proxy-ondemand starts a FastAPI server immediately
2. **First Request**: When a request is made to `/v1/*`, the vLLM server is started
3. **Proxying**: All subsequent requests are forwarded to the vLLM server
4. **Idle Monitoring**: After the configured idle timeout, the vLLM server is shutdown
5. **Auto-restart**: The vLLM server will be restarted on the next request

## SLURM Integration Details

When using `--use-slurm`, llm-proxy-ondemand:

1. Constructs a SLURM job using the provided `--srun-cmd`
2. Sets up SSH reverse tunneling from the compute node back to the proxy server
3. Runs the vLLM command on the compute node
4. Forwards requests through the SSH tunnel

The generated SLURM command looks like:

```bash
srun --mem=30G --gres=gpu:1 bash -c "
# Start reverse SSH tunnel in background
ssh -v -N -f -R 8101:localhost:8101 user@host

# Start vLLM server
uv run --with vllm python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8101 --model some-model
"
```

## Requirements

For SLURM usage:

- SSH access from compute node to current node
- Proper SSH key setup for password-less authentication

## Development

```bash
git clone <repository-url>
cd llm-proxy-ondemand
pip install -e .
```

## License

MIT License
