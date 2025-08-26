"""
Command-line interface for llm-proxy.
"""

import asyncio
import logging
import signal
import sys
import uuid
from typing import List, Optional
import click
import uvicorn
from .utils import find_available_port, setup_logging
from .process_manager import ProcessManager
from .server import ProxyServer

logger = logging.getLogger(__name__)


class CLIContext:
    def __init__(self):
        self.proxy_server: Optional[ProxyServer] = None
        self.shutdown_event = asyncio.Event()


cli_context = CLIContext()


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    cli_context.shutdown_event.set()


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option(
    "--port",
    default=8100,
    help="Port for the FastAPI proxy server (default: 8100)"
)
@click.option(
    "--target-port",
    default=None,
    type=int,
    help="Port for the vLLM server (default: auto-select from 8101+)"
)
@click.option(
    "--use-slurm",
    is_flag=True,
    help="Use SLURM to run the vLLM server"
)
@click.option(
    "--srun-cmd",
    default="srun --mem=30G --gres=gpu:1",
    help="SLURM srun command (default: 'srun --mem=30G --gres=gpu:1')"
)
@click.option(
    "--loopback-user",
    help="SSH user for reverse tunneling (required with --use-slurm)"
)
@click.option(
    "--loopback-host",
    help="SSH host for reverse tunneling (required with --use-slurm)"
)
@click.option(
    "--idle-timeout",
    default=1800,
    help="Idle timeout in seconds before shutting down vLLM server (default: 1800)"
)
@click.option(
    "--ping-path",
    default="/ping",
    help="Path to ping the target server to check if it's ready (default: /ping)"
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging level (default: INFO)"
)
@click.option(
    "--api-key",
    help="API key for Bearer token authentication (optional)"
)
@click.pass_context
def main(
    ctx: click.Context,
    port: int,
    target_port: Optional[int],
    use_slurm: bool,
    srun_cmd: str,
    loopback_user: Optional[str],
    loopback_host: Optional[str],
    idle_timeout: int,
    ping_path: str,
    log_level: str,
    api_key: Optional[str],
):
    """
    llm-proxy: A FastAPI proxy server for vLLM with SLURM support.

    Start a FastAPI server that proxies requests to a vLLM server.
    The vLLM server is started on-demand when the first request is received.

    Use -- to separate llm-proxy options from the vLLM command.

    Examples:

    \b
    # Basic usage
    llm-proxy -- uv run --with vllm python -m vllm.entrypoints.openai.api_server --model some-model

    \b
    # With SLURM
    llm-proxy --use-slurm --loopback-user user --loopback-host host -- uv run --with vllm python -m vllm.entrypoints.openai.api_server --model some-model

    \b
    # Custom ports and timeout
    llm-proxy --port 8085 --target-port 8084 --idle-timeout 3600 -- uv run --with vllm python -m vllm.entrypoints.openai.api_server --model some-model
    """
    # Set up logging
    setup_logging(log_level)

    # Get vLLM command from extra arguments
    vllm_command = ctx.args
    logger.debug(f"vLLM command: {vllm_command}")

    # Validate arguments
    if use_slurm and (not loopback_user or not loopback_host):
        click.echo(
            "Error: --loopback-user and --loopback-host are required when using --use-slurm", err=True)
        sys.exit(1)

    if not vllm_command:
        click.echo("Error: vLLM command is required", err=True)
        sys.exit(1)

    # Auto-select target port if not provided
    if target_port is None:
        try:
            target_port = find_available_port()
            logger.info(f"Auto-selected target port: {target_port}")
        except RuntimeError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

    # Convert vllm_command to list
    vllm_command_list = list(vllm_command)

    # Generate unique short ID for this instance
    instance_id = str(uuid.uuid4())[:8]

    logger.info(
        f"Starting llm-proxy on port {port}, target port {target_port}")
    logger.info(f"Instance ID: {instance_id}")
    logger.info(f"vLLM command: {' '.join(vllm_command_list)}")

    # Run the async main function
    asyncio.run(async_main(
        port=port,
        target_port=target_port,
        use_slurm=use_slurm,
        srun_cmd=srun_cmd,
        loopback_user=loopback_user,
        loopback_host=loopback_host,
        idle_timeout=idle_timeout,
        ping_path=ping_path,
        vllm_command=vllm_command_list,
        instance_id=instance_id,
        api_key=api_key
    ))


async def async_main(
    port: int,
    target_port: int,
    use_slurm: bool,
    srun_cmd: str,
    loopback_user: Optional[str],
    loopback_host: Optional[str],
    idle_timeout: int,
    ping_path: str,
    vllm_command: List[str],
    instance_id: str,
    api_key: Optional[str]
):
    """Async main function."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create process manager
    process_manager = ProcessManager(
        target_port=target_port,
        use_slurm=use_slurm,
        srun_cmd=srun_cmd,
        loopback_user=loopback_user,
        loopback_host=loopback_host,
        instance_id=instance_id
    )

    # Create proxy server
    proxy_server = ProxyServer(
        port=port,
        target_port=target_port,
        process_manager=process_manager,
        api_key=api_key,
        idle_timeout=idle_timeout,
        ping_path=ping_path
    )

    # Set the vLLM command
    proxy_server.set_vllm_command(vllm_command)

    # Store in global context for signal handler
    cli_context.proxy_server = proxy_server

    # Configure uvicorn
    config = uvicorn.Config(
        app=proxy_server.app,
        host="0.0.0.0",
        port=port,
        log_level=logging.getLogger().level,
        access_log=False
    )

    server = uvicorn.Server(config)

    try:
        logger.info(f"FastAPI server starting on http://0.0.0.0:{port}")
        logger.info("Health endpoint available at /health")
        logger.info("vLLM proxy endpoints available at /v1/*")

        # Start server in a task
        server_task = asyncio.create_task(server.serve())

        # Wait for shutdown signal
        await cli_context.shutdown_event.wait()

        logger.info("Shutting down server...")
        server.should_exit = True

        # Wait for server to shutdown
        await server_task

    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        # Clean up
        logger.info("Cleaning up...")
        await proxy_server.cleanup()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
