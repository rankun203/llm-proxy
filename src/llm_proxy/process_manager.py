"""
Process management for vLLM server with SLURM support.
"""

import asyncio
import logging
import shlex
import signal
import subprocess
from datetime import datetime
from typing import List, Optional
from .utils import parse_vllm_command

logger = logging.getLogger(__name__)


class ProcessManager:
    def __init__(
        self,
        target_port: int,
        use_slurm: bool = False,
        srun_cmd: str = "srun --mem=30G --gres=gpu:1",
        loopback_user: Optional[str] = None,
        loopback_host: Optional[str] = None,
    ):
        self.target_port = target_port
        self.use_slurm = use_slurm
        self.srun_cmd = srun_cmd
        self.loopback_user = loopback_user
        self.loopback_host = loopback_host
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.stdout_file = None
        self.stderr_file = None

    async def start_vllm_server(self, vllm_command: List[str]) -> bool:
        """
        Start the vLLM server process.

        Args:
            vllm_command: The vLLM command arguments

        Returns:
            True if started successfully, False otherwise
        """
        if self.is_running:
            logger.warning("vLLM server is already running")
            return True

        try:
            # Parse and modify the vLLM command
            modified_command = parse_vllm_command(
                vllm_command, self.target_port)

            if self.use_slurm:
                final_command = self._build_slurm_command(modified_command)
            else:
                final_command = modified_command

            logger.info(
                f"Starting vLLM server with command: {' '.join(final_command)}")

            # Create log files with timestamp in /tmp directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stdout_log_path = f"/tmp/llm_proxy_stdout_{timestamp}.log"
            stderr_log_path = f"/tmp/llm_proxy_stderr_{timestamp}.log"

            # Open log files
            self.stdout_file = open(stdout_log_path, 'w')
            self.stderr_file = open(stderr_log_path, 'w')

            logger.info(f"Logging stdout to: {stdout_log_path}")
            logger.info(f"Logging stderr to: {stderr_log_path}")

            # Start the process
            self.process = subprocess.Popen(
                final_command,
                stdout=self.stdout_file,
                stderr=self.stderr_file,
                preexec_fn=None if self.use_slurm else lambda: signal.signal(
                    signal.SIGINT, signal.SIG_IGN)
            )

            self.is_running = True
            logger.info(f"vLLM server started with PID: {self.process.pid}")

            # Wait a bit to check if the process started successfully
            await asyncio.sleep(2)

            if self.process.poll() is not None:
                # Process has already terminated
                logger.error(
                    "vLLM server failed to start. Check log files for details.")
                self.is_running = False
                self._close_log_files()
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to start vLLM server: {e}")
            self.is_running = False
            self._close_log_files()
            return False

    def _build_slurm_command(self, vllm_command: List[str]) -> List[str]:
        """
        Build the SLURM command with SSH reverse tunneling.

        Args:
            vllm_command: Modified vLLM command

        Returns:
            Complete SLURM command
        """
        if not self.loopback_user or not self.loopback_host:
            raise ValueError(
                "loopback_user and loopback_host are required when using SLURM")

        # Build the bash script content
        bash_script = f"""
# Start reverse SSH tunnel in background
ssh -vv -N -f -R {self.target_port}:localhost:{self.target_port} {self.loopback_user}@{self.loopback_host}

# Start vLLM server
{' '.join(shlex.quote(arg) for arg in vllm_command)}
"""

        # Build the complete SLURM command
        srun_parts = shlex.split(self.srun_cmd)
        final_command = srun_parts + ["bash", "-c", bash_script.strip()]

        return final_command

    async def stop_vllm_server(self) -> bool:
        """
        Stop the vLLM server process.

        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.is_running or not self.process:
            logger.info("vLLM server is not running")
            return True

        try:
            logger.info("Stopping vLLM server...")

            # Send SIGTERM first for graceful shutdown
            self.process.terminate()

            # Wait for graceful shutdown
            try:
                await asyncio.wait_for(
                    asyncio.create_task(self._wait_for_process()),
                    timeout=10.0
                )
                logger.info("vLLM server stopped gracefully")
            except asyncio.TimeoutError:
                # Force kill if graceful shutdown failed
                logger.warning("Graceful shutdown timed out, force killing...")
                self.process.kill()
                await asyncio.create_task(self._wait_for_process())
                logger.info("vLLM server force killed")

            self.is_running = False
            self.process = None
            self._close_log_files()
            return True

        except Exception as e:
            logger.error(f"Failed to stop vLLM server: {e}")
            return False

    async def _wait_for_process(self):
        """Wait for the process to terminate."""
        while self.process and self.process.poll() is None:
            await asyncio.sleep(0.1)

    def _close_log_files(self):
        """Close log files if they are open."""
        if self.stdout_file:
            try:
                self.stdout_file.close()
            except Exception as e:
                logger.warning(f"Error closing stdout log file: {e}")
            finally:
                self.stdout_file = None

        if self.stderr_file:
            try:
                self.stderr_file.close()
            except Exception as e:
                logger.warning(f"Error closing stderr log file: {e}")
            finally:
                self.stderr_file = None

    def is_server_running(self) -> bool:
        """
        Check if the vLLM server is running.

        Returns:
            True if running, False otherwise
        """
        if not self.is_running or not self.process:
            return False

        # Check if process is still alive
        if self.process.poll() is not None:
            self.is_running = False
            return False

        return True

    async def cleanup(self):
        """Clean up resources."""
        if self.is_running:
            await self.stop_vllm_server()
        self._close_log_files()
