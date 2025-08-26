"""
Utility functions for llm-proxy.
"""

import socket
import logging
from typing import List

logger = logging.getLogger(__name__)


def find_available_port(start_port: int = 8101, max_attempts: int = 100) -> int:
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                logger.debug(f"Found available port: {port}")
                return port
        except OSError:
            continue

    raise RuntimeError(
        f"No available ports found in range {start_port}-{start_port + max_attempts}")


def is_port_available(port: int, host: str = 'localhost') -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except OSError:
        return False


def parse_vllm_command(command_args: List[str], target_port: int) -> List[str]:
    """
    Parse and modify vLLM command arguments to ensure proper host and port settings.

    Args:
        command_args: Original command arguments
        target_port: Target port to use

    Returns:
        Modified command arguments
    """
    modified_args = []
    i = 0
    host_set = False
    port_set = False

    while i < len(command_args):
        arg = command_args[i]

        if arg == '--host':
            # Replace host with 0.0.0.0
            modified_args.extend(['--host', '0.0.0.0'])
            host_set = True
            i += 2  # Skip the next argument (the host value)
        elif arg == '--port':
            # Replace port with target_port
            modified_args.extend(['--port', str(target_port)])
            port_set = True
            i += 2  # Skip the next argument (the port value)
        elif arg.startswith('--host='):
            # Handle --host=value format
            modified_args.append('--host=0.0.0.0')
            host_set = True
            i += 1
        elif arg.startswith('--port='):
            # Handle --port=value format
            modified_args.append(f'--port={target_port}')
            port_set = True
            i += 1
        else:
            modified_args.append(arg)
            i += 1

    # Add host and port if not already present
    if not host_set:
        modified_args.extend(['--host', '0.0.0.0'])
    if not port_set:
        modified_args.extend(['--port', str(target_port)])

    logger.debug(f"Modified vLLM command: {' '.join(modified_args)}")
    return modified_args


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
