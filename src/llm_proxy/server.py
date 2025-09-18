"""
FastAPI server with health endpoint and vLLM proxy functionality.
"""

import asyncio
import json
import logging
import time
from typing import Optional
from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from .process_manager import ProcessManager

logger = logging.getLogger(__name__)


class ProxyServer:
    def __init__(
        self,
        port: int,
        target_port: int,
        process_manager: ProcessManager,
        api_key: Optional[str] = None,
        idle_timeout: int = 1800,  # 30 minutes in seconds
        ping_path: str = "/ping",
    ):
        self.port = port
        self.target_port = target_port
        self.process_manager = process_manager
        self.api_key = api_key
        self.idle_timeout = idle_timeout
        self.ping_path = ping_path
        self.last_request_time = time.time()
        self.vllm_command: Optional[list] = None
        self.app = FastAPI(title="llm-proxy", version="0.1.0")
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 minute timeout

        self._setup_routes()
        self._start_idle_monitor()

    def verified_api_token(self, authorization: Optional[str] = Header(None)):
        """Verify API token if API key is configured."""
        if not self.api_key:
            return True

        if not authorization:
            raise HTTPException(
                status_code=401, detail="Authorization header required")

        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401, detail="Invalid authorization format")

        token = authorization[7:]  # Remove "Bearer " prefix
        if token != self.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

        return True

    def _setup_routes(self):
        """Set up FastAPI routes."""

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            idle_time = time.time() - self.last_request_time
            return {
                "status": "healthy",
                "worker_running": self.process_manager.is_process_running(),
                "worker_starting": self.process_manager.is_starting,
                "target_port": self.target_port,
                "last_request": self.last_request_time,
                "idle_time": idle_time,
                "idle_time_left": self.idle_timeout - idle_time,
            }

        @self.app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
        async def proxy_vllm(request: Request, path: str, authorized: bool = Depends(self.verified_api_token)):
            """Proxy requests to vLLM server."""
            return await self._handle_proxy_request(request, path)

    def _is_streaming_request(self, body: bytes) -> bool:
        if not body:
            return False

        try:
            # Decode and parse the JSON body
            body_str = body.decode('utf-8').strip()
            if not body_str.startswith('{'):
                return False

            body_json = json.loads(body_str)

            # Check if stream property exists and is True
            return body_json.get('stream', False) is True

        except (json.JSONDecodeError, UnicodeDecodeError):
            return False

    async def _handle_proxy_request(self, request: Request, path: str):
        """Handle proxy requests to vLLM server."""
        self.last_request_time = time.time()

        # Check if server is starting or running
        if not self.process_manager.is_process_starting_or_running():
            # Server is not started yet, start it
            if not self.vllm_command:
                raise HTTPException(
                    status_code=503,
                    detail="vLLM command not set. Server cannot be started."
                )

            logger.info("Starting vLLM server for incoming request...")
            success = await self.process_manager.start_vllm_server(self.vllm_command)
            if not success:
                raise HTTPException(
                    status_code=503,
                    detail="Failed to start vLLM server"
                )

        # Always wait for vLLM server to be fully ready
        await self._wait_for_vllm_ready()

        # Forward the request
        target_url = f"http://localhost:{self.target_port}/v1/{path}"

        try:
            # Get request body
            body = await request.body()

            # Prepare headers (exclude host and content-length)
            headers = {
                key: value for key, value in request.headers.items()
                if key.lower() not in ["host", "content-length"]
            }

            # Check if this should be a streaming request
            if self._is_streaming_request(body):
                # Handle streaming responses
                return StreamingResponse(
                    self._stream_from_vllm(
                        method=request.method,
                        url=target_url,
                        params=request.query_params,
                        headers=headers,
                        content=body
                    ),
                    media_type="text/event-stream"
                )
            else:
                # Make the request to vLLM server for non-streaming
                response = await self.client.request(
                    method=request.method,
                    url=target_url,
                    params=request.query_params,
                    headers=headers,
                    content=body
                )

            # Handle regular responses
            return JSONResponse(
                content=response.json() if response.headers.get(
                    "content-type", "").startswith("application/json") else response.text,
                status_code=response.status_code,
                headers={k: v for k, v in response.headers.items()
                         if k.lower() != "content-length"}
            )

        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail="Cannot connect to vLLM server"
            )
        except Exception as e:
            logger.error(f"Proxy request failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Proxy request failed: {str(e)}"
            )

    async def _stream_from_vllm(self, method, url, params, headers, content):
        """Stream response from vLLM server with proper context management."""
        async with self.client.stream(
            method=method,
            url=url,
            params=params,
            headers=headers,
            content=content
        ) as response:
            async for chunk in response.aiter_bytes():
                logger.debug(f"Streaming chunk: {chunk}")
                yield chunk

    async def _stream_response(self, response: httpx.Response):
        """Stream response from vLLM server."""
        async for chunk in response.aiter_bytes():
            logger.debug(f"Streaming chunk: {chunk}")
            yield chunk

    async def _wait_for_vllm_ready(self):
        """Wait for vLLM server to be ready to accept requests."""
        start_time = time.time()
        last_logged_at = start_time

        while True:
            try:
                ping_url = f"http://localhost:{self.target_port}{self.ping_path}"
                response = await self.client.get(ping_url, timeout=5.0)
                if response.status_code == 200:
                    logger.info("vLLM server is ready")
                    return
            except Exception as e:
                logger.debug(f"vLLM server is not ready yet: {e}")
                pass

            if time.time() - last_logged_at > 60:
                logger.info("Waited for 60s, vLLM server still starting...")
                last_logged_at = time.time()
            await asyncio.sleep(1)

    def _start_idle_monitor(self):
        """Start the idle timeout monitor."""
        asyncio.create_task(self._idle_monitor())

    async def _idle_monitor(self):
        """Monitor for idle timeout and shutdown vLLM server."""
        while True:
            await asyncio.sleep(60)  # Check every minute

            if self.process_manager.is_process_running():
                idle_time = time.time() - self.last_request_time
                if idle_time > self.idle_timeout:
                    logger.info(
                        f"Shutting down vLLM server after {idle_time:.0f}s of inactivity")
                    await self.process_manager.stop_vllm_server()

    def set_vllm_command(self, command: list):
        """Set the vLLM command to use when starting the server."""
        self.vllm_command = command

    async def cleanup(self):
        """Clean up resources."""
        await self.client.aclose()
        await self.process_manager.cleanup()
