#!/usr/bin/env python3
"""
Qwen3 Server

This script runs a vLLM server with OpenAI-compatible API for Qwen3 inference.
Qwen3 is a text-only reasoning model served via vLLM, used for thinking-based
time series analysis with injection support (continue_final_message).
"""

import os
import sys
import time
import signal
import argparse
import subprocess
from pathlib import Path

# Parse command line arguments
parser = argparse.ArgumentParser(description="Qwen3 Server")
parser.add_argument("--model_path", type=str, required=True, help="Path to Qwen3 model")
parser.add_argument("--port", type=int, default=5001, help="Port to run server on")
parser.add_argument("--device", type=str, default="0,1,2,3", help="GPU device IDs")
parser.add_argument("--data_parallel_size", type=int, default=2, help="Data parallel size")
parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Tensor parallel size")
parser.add_argument("--context_length", type=int, default=32768, help="Max context length")
parser.add_argument("--pid_file", type=str, default="/tmp/qwen3_server.pid", help="File to store server PID")
parser.add_argument("--log_file", type=str, default=None, help="File to log server output")
parser.add_argument("--initial_wait", type=int, default=120, help="Initial wait time in seconds")
parser.add_argument("--chat_template", type=str, default=None, help="Path to custom chat template file")

args = parser.parse_args()

# Print all args for debugging
print("Arguments received:")
for arg in vars(args):
    print(f"  {arg}: {getattr(args, arg)}")

# Set up GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
print(f"Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

# Using V1 without multiprocessing
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V1"] = "1"
print("Enabled vLLM v1 engine with environment variables:")
print(f"  VLLM_ENABLE_V1_MULTIPROCESSING={os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING')}")
print(f"  VLLM_USE_V1={os.environ.get('VLLM_USE_V1')}")

# Check if vLLM is available
try:
    import vllm
    print(f"vLLM package found. Using vLLM for Qwen3 server.")
    subprocess.run(["vllm", "--version"], capture_output=True, check=False)
    print("vLLM CLI tool is available.")
except ImportError:
    print("Error: vLLM is not installed. Please install vllm.")
    sys.exit(1)
except subprocess.CalledProcessError:
    print("Warning: vLLM CLI tool not found or not working properly. Continuing anyway...")
except FileNotFoundError:
    print("Warning: vLLM CLI tool not found in PATH. Continuing anyway...")

# Create log file directory if needed
if args.log_file:
    log_dir = os.path.dirname(args.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_file = open(args.log_file, 'w')
else:
    log_file = None

# Write PID to file for cleanup
with open(args.pid_file, "w") as f:
    f.write(str(os.getpid()))
print(f"Server PID {os.getpid()} written to {args.pid_file}")

# Graceful shutdown handler
def signal_handler(sig, frame):
    print(f"Received signal {sig}, shutting down...")
    if server_process and server_process.poll() is None:
        server_process.terminate()
        server_process.wait(timeout=10)

    if os.path.exists(args.pid_file):
        os.remove(args.pid_file)

    if log_file:
        log_file.close()

    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def start_vllm_server():
    """Start the vLLM server with OpenAI-compatible API"""

    env = os.environ.copy()

    cmd = [
        "vllm", "serve", args.model_path,
        "--served-model-name", "qwen3",
        "--trust-remote-code",
        "--max-model-len", str(args.context_length),
        "--gpu-memory-utilization", "0.95",
        # Uncomment for RoPE scaling when context exceeds 32k (e.g., TSEvol benchmark):
        # "--rope-scaling", '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}',
        "--host", "0.0.0.0",
        "--port", str(args.port),
        "--uvicorn-log-level", "debug",
        "--data-parallel-size", str(args.data_parallel_size),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
    ]

    # Add chat template if specified
    if args.chat_template:
        chat_template_path = os.path.abspath(args.chat_template)
        if os.path.exists(chat_template_path):
            cmd.extend(["--chat-template", chat_template_path])
            print(f"Using custom chat template from: {chat_template_path}")
        else:
            print(f"Warning: Specified chat template file '{chat_template_path}' not found. Using default template.")

    print(f"Starting vLLM server with command: {' '.join(cmd)}")
    print(f"Data Parallel Size: {args.data_parallel_size}, Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"GPU Configuration: {args.device}")

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=log_file if log_file else subprocess.STDOUT
    )

    return process

def check_server_health(max_retries=60, retry_interval=5):
    """Check if the server is healthy by polling the health endpoint"""
    import requests
    from requests.exceptions import ConnectionError

    initial_wait = args.initial_wait
    print(f"Waiting {initial_wait} seconds for initial model loading...")
    time.sleep(initial_wait)

    print(f"Checking if server is ready at http://localhost:{args.port}/v1/models...")

    for i in range(max_retries):
        try:
            response = requests.get(f"http://localhost:{args.port}/v1/models", timeout=10)
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except ConnectionError:
            pass
        except requests.exceptions.Timeout:
            print("Request timed out. Server might be busy loading the model.")

        print(f"Server not ready yet, retrying in {retry_interval} seconds... ({i+1}/{max_retries})")
        time.sleep(retry_interval)

    print("Server failed to start within the expected time")
    return False

if __name__ == "__main__":
    server_process = start_vllm_server()

    if not check_server_health():
        print("Failed to start server, exiting")
        if server_process and server_process.poll() is None:
            server_process.terminate()

        if os.path.exists(args.pid_file):
            os.remove(args.pid_file)

        if log_file:
            log_file.close()

        sys.exit(1)

    try:
        while server_process.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

    exit_code = server_process.returncode
    print(f"Server process exited with code {exit_code}")

    if os.path.exists(args.pid_file):
        os.remove(args.pid_file)

    if log_file:
        log_file.close()

    sys.exit(exit_code)
