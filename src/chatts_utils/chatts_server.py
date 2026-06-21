#!/usr/bin/env python3
"""
ChatTS Server

This script runs a vLLM server with OpenAI-compatible API for ChatTS inference.
It leverages the timeseries branch of vLLM for serving ChatTS models.
"""

import os
import sys
import time
import signal
import argparse
import json
import subprocess
from pathlib import Path

# Set environment variable for vLLM to allow insecure serialization
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

# Parse command line arguments
parser = argparse.ArgumentParser(description="ChatTS Server")
parser.add_argument("--model_path", type=str, required=True, help="Path to ChatTS model")
parser.add_argument("--chatts_path", type=str, required=True, help="Path to ChatTS directory")
parser.add_argument("--port", type=int, default=5000, help="Port to run server on")
parser.add_argument("--device", type=str, default="0", help="GPU device ID")
parser.add_argument("--context_length", type=int, default=6000, help="Max context length")
parser.add_argument("--pid_file", type=str, default="/tmp/chatts_server.pid", help="File to store server PID")
parser.add_argument("--log_file", type=str, default=None, help="File to log server output")
parser.add_argument("--initial_wait", type=int, default=120, help="Initial wait time in seconds")

args = parser.parse_args()

# Set up GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
print(f"Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"Using VLLM_ALLOW_INSECURE_SERIALIZATION={os.environ.get('VLLM_ALLOW_INSECURE_SERIALIZATION', '0')}")

# Add ChatTS to Python path
sys.path.insert(0, args.chatts_path)

# Check if vLLM is available
try:
    import vllm
    print(f"vLLM package found. Using vLLM for ChatTS server.")
    # Try to run a simple vLLM command to test if it's properly installed
    subprocess.run(["vllm", "--version"], capture_output=True, check=False)
    print("vLLM CLI tool is available.")
except ImportError:
    print("Error: vLLM is not installed. Please install the timeseries branch from https://github.com/xiez22/vllm")
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
    
    # Ensure environment variables are passed to the subprocess
    env = os.environ.copy()
    env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    
    # Determine data-parallel-size based on device parameter
    try:
        # If there are commas in the device parameter, split it
        if ',' in args.device:
            devices = [d.strip() for d in args.device.split(',') if d.strip()]
            data_parallel_size = len(devices)
        else:
            # Single device
            devices = [args.device.strip()]
            data_parallel_size = 1
            
        # Validate that we have at least one device
        if not devices or data_parallel_size < 1:
            print(f"Warning: Invalid device parameter '{args.device}'. Using data-parallel-size=1")
            data_parallel_size = 1
    except Exception as e:
        print(f"Error parsing device parameter: {e}. Using data-parallel-size=1")
        devices = ["0"]
        data_parallel_size = 1
    
    print(f"Device parameter: {args.device}, detected {data_parallel_size} GPUs")
    
    cmd = [
        "vllm", "serve", args.model_path,
        "--served-model-name", "chatts",
        "--trust-remote-code",
        "--hf-overrides", '{"model_type":"chatts"}',
        "--max-model-len", str(args.context_length),
        "--gpu-memory-utilization", "0.95",
        "--limit-mm-per-prompt", f"timeseries=50",
        "--allowed-local-media-path", os.path.abspath(os.getcwd()),
        "--host", "0.0.0.0",
        "--port", str(args.port),
        "--uvicorn-log-level", "debug",
        "--data-parallel-size", str(data_parallel_size)
    ]
    
    print(f"Starting vLLM server with command: {' '.join(cmd)}")
    print(f"Environment: VLLM_ALLOW_INSECURE_SERIALIZATION={env['VLLM_ALLOW_INSECURE_SERIALIZATION']}")
    print(f"Data Parallel Configuration: {data_parallel_size} GPUs ({args.device})")
    
    # Start server process
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
    
    # First, wait for the initial loading period
    initial_wait = args.initial_wait  # Default is 120 seconds
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
    # Start the vLLM server
    server_process = start_vllm_server()
    
    # Check server health
    if not check_server_health():
        print("Failed to start server, exiting")
        if server_process and server_process.poll() is None:
            server_process.terminate()
        
        if os.path.exists(args.pid_file):
            os.remove(args.pid_file)
            
        if log_file:
            log_file.close()
            
        sys.exit(1)
    
    # Keep the script running until the server exits
    try:
        while server_process.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    
    # Server process exited
    exit_code = server_process.returncode
    print(f"Server process exited with code {exit_code}")
    
    # Clean up
    if os.path.exists(args.pid_file):
        os.remove(args.pid_file)
        
    if log_file:
        log_file.close()
        
    sys.exit(exit_code)