import logging
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

import yaml
import argparse

import torch
import torch.distributed as dist

def setup_logger(log_file, log_name="TrainingLogger"):
    """
    Sets up logging to both console and a specified log file

    Args:
        log_file (str): Path to the log file
        log_name (str): Name of the logger
    """
    rank = int(os.environ.get("RANK", 0))

    # Set up logger
    logger = logging.getLogger(log_name)

    # Prevent non-main process from logging non ERROR messages
    if rank != 0:
        logger.setLevel(logging.ERROR)
        # Prevent logs from propagating to the root logger
        logger.propagate = False 
    else:
        logger.setLevel(logging.INFO)

    # Formatting
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if rank == 0:
        # File Handler (writes to train.log)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console Handler (prints to terminal)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def parse_args():
    """
    Parses command-line arguments, specifically the config file path.
    """
    
    parser = argparse.ArgumentParser(description="Requires config.yml path to initialize training process")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the YAML configuration file"
    )
    args = parser.parse_args()
    return args

def load_config(config_path):
    """
    Loads configuration from a YAML file

    Args:
        config_path (str): Path to the config.yml file
    """

    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Error: The configuration file '{config_path}' was not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        sys.exit(1)

    return config_data

def init_dist(backend):
    """Initializes the distributed backend."""
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
        # Set the device for the current process
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

def main():
    """
    Initializes accodring to config.yml
    Sets up logging, all paths, and starts training process
    Once training process is complete, runs validation scripts
    """
    
    # Access config.yml for configurations
    args = parse_args()
    config_path = args.config
    
    config_data = load_config(config_path)
    logger.info(f"Configuration loaded from {config_path}")

    # Initialize DDP and CUDA
    init_dist(config_data["os_backend"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    logger.info(f"Distributed process initialized on device {device}")

    # Destroy the process group after training is complete
    dist.destroy_process_group()
    logger.info("Distributed process completed.")

    return 0

if __name__ == "__main__":
    logger = setup_logger("train.log")
    logger.info("Logger initialized.")
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving and exiting...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Process crashed with error: {e}")
        sys.exit(1)