from typing import Optional

import torch
from loguru import logger


def parse_torch_device(device_str: Optional[str]):
    if device_str and device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA device not available. Falling back to CPU")
            return torch.device("cpu")
        else:
            device = torch.device(device_str)
            logger.debug(f"Use CUDA device {device}")
            return device

    logger.debug("Use CPU")
    return torch.device("cpu")
