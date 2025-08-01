import os
import random
import numpy as np
import torch
import logging

def set_seed(seed: int = 42):
    """난수 시드 고정 (Python, NumPy, PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

def ensure_dir(path: str):
    """디렉토리 없으면 생성."""
    os.makedirs(path, exist_ok=True)

def save_checkpoint(model: torch.nn.Module, path: str):
    """모델 state_dict 저장."""
    ensure_dir(os.path.dirname(path))
    torch.save(model.state_dict(), path)

def load_checkpoint(model: torch.nn.Module, path: str, device: str = "cpu") -> torch.nn.Module:
    """모델 state_dict 불러오기."""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model

def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """콘솔 출력용 로거 생성."""
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        fmt = "[%(asctime)s] %(levelname)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
