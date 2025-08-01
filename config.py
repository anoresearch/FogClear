import os
import math

# ── Hardware ──
DEVICE_ID = "0"

# ── Dataset ──
DATASET = "CIFAR100_LT"
IMB_TYPE = "exp"
IMB_FACTOR = 1.0

# ── Training ──
BATCH_SIZE = 128
NUM_WORKERS = 4
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-3
NUM_CLASSES = 100
NUM_EPOCHS_STAGE_A = 200
NUM_EPOCHS_STAGE_B = 40
PRETRAIN = False

# ── Fog Stage-A ──
FOG_EMA_BETA   = 0.95
T_KAPPA        = 0.4
H_TARGET       = 0.8
THR_ABS_RATIO  = 0.10  # 로그 100 기준 비율

# ── Stage-B ──
FINE_EPOCHS   = 20
DELTA_MAX     = 0.5
ALPHA_S       = 1.0
BETA_S        = 1.0
GAMMA_S       = 0.0
FOG_IMB_ALPHA = 0.5
LCONF         = 0.2
TAU           = 0.7
CLIP_W        = 3.0
EMA_UC        = 0.95

# ── Directories ──
EXP_DIR      = "./exp"
STAGE1_DIR   = os.path.join(EXP_DIR, "stage_1")
STAGE2_DIR   = os.path.join(EXP_DIR, "stage_2")
