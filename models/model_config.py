from detectron2.config import get_cfg

def setup_model_cfg():
    """
    Configuration for Solar SegFormer model.
    """
    cfg = get_cfg()

    # Model Settings
    cfg.MODEL.ARCHITECTURE = "SolarSegFormer"
    cfg.MODEL.BACKBONE = "nvidia/mit-b0" # SegFormer B0
    cfg.MODEL.NUM_CLASSES = 2

    # Training Hyperparameters
    cfg.SOLVER.IMS_PER_BATCH = 12
    cfg.SOLVER.BASE_LR = 0.0003
    cfg.SOLVER.WEIGHT_DECAY = 0.05
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.STEPS = (7000, 9000)

    return cfg
