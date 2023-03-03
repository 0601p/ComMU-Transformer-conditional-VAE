from yacs.config import CfgNode as CN


def model(cfg):
    # For model
    cfg.MODEL = CN()
    cfg.MODEL.num_layers = 6
    cfg.MODEL.num_heads = 10
    cfg.MODEL.units = 500
    cfg.MODEL.inner_size = 1000
    cfg.MODEL.dropout = 0.1
    cfg.MODEL.attention_dropout = 0.1
    cfg.MODEL.clamp_len = -1
    cfg.MODEL.same_length = False
    return cfg


def train(cfg):
    # For training
    cfg.TRAIN = CN()
    cfg.TRAIN.batch_size = 256
    cfg.TRAIN.batch_chunk = 4
    cfg.TRAIN.tgt_length = 128
    cfg.TRAIN.mem_length = 1024
    cfg.TRAIN.seed = 1111
    cfg.TRAIN.lr = 0.004
    cfg.TRAIN.lr_min = 0.0001
    cfg.TRAIN.warmup_step = 100
    cfg.TRAIN.clip = 1.0
    cfg.TRAIN.max_step = 20000
    cfg.TRAIN.log_interval = 100
    cfg.TRAIN.eval_interval = 1000
    cfg.TRAIN.weight_decay = 0.0
    return cfg

def optim(cfg):
    # config for optimizer - dynamic evaluation with RMSProp
    cfg.OPTIM = CN()
    cfg.OPTIM.lamb = 0.002      # decay parameter lambda ( dynamic evaluation )
    cfg.OPTIM.lr = 0.00005      # learning rate eta ( dynamic evaluation )
    cfg.OPTIM.epsilon = 0.00002 # stabilization parameter epsilon ( dynamic evaluation )
    cfg.OPTIM.max_step = 100
    cfg.OPTIM.batch_size = 64
    cfg.OPTIM.seq_len = 128
    cfg.OPTIM.device = 'cuda'
    return cfg


def init(cfg):
    # For initialization
    cfg.INITIALIZER = CN()
    cfg.INITIALIZER.base_init = 0.01
    cfg.INITIALIZER.embed_init = 0.01

    # For evaluation
    cfg.EVALUATE = CN()
    cfg.EVALUATE.batch_size = 10
    cfg.EVALUATE.tgt_length = 128
    cfg.EVALUATE.mem_length = 2048

    return cfg


def get_default_cfg_training():
    cfg = CN()
    cfg = init(cfg)
    cfg = model(cfg)
    cfg = train(cfg)
    cfg = optim(cfg)
    cfg.freeze()
    return cfg


def get_default_cfg_inference():
    """Get a yacs CfgNode object with default values."""
    cfg = CN()

    # # Model related parameters
    cfg.MODEL = CN()
    cfg.MODEL.memory_length = 4146
    cfg.MODEL.device = "cuda"
    # Sampling related parameters
    cfg.SAMPLING = CN()
    cfg.SAMPLING.threshold = 32.0
    cfg.SAMPLING.temperature = 0.95

    # Model related parameters
    cfg.GENERATION = CN()
    cfg.GENERATION.generation_length = 4096
    cfg = optim(cfg)

    cfg.freeze()
    return cfg
