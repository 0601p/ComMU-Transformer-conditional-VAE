import torch
import os
import pickle
import argparse
from commu.model.model import MemTransformerLM
from commu.model.config_helper import get_default_cfg_training
from commu.model.dataset import ComMUDataset
from commu.model.RMSProp_dynamic import gradstat

def parse_args():
    parser = argparse.ArgumentParser(description="Dynamic Evaluation")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True, 
        help="location of the data corpus"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Base directory to save the trained model.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="save directory"
    )
    args = parser.parse_args()
    return args



args = parse_args()
cfg = get_default_cfg_training()
dataset = ComMUDataset(args.data_dir, cfg)
train_iter = dataset.get_iterator(
    cfg.OPTIM.batch_size, cfg.OPTIM.seq_len, cfg.OPTIM.device, "train", False
)

model = MemTransformerLM(cfg, dataset._vocab).to(cfg.OPTIM.device)
checkpoint = torch.load(args.checkpoint_path,  map_location=torch.device('cuda'))
model.load_state_dict(checkpoint["model"], strict=False)

MS, data0, decrate = gradstat(model, cfg.OPTIM, train_iter())

with open(os.path.join(args.save_dir, "MS.pt"), "wb") as f:
    pickle.dump(MS, f)

with open(os.path.join(args.save_dir, "data0.pt"), "wb") as f:
    pickle.dump(data0, f)

with open(os.path.join(args.save_dir, "decrate.pt"), "wb") as f:
    pickle.dump(decrate, f)