import torch
import pickle
import os
from commu.model.model import MemTransformerLM


class RMSProp_dynamic:
    def __init__(self, cfg_optim, save_dir) -> None:
        self.lamb = cfg_optim.lamb
        self.lr = cfg_optim.lr
        self.epsilon = cfg_optim.epsilon
        self.load_data(save_dir)
    
    def load_data(self, save_dir):
        print("Loading dynamic evaluation data ...")
        
        with open(os.path.join(save_dir, "MS.pt"), "rb") as f:
            self.MS = pickle.load(f)

        with open(os.path.join(save_dir, "data0.pt"), "rb") as f:
            self.data0 = pickle.load(f)

        with open(os.path.join(save_dir, "decrate.pt"), "rb") as f:
            self.decrate = pickle.load(f)

        print("Done!")


    def update(self, model):
        for name, param in model.named_parameters():
            dW = self.lamb*self.decrate[name]*(self.data0[name]-param.data)-self.lr*param.grad.data/(self.MS[name]+self.epsilon)
            param.data+=dW



def gradstat(model, cfg_optim, train_iter):


    total_loss = 0

    MS = {}
    data0 = {}
    decrate = {}
    for name, param in model.named_parameters():
        MS[name] = 0*param.data

    mems = None
    model.eval()
    for batch, (data, target, reset_mems, batch_token_num) in enumerate(
        train_iter
    ):
        model.zero_grad()

        ret = model(data, target, reset_mems, mems)
        loss, mems = ret
        loss = loss[target != 0]
        loss = loss.float().mean()
        loss.backward()

        for name, param in model.named_parameters():
            MS[name] += param.grad.data*param.grad.data

        total_loss += loss.item()
        if batch == cfg_optim.max_step:
            break

    gsum = 0
    count = 0

    for name, param in model.named_parameters():

        MS[name] = torch.sqrt(MS[name]/batch)

        gsum+=torch.mean(MS[name])
        count+=1
    gsum/=count

    for name, param in model.named_parameters():
        decrate[name] = MS[name]/gsum
        data0[name] = 1*param.data
    
    return MS, data0, decrate