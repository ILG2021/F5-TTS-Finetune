import torch

# 放到ckpts下执行
checkpoint = torch.load("model_last.pt", weights_only=True)
print(checkpoint.keys())
schedulers  = checkpoint["scheduler_state_dict"]["_schedulers"]
print(schedulers)