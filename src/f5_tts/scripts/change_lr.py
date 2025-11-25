import torch

# 放到ckpts下执行
checkpoint = torch.load("model_last.pt", weights_only=True)
schedulers  = checkpoint["scheduler_state_dict"]["_schedulers"]
optim = checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"] = 7e-6
schedulers[1]["total_iters"] = 1000000
schedulers[1]["_last_lr"] = [7e-6]
torch.save(checkpoint, "model_last.pt")