import os
import wandb
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from TE_unrolled_net.UnrolledNet import UnrolledNet
from configs import Config
from utils import *

# ====================== CONFIGURATION ======================

# Command-line arguments
conf = Config().parse()

if conf.spirit:
    from DC_SPIRIT import DC_SPIRIT as Data_consistency
else:
    from DC_SENSE import DC_SENSE as Data_consistency

os.environ['CUDA_VISIBLE_DEVICES'] = conf.cuda
model_device = 'cuda:0'
data_device = 'cpu'

torch.manual_seed(conf.seed)

#Prepare paths to output results locally
output_path = Path(f"./saved_results/{conf.wandb_group}/{conf.name}/")
results_path = output_path / f"example_slices/" #to save example reconstructions
output_path.mkdir(parents=True, exist_ok=True)
results_path.mkdir(parents=True, exist_ok=True)

if conf.wandb:
    wandb.init(
    project = "parallel_recon",
    entity = "avela019-umn",
    group = conf.wandb_group,
    job_type = "train",
    name = conf.name, 
    config = vars(conf)
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")

# ====================== DATASET SETUP ======================
res = "1pt17"
subj = 2
acceleration = conf.ipa
mb_factor = 5
base_dir = Path("/home/naxos2-raid40/avela019/rafael_data/fMRI_2026/dwi_data_v2")
base_dir.mkdir(exist_ok=True, parents=True)
data_path = base_dir / f"dwi_{res}_MB{mb_factor}R{acceleration}_subj{subj}.h5"

train_dataset = Zeroshot_dataset(data_path, data_device)
train_loader = Zeroshot_dataloader(train_dataset, conf.batch_size, device = model_device)

print("LEN")
print(len(train_loader))

n_iter = conf.n_masks * conf.n_train // conf.batch_size

# ====================== MODEL & OPTIMIZER ======================

dc = Data_consistency()
unrolled_model = UnrolledNet(train_dataset.size).to(model_device)
optimizer = torch.optim.AdamW(unrolled_model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
n_params = sum(p.numel() for p in unrolled_model.parameters() if p.requires_grad)
print("Number of parameters: ", n_params)

# ====================== PREPARE METRICS ======================

train_metrics = []
val_metrics = []
iter = 0

# ====================== MAIN TRAINING LOOP ======================
if conf.wandb:
    plot_slices_zeroshot(train_loader, unrolled_model, epoch = 0)

for epoch in range(conf.n_epochs):

    avg_loss = 0.0
    unrolled_model.train()
    train_loader.restart_count(epoch)
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{conf.n_epochs} [Training]")

    for ksp, coils, theta_mask, lambda_mask in train_loader:
   
        # Get undersampled measurements
        y_theta  = ksp*theta_mask
        y_lambda = ksp*lambda_mask

        args = (y_theta, coils, theta_mask)
        output, mu = unrolled_model(args)

        # Apply forward model to the output
        enc_output = dc.E(output, coils, lambda_mask)

        # Evaluate loss
        loss = L1_L2_norm(enc_output, y_lambda)

        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_bar.set_postfix(loss=loss.item())
        avg_loss += loss.item()

        if conf.wandb:
            wandb.log({"epoch": epoch, "iter": iter,"train/loss_iter": loss.item()})

        iter+=1

    if conf.wandb:
        if (epoch % conf.plot_freq == 0 or epoch == conf.n_epochs-1 or epoch in [1,2]):
           plot_slices_zeroshot(train_loader, unrolled_model, epoch = epoch, n_slices = conf.n_plot)

        wandb.log({"epoch": epoch,"train/loss": avg_loss/len(train_bar)})
        wandb.log({"epoch": epoch,"train/mu": mu})

        #if (epoch % conf.val_freq == 0 or epoch == conf.n_epochs-1 or epoch in [1,2]):
        #    val_loss = validate_model(test_loader, unrolled_model)
        #    val_psnr, val_ssim = test_psnr_ssim(test_loader, unrolled_model)
        #    wandb.log({"epoch": epoch,"val/loss": val_loss})
        #    wandb.log({"epoch": epoch,"val/psnr": val_psnr})
        #    wandb.log({"epoch": epoch,"val/ssim": val_ssim})

        if (epoch % conf.save_freq == 0 or epoch == conf.n_epochs-1):
            torch.save(unrolled_model.state_dict(), output_path / f"unrolled_model.pth")
   
print("---------DONE TRAINING---------")


