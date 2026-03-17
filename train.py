import argparse
import os
import torch
from dataloader import ImageSet
from loss import compute_loss
from perceptual_loss import PerceptualLossModule
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from utils import validate_model, save_checkpoint, load_checkpoint
from utils_model import get_model
import wandb

wandb.init(project="IFBLEND_TRAIN")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="ifblend")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--decay_epoch", type=int, default=70)
    parser.add_argument("--n_steps", type=int, default=2)

    parser.add_argument("--data_src", default="/NTIRE2026/C2_ALN_White")

    parser.add_argument("--res_dir", default="./results")
    parser.add_argument("--ckp_dir", default="./checkpoints")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_cpu", type=int, default=8)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--b1", type=float, default=0.5)
    parser.add_argument("--b2", type=float, default=0.999)

    parser.add_argument("--clip", type=int, default=0)

    # LOSS WEIGHTS
    parser.add_argument("--alpha_1", type=float, default=0.2)   # SSIM
    parser.add_argument("--alpha_2", type=float, default=0.05)  # Gradient
    parser.add_argument("--alpha_3", type=float, default=0.1)   # Content
    parser.add_argument("--alpha_4", type=float, default=0.05)  # Edge
    parser.add_argument("--alpha_5", type=float, default=0.05)  # Color

    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--img_width", type=int, default=256)

    parser.add_argument("--valid_checkpoint", type=int, default=1)
    parser.add_argument("--save_checkpoint", type=int, default=10)

    parser.add_argument("--description", default="IFBlend_ALN")

    parser.add_argument("--resume", type=int, default=0, help="resume training from latest checkpoint")
    parser.add_argument("--resume_epoch", type=int, default=0, help="epoch to resume from")

    opt = parser.parse_args()
    print(opt)

    # -------------------------
    # MODEL
    # -------------------------

    model_net = get_model(opt.model_name)

    if torch.cuda.device_count() >= 1:
        model_net = torch.nn.DataParallel(model_net)

    betas = (opt.b1, opt.b2)

    optimizer = torch.optim.Adam(
        model_net.parameters(),
        lr=opt.lr,
        betas=betas
    )

    step_sz = (opt.n_epochs - opt.decay_epoch) // opt.n_steps
    milestones = [opt.decay_epoch + k * step_sz for k in range(opt.n_steps)]

    scheduler = MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=0.6
    )

    # -------------------------
    # DATASET
    # -------------------------

    train_dataloader = DataLoader(
        ImageSet(
            opt.data_src,
            set_type="train",
            size=(opt.img_height, opt.img_width),
            aug=True
        ),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu
    )

    val_dataloader = DataLoader(
        ImageSet(
            opt.data_src,
            set_type="val",
            size=None,
            aug=False
        ),
        batch_size=1,
        shuffle=False,
        num_workers=opt.n_cpu
    )
    # -------------------------
    # DEVICE
    # -------------------------

    if torch.cuda.is_available():

        Tensor = torch.cuda.FloatTensor

        model_net = model_net.cuda()

        field_loss_module = PerceptualLossModule(
            device=torch.device("cuda")
        )

    else:

        Tensor = torch.FloatTensor

        field_loss_module = PerceptualLossModule(
            device=torch.device("cpu")
        )

    best_psnr = 0

    bst_ckp_path = "{}/{}/best".format(
        opt.ckp_dir,
        opt.description
    )

    os.makedirs(bst_ckp_path, exist_ok=True)

    start_epoch = 0

    # -------------------------
    # RESUME TRAINING
    # -------------------------
    if opt.resume == 1:

        resume_ckpt = "{}/{}/{}/checkpoint.pt".format(
            opt.ckp_dir,
            opt.description,
            opt.resume_epoch
        )

        if os.path.exists(resume_ckpt):

            model_net, optimizer, scheduler = load_checkpoint(
                resume_ckpt,
                model_net,
                optimizer,
                scheduler
            )

            start_epoch = opt.resume_epoch

            print("Resumed training from epoch:", start_epoch)

        else:
            print("Checkpoint not found. Starting from scratch.")
        print("Current LR:", optimizer.param_groups[0]['lr'])    

    # -------------------------
    # TRAINING LOOP
    # -------------------------

    for epoch in range(start_epoch, opt.n_epochs):

        epoch_loss = 0

        for i, batch in enumerate(train_dataloader):

            input_img = Variable(batch[0].type(Tensor))
            gt_img = Variable(batch[1].type(Tensor))

            optimizer.zero_grad()

            out_img = model_net(input_img)

            loss = compute_loss(
                out_img,
                gt_img,
                opt,
                field_loss_module=field_loss_module
            )

            loss.backward()

            if opt.clip == 1:
                torch.nn.utils.clip_grad_norm_(
                    model_net.parameters(),
                    0.01
                )

            optimizer.step()

            epoch_loss += loss.detach().item()

        scheduler.step()

        print("Epoch:", epoch+1, "Training Loss:", epoch_loss)

        # -------------------------
        # VALIDATION
        # -------------------------

        if (epoch + 1) % opt.valid_checkpoint == 0:

            save_disk = (epoch + 1) % opt.save_checkpoint == 0

            if save_disk:

                out_path = "{}/{}/{}".format(
                    opt.res_dir,
                    opt.description,
                    epoch + 1
                )

                ckp_path = "{}/{}/{}".format(
                    opt.ckp_dir,
                    opt.description,
                    epoch + 1
                )

                os.makedirs(out_path, exist_ok=True)
                os.makedirs(ckp_path, exist_ok=True)

                save_checkpoint(
                    ckp_path,
                    model_net,
                    optimizer,
                    scheduler
                )

            else:
                out_path = None

            val_report = validate_model(
                model_net,
                val_dataloader,
                save_disk=save_disk,
                out_dir=out_path
            )

            torch.cuda.empty_cache()

            wandb.log({
                "val_mse": val_report["MSE"],
                "val_psnr": val_report["PSNR"],
                "val_ssim": val_report["SSIM"]
            })

            if val_report["PSNR"] > best_psnr:

                best_psnr = val_report["PSNR"]

                save_checkpoint(
                    bst_ckp_path,
                    model_net,
                    optimizer,
                    scheduler
                )

            print(
                "Epoch:", epoch+1,
                "PSNR:", val_report["PSNR"],
                "SSIM:", val_report["SSIM"]
            )
