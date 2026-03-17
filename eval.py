import argparse
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR
from PIL import Image
import torchvision.utils as vutils

from utils_model import get_model
from utils import load_checkpoint


# ----------------------------------
# Test-Only Dataset (LQ images only)
# ----------------------------------
class TestOnlyDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.to_tensor(img), os.path.basename(self.image_paths[idx])


# ----------------------------------
# Main
# ----------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="ifblend")
    parser.add_argument("--data_src", default="./NTIRE2026/C2_ALN_White/valid/LQ")
    parser.add_argument("--res_dir", default="./final-results")
    parser.add_argument("--ckp_dir", default="./checkpoints")
    parser.add_argument("--load_from", default="IFBlend_ambient6k")

    opt = parser.parse_args()
    print(opt)

    # ----------------------------------
    # Load Model
    # ----------------------------------
    model_net = get_model(opt.model_name)

    if torch.cuda.device_count() >= 1:
        model_net = torch.nn.DataParallel(model_net)

    if torch.cuda.is_available():
        model_net = model_net.cuda()

    model_net.eval()

    # ----------------------------------
    # Dummy optimizer + scheduler
    # (Required to load checkpoint)
    # ----------------------------------
    optimizer_dummy = torch.optim.Adam(model_net.parameters(), lr=0.0002)
    scheduler_dummy = MultiStepLR(optimizer_dummy, milestones=[], gamma=0.1)

    load_model_checkpoint = "{}/{}/best/checkpoint.pt".format(
        opt.ckp_dir, opt.load_from
    )

    model_net, _, _ = load_checkpoint(
        load_model_checkpoint,
        model_net,
        optimizer_dummy,
        scheduler_dummy
    )

    print("Checkpoint loaded successfully.")

    # ----------------------------------
    # Output directory
    # ----------------------------------
    out_path = "{}/{}/".format(opt.res_dir, opt.load_from)
    os.makedirs(out_path, exist_ok=True)

    # ----------------------------------
    # DataLoader
    # ----------------------------------
    test_dataset = TestOnlyDataset(opt.data_src)

    val_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    print("Number of test images:", len(test_dataset))

    # ----------------------------------
    # Inference
    # ----------------------------------
    with torch.no_grad():
        for inp, name in val_dataloader:

            if torch.cuda.is_available():
                inp = inp.cuda()

            output = model_net(inp)

            save_path = os.path.join(out_path, name[0])
            vutils.save_image(output, save_path)

            print("Saved:", name[0])

    print("\nInference completed successfully.")
    print("Results saved to:", out_path)
