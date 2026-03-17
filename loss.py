'''import numpy as np
import torch
from pytorch_msssim import ssim
import torch.nn.functional as F
import wandb


def get_image_gradients(x):
    dx = torch.Tensor([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]])

    dx = dx.view((1, 1, 3, 3))

    dy = torch.Tensor([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]])

    dy = dy.view((1, 1, 3, 3))

    b, c, h, w = x.shape
    G_x = torch.zeros_like(x)
    G_y = torch.zeros_like(x)

    if torch.cuda.is_available():
        x = x.cuda()
        dx = dx.cuda()
        dy = dy.cuda()
        G_x = G_x.cuda()
        G_y = G_y.cuda()

    for k in range(b):
        for i in range(c):
            G_x[k, i, :, :] = F.conv2d(x[k, i, :, :].unsqueeze(0), dx, padding=1)
            G_y[k, i, :, :] = F.conv2d(x[k, i, :, :].unsqueeze(0), dy, padding=1)

    return G_x, G_y


def compute_gradient_loss(out, gt):
    dout_dx, dout_dy = get_image_gradients(out)
    dgt_dx, dgt_dy = get_image_gradients(gt)
    return F.l1_loss(dout_dx, dgt_dx) + F.l1_loss(dout_dy, dgt_dy)


def compute_ssim_loss(out, gt):
    return 1 - ssim(out, gt, data_range=1, size_average=True)


def compute_loss(out, gt, opt, mode='l1', field_loss_module=None):
    if mode == 'l1':
        reconstruction_loss = F.l1_loss(out, gt)
    else:
        reconstruction_loss = F.mse_loss(out, gt)

    if opt.alpha_1 > 0:
        ssim_loss = compute_ssim_loss(out, gt)
    else:
        ssim_loss = 0

    if opt.alpha_2 > 0:
        grad_loss = compute_gradient_loss(out, gt)
    else:
        grad_loss = 0

    content_loss = field_loss_module.compute_content_loss(out, gt)
    wandb.log({
		"PIX loss": reconstruction_loss,
		"CNT loss": content_loss,
		"SSIM loss": ssim_loss,
		"GRAD loss": grad_loss
		
	})
    
    return reconstruction_loss + opt.alpha_1 * ssim_loss + opt.alpha_2 * grad_loss + opt.alpha_3 * content_loss


if __name__ == '__main__':
    out = torch.rand((5, 3, 256, 256))
    gt = torch.rand((5, 3, 256, 256))

    outx, outy = get_image_gradients(out)
    print(outx.shape, outy.shape)
'''




'''import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
import wandb


# -----------------------------
# MS-SSIM Loss
# -----------------------------
def compute_ssim_loss(out, gt):
    return 1 - ms_ssim(out, gt, data_range=1.0, size_average=True)


# -----------------------------
# Edge Loss (Sobel)
# -----------------------------
def compute_edge_loss(out, gt):
    device = out.device
    c = out.shape[1]

    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=device)

    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32, device=device)

    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1)

    grad_out_x = F.conv2d(out, sobel_x, padding=1, groups=c)
    grad_out_y = F.conv2d(out, sobel_y, padding=1, groups=c)

    grad_gt_x = F.conv2d(gt, sobel_x, padding=1, groups=c)
    grad_gt_y = F.conv2d(gt, sobel_y, padding=1, groups=c)

    return F.l1_loss(grad_out_x, grad_gt_x) + F.l1_loss(grad_out_y, grad_gt_y)


# -----------------------------
# Contrast Loss
# -----------------------------
def compute_contrast_loss(out, gt):
    out_mean = torch.mean(out, dim=[2, 3], keepdim=True)
    gt_mean = torch.mean(gt, dim=[2, 3], keepdim=True)

    out_std = torch.sqrt(torch.mean((out - out_mean) ** 2, dim=[2, 3]) + 1e-8)
    gt_std = torch.sqrt(torch.mean((gt - gt_mean) ** 2, dim=[2, 3]) + 1e-8)

    return F.l1_loss(out_std, gt_std)


# -----------------------------
# Total Loss
# -----------------------------
def compute_loss(out, gt, opt, perceptual_module):

    # Reconstruction
    if opt.recon_type == "l1":
        reconstruction_loss = F.l1_loss(out, gt)
    else:
        reconstruction_loss = F.mse_loss(out, gt)

    # MS-SSIM
    ssim_loss = compute_ssim_loss(out, gt) if opt.alpha_ssim > 0 else 0

    # Edge
    edge_loss = compute_edge_loss(out, gt) if opt.alpha_edge > 0 else 0

    # Contrast
    contrast_loss = compute_contrast_loss(out, gt) if opt.alpha_contrast > 0 else 0

    # Perceptual
    perceptual_loss = (
        perceptual_module.compute_content_loss(out, gt)
        if opt.alpha_perceptual > 0 else 0
    )

    total_loss = (
        reconstruction_loss
        + opt.alpha_ssim * ssim_loss
        + opt.alpha_edge * edge_loss
        + opt.alpha_contrast * contrast_loss
        + opt.alpha_perceptual * perceptual_loss
    )

    wandb.log({
        "Total Loss": total_loss.detach(),
        "Recon Loss": reconstruction_loss.detach(),
        "SSIM Loss": ssim_loss if isinstance(ssim_loss, int) else ssim_loss.detach(),
        "Edge Loss": edge_loss if isinstance(edge_loss, int) else edge_loss.detach(),
        "Contrast Loss": contrast_loss if isinstance(contrast_loss, int) else contrast_loss.detach(),
        "Perceptual Loss": perceptual_loss if isinstance(perceptual_loss, int) else perceptual_loss.detach(),
    })

    return total_loss

'''




import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
import wandb


# -------------------------------
# Sobel Gradient Computation
# -------------------------------
def get_image_gradients(x):

    sobel_x = torch.tensor([[1,0,-1],
                            [2,0,-2],
                            [1,0,-1]], dtype=torch.float32).view(1,1,3,3).to(x.device)

    sobel_y = torch.tensor([[1,2,1],
                            [0,0,0],
                            [-1,-2,-1]], dtype=torch.float32).view(1,1,3,3).to(x.device)

    sobel_x = sobel_x.repeat(x.shape[1],1,1,1)
    sobel_y = sobel_y.repeat(x.shape[1],1,1,1)

    grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.shape[1])
    grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.shape[1])

    return grad_x, grad_y


# -------------------------------
# Gradient Loss
# -------------------------------
def compute_gradient_loss(out, gt):

    dout_dx, dout_dy = get_image_gradients(out)
    dgt_dx, dgt_dy = get_image_gradients(gt)

    return F.l1_loss(dout_dx, dgt_dx) + F.l1_loss(dout_dy, dgt_dy)


# -------------------------------
# SSIM Loss
# -------------------------------
def compute_ssim_loss(out, gt):

    return 1 - ssim(out, gt, data_range=1, size_average=True)


# -------------------------------
# Edge Loss (Laplacian)
# -------------------------------
def compute_edge_loss(out, gt):

    laplacian_kernel = torch.tensor([[0,-1,0],
                                     [-1,4,-1],
                                     [0,-1,0]], dtype=torch.float32).view(1,1,3,3).to(out.device)

    laplacian_kernel = laplacian_kernel.repeat(out.shape[1],1,1,1)

    edge_out = F.conv2d(out, laplacian_kernel, padding=1, groups=out.shape[1])
    edge_gt = F.conv2d(gt, laplacian_kernel, padding=1, groups=gt.shape[1])

    return F.l1_loss(edge_out, edge_gt)


# -------------------------------
# Color Consistency Loss
# -------------------------------
def compute_color_loss(out, gt):

    r_out, g_out, b_out = out[:,0,:,:], out[:,1,:,:], out[:,2,:,:]
    r_gt, g_gt, b_gt = gt[:,0,:,:], gt[:,1,:,:], gt[:,2,:,:]

    loss_rg = F.l1_loss(r_out - g_out, r_gt - g_gt)
    loss_rb = F.l1_loss(r_out - b_out, r_gt - b_gt)
    loss_gb = F.l1_loss(g_out - b_out, g_gt - b_gt)

    return loss_rg + loss_rb + loss_gb


# -------------------------------
# Total Loss Function
# -------------------------------
def compute_loss(out, gt, opt, mode='l1', field_loss_module=None):

    # Pixel reconstruction loss
    if mode == 'l1':
        reconstruction_loss = F.l1_loss(out, gt)
    else:
        reconstruction_loss = F.mse_loss(out, gt)

    # SSIM loss
    ssim_loss = compute_ssim_loss(out, gt) if opt.alpha_1 > 0 else 0

    # Gradient loss
    grad_loss = compute_gradient_loss(out, gt) if opt.alpha_2 > 0 else 0

    # Perceptual / content loss
    content_loss = field_loss_module.compute_content_loss(out, gt)

    # Edge loss
    edge_loss = compute_edge_loss(out, gt) if opt.alpha_4 > 0 else 0

    # Color consistency loss
    color_loss = compute_color_loss(out, gt) if opt.alpha_5 > 0 else 0

    # Log losses to WandB
    wandb.log({
        "PIX loss": reconstruction_loss.item(),
        "CNT loss": content_loss.item(),
        "SSIM loss": ssim_loss.item() if opt.alpha_1 > 0 else 0,
        "GRAD loss": grad_loss.item() if opt.alpha_2 > 0 else 0,
        "EDGE loss": edge_loss.item() if opt.alpha_4 > 0 else 0,
        "COLOR loss": color_loss.item() if opt.alpha_5 > 0 else 0
    })

    # Total loss
    total_loss = (
        reconstruction_loss
        + opt.alpha_1 * ssim_loss
        + opt.alpha_2 * grad_loss
        + opt.alpha_3 * content_loss
        + opt.alpha_4 * edge_loss
        + opt.alpha_5 * color_loss
    )

    return total_loss


# -------------------------------
# Testing the loss file
# -------------------------------
if __name__ == '__main__':

    out = torch.rand((4,3,256,256))
    gt = torch.rand((4,3,256,256))

    gx, gy = get_image_gradients(out)

    print("Gradient X shape:", gx.shape)
    print("Gradient Y shape:", gy.shape)
