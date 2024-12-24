import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import os
import yaml

import sys
sys.path.append(".")
sys.path.append("..")

from models.vqgan_fcm import VQGANFCM
from datasets.general_dataloader import load_data
from datasets.statistic import mean, std
from losses.lpips import LPIPS
from utils import save_model
from accelerate import Accelerator, DistributedDataParallelKwargs

#python favae_scripts/test_favae.py --ds test_output -ckpt_path output/output/best.pt --test_file pkl_files/mydataset_test.pkl 
# 定义测试函数
@torch.no_grad()
def test(loader, model, lpips, perceptual_weight, writer, accelerator, args):
    model.eval()
    device = accelerator.device
    total, loss_l1s, loss_perceptuals, loss_recons = torch.zeros(4).to(device)

    total_steps = 0
    for step, (x) in enumerate(loader):
        x = x.to(device)
        x_recon, _, _, _, _, _ = model(x, stage=0)
        loss_l1 = (x - x_recon).abs().mean()
        loss_perceptual = lpips(x, x_recon).mean()
        loss_recon = loss_l1 + perceptual_weight * loss_perceptual

        loss_l1s += loss_l1 * x.shape[0]
        loss_perceptuals += loss_perceptual * x.shape[0]
        loss_recons += loss_recon * x.shape[0]
        total += x.shape[0]

        total_steps += 1

        # 每一定步数保存重建图像（可根据需要调整）
        if step % args.img_steps == 0 and accelerator.is_main_process:
            std1 = torch.tensor(std).view(1, -1, 1, 1).cuda()
            mean1 = torch.tensor(mean).view(1, -1, 1, 1).cuda()
            x_recon = x_recon * std1 + mean1  # [B, C, H, W]
            x = x * std1 + mean1
            img = torch.cat([x, x_recon], dim=0).clamp(0, 1)
            img = make_grid(img, x.size(0))
            writer.add_image("test/img-recon", img, step)
            writer.flush()

    loss_recon = loss_recons.item() / total.item()
    loss_l1 = loss_l1s.item() / total.item()
    loss_perceptual = loss_perceptuals.item() / total.item()

    metrics = {'loss_recon': loss_recon, 'loss_l1': loss_l1, 'loss_perceptual': loss_perceptual}

    if accelerator.num_processes > 1:
        # 同步多进程的指标
        metrics_order = sorted(metrics.keys())
        metrics_tensor = torch.zeros(1, len(metrics), device=device, dtype=torch.float)
        for i, metric_name in enumerate(metrics_order):
            metrics_tensor[0, i] = metrics[metric_name]
        metrics_tensor = accelerator.gather(metrics_tensor)
        metrics_tensor = metrics_tensor.mean(dim=0)
        for i, metric_name in enumerate(metrics_order):
            metrics[metric_name] = metrics_tensor[i].item()

    if accelerator.is_main_process:
        writer.add_scalar("test/loss_recon", loss_recon, 0)
        writer.add_scalar("test/loss_l1", loss_l1, 0)
        writer.add_scalar("test/loss_perceptual", loss_perceptual, 0)
        print(f"=== Test: loss_recon {metrics['loss_recon']:.3f}, loss_l1 {metrics['loss_l1']:.3f}, loss_perceptual {metrics['loss_perceptual']:.3f}")

    return metrics['loss_recon']

def main(args, save_path):
    torch.manual_seed(0)
    ########################################
    # 初始化Accelerator
    ########################################
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    ########################################
    # 初始化TensorBoard（如果需要记录测试结果）
    ########################################
    writer = None
    if accelerator.is_main_process:
        log_path = os.path.join(save_path, "test_runs")
        writer = SummaryWriter(log_path, comment="test_results")

    ########################################
    # 初始化FA-VAE模型
    ########################################
    if args.downsample_factor == 16:
        ch_mult = (1, 1, 2, 2, 4)
        attn_resolutions = [16]
    elif args.downsample_factor == 4:
        ch_mult = (1, 2, 4)
        attn_resolutions = []
    elif args.downsample_factor == 8:
        ch_mult = (1, 2, 2, 4)
        attn_resolutions = [32]

    # 带有FCM
    if args.with_fcm:
        model = VQGANFCM(args.codebook_size, args.embed_dim, args.double_z, ch_mult=ch_mult, attn_resolutions=attn_resolutions, use_cosine_sim=args.use_cosine_sim, codebook_dim=args.codebook_dim,
                    orthogonal_reg_weight=args.orthogonal_reg_weight, orthogonal_reg_max_codes=args.orthogonal_reg_max_codes, use_l2_quantizer=args.use_l2_quantizer,
                    commitment_weight=args.codebook_weight, use_non_pair_conv=args.use_non_pair_conv, kernel_size=args.gaussian_kernel, dsl_init_sigma=args.dsl_init_sigma,
                    device=accelerator.device, use_gauss_resblock=args.use_gauss_resblock, use_gauss_attn=args.use_gauss_attn, use_same_conv_gauss=args.use_same_conv_gauss,
                    use_same_gauss_resblock=args.use_same_gauss_resblock, use_patch_discriminator=args.use_patch_discriminator, disc_n_layers=args.disc_n_layers,
                    use_ffl_with_fcm=args.use_ffl_with_fcm, num_groups=args.num_groups)

    # 加载预训练模型权重
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = accelerator.prepare(model)

    ########################################
    # 初始化数据加载器
    ########################################
    _, test_loader = load_data(args)
    test_loader = accelerator.prepare(test_loader)

    ########################################
    # 初始化感知损失
    ########################################
    lpips = LPIPS().cuda().eval()

    ########################################
    # 测试模型
    ########################################
    test(test_loader, model, lpips, args.perceptual_weight, writer, accelerator, args)

    if writer:
        writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FA-VAE")
    parser.add_argument("--ds", type=str, help="path to save outputs (ckpt, tensorboard runs)")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--downsample_factor", type=int, default=16, help="downsample factor for FA-VAE")
    parser.add_argument("--perceptual_weight", type=float, default=1.0, help="the lpips weight")
    parser.add_argument("--codebook_weight", type=float, default=1.0, help="the codebook weight")
    parser.add_argument("--codebook_size", type=int, default=16384, help="the number of codebook entries")
    parser.add_argument("--embed_dim", type=int, default=256, help="the dimension of codebook entries")
    parser.add_argument("--codebook_dim", type=int, default=None, help="for projection in VitVQGAN: codebook dim is the dimension to be projected")
    parser.add_argument("--resolution", type=int, default=256, help="image resolution")
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers for data loading")
    parser.add_argument("--img_steps", type=int, default=100, help="log test images to tensorboard frequency")
    parser.add_argument("--ckpt_path", type=str, required=True, help="path to the checkpoint file")
    parser.add_argument("--test_file", type=str, required=True, help="the test file")

    args = parser.parse_args()

    # 保存结果的路径
    save_path = os.path.join("output/{}/".format(args.ds))
    os.makedirs(save_path, exist_ok=True)

    main(args, save_path)