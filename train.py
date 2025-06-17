import os
import math
import time
import logging
import wandb
from tqdm.auto import tqdm
from PIL import Image
import random
import shutil
from evaluation import evaluate_folder, init_models

import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler

from torchvision.utils import make_grid

from dataset.font_dataset import FontDataset
from dataset.collate_fn import CollateFN
from configs.fontdiffuser import get_parser
from src import (FontDiffuserModel,
                 ContentPerceptualLoss,
                 build_unet,
                 build_style_encoder,
                 build_content_encoder,
                 build_ddpm_scheduler,
                 build_scr)
from utils import (save_args_to_yaml,
                   x0_from_epsilon, 
                   reNormalize_img, 
                   normalize_mean_std)

logger = get_logger(__name__)

def get_args():
    parser = get_parser()
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args

def run_validation(global_step, args, accelerator):
    from sample import arg_parse, load_fontdiffuer_pipeline, sampling
    from evaluation import evaluate_folder

    tmp_gen_dir = os.path.join(args.validation_tmp_dir, "tmp_gen")
    tmp_gt_dir = os.path.join(args.validation_tmp_dir, "tmp_gt")    

    # 1. 准备采样 pipeline
    args_for_sample = arg_parse()
    args_for_sample.ckpt_dir = f"{args.output_dir}/global_step_{global_step}"
    # args_for_sample.ckpt_dir = "/scratch/yl10337/important_result/global_step_200000_R"
    args_for_sample.content_dir = args.validation_content_dir
    args_for_sample.style_dir = args.validation_style_dir
    args_for_sample.save_image = True
    args_for_sample.save_image_dir = tmp_gen_dir
    args_for_sample.device = accelerator.device

    pipe = load_fontdiffuer_pipeline(args=args_for_sample)

    # 2. 清空 tmp_gen 并开始采样
    if os.path.exists(args.validation_tmp_dir):
        shutil.rmtree(args.validation_tmp_dir)
    os.makedirs(tmp_gen_dir, exist_ok=True)
    os.makedirs(tmp_gt_dir, exist_ok=True)

    # content_imgs = sorted(os.listdir(args.validation_content_dir))
    # style_imgs = sorted(os.listdir(args.validation_style_dir))
    # common = list(set(content_imgs) & set(style_imgs))
    # common.sort()
    # for fname in tqdm(common[:args.validation_num_samples]):
    #     args_for_sample.content_image_path = os.path.join(args.validation_content_dir, fname)
    #     args_for_sample.style_image_path = os.path.join(args.validation_style_dir, fname)
    #     sampling(args=args_for_sample, pipe=pipe, output_file=fname)
    import random

    all_ids = list(range(1, 1201))  # 1 to 1200 inclusive
    sampled_ids = sorted(random.sample(all_ids, 1200))  # sample 200 unique ids

    for idx in tqdm(sampled_ids, desc="Validation Sampling"):
        fname = f"{idx}.jpg"  # assumes images are named like '1.png', '2.png', etc.
        args_for_sample.content_image_path = os.path.join(args.validation_content_dir, fname)
        args_for_sample.style_image_path = os.path.join(args.validation_style_dir, fname)
        sampling(args=args_for_sample, pipe=pipe, output_file=fname)

        # 拷贝 ground truth 到 tmp_gt
        shutil.copy(
            os.path.join(args.validation_gt_dir, fname),
            os.path.join(tmp_gt_dir, fname)
        )

    # 3. evaluate
    class DummyArgs:
        folder = True
        rmse = True
        l1 = True
        ssim = True
        psnr = True
        fid = True
        lpips = True
        dino = True
        use_gpu = True

    # eval_result = evaluate_folder(args.validation_gt_dir, args.validation_tmp_dir, DummyArgs())
    # eval_result = evaluate_folder(tmp_gt_dir, args.validation_tmp_dir, DummyArgs())

    feature_extractor = init_models(accelerator.device)  # 你应该已经有这个函数了

    eval_result = evaluate_folder(
        tmp_gt_dir,
        tmp_gen_dir,
        DummyArgs(),
        feature_extractor,
        accelerator.device
    )

    # 4. 拼图 log 四张图像
    if accelerator.is_main_process:
        log_imgs = []
        for i, idx in enumerate(sampled_ids[:min(4, len(sampled_ids))]):
            fname = f"{idx}.jpg"
            c = Image.open(os.path.join(args.validation_content_dir, fname)).convert("RGB")
            s = Image.open(os.path.join(args.validation_style_dir, fname)).convert("RGB")
            g = Image.open(os.path.join(tmp_gen_dir, f"{idx}.jpg.png")).convert("RGB")
            t = Image.open(os.path.join(tmp_gt_dir, fname)).convert("RGB")
            quad = transforms.ToTensor()(Image.fromarray(np.concatenate([np.array(c), np.array(s), np.array(g), np.array(t)], axis=1)))
            log_imgs.append(wandb.Image(quad, caption=f"Val {fname}"))

        wandb.log({"Validation Quads": log_imgs}, step=global_step)

    # 5. Log 所有 metric
    wandb.log({f"val/{k}": v for k, v in eval_result.items()}, step=global_step)


def main():

    args = get_args()

    logging_dir = f"{args.output_dir}/{args.logging_dir}"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        filename=f"{args.output_dir}/fontdiffuser_training.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    # Ser training seed
    if args.seed is not None:
        set_seed(args.seed)

    # Load model and noise_scheduler
    unet = build_unet(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    noise_scheduler = build_ddpm_scheduler(args)
    if args.phase_2:
        unet.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/unet.pth"))
        style_encoder.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/style_encoder.pth"))
        content_encoder.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/content_encoder.pth"))

    model = FontDiffuserModel(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder)

    # Build content perceptaual Loss
    perceptual_loss = ContentPerceptualLoss()

    # Load SCR module for supervision
    if args.phase_2:
        scr = build_scr(args=args)
        scr.load_state_dict(torch.load(args.scr_ckpt_path))
        scr.requires_grad_(False)

    # Load the datasets
    content_transforms = transforms.Compose(
        [transforms.Resize(args.content_image_size, 
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    style_transforms = transforms.Compose(
        [transforms.Resize(args.style_image_size, 
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    target_transforms = transforms.Compose(
        [transforms.Resize((args.resolution, args.resolution), 
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    train_font_dataset = FontDataset(
        args=args,
        phase='train', 
        transforms=[
            content_transforms, 
            style_transforms, 
            target_transforms],
        scr=args.phase_2)
    train_dataloader = torch.utils.data.DataLoader(
        train_font_dataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=CollateFN())
    
    # Build optimizer and learning rate
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,)

    # Accelerate preparation
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)
    ## move scr module to the target deivces
    if args.phase_2:
        scr = scr.to(accelerator.device)

    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.experience_name)
        save_args_to_yaml(args=args, output_file=f"{args.output_dir}/{args.experience_name}_config.yaml")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Convert to the training epoch
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    for epoch in range(num_train_epochs):
        train_loss = 0.0
        for step, samples in enumerate(train_dataloader):
            model.train()
            content_images = samples["content_image"]
            style_images = samples["style_image"]
            target_images = samples["target_image"]
            nonorm_target_images = samples["nonorm_target_image"]
            
            with accelerator.accumulate(model):
                # Sample noise that we'll add to the samples
                noise = torch.randn_like(target_images)
                bsz = target_images.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=target_images.device)
                timesteps = timesteps.long()

                # Add noise to the target_images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_target_images = noise_scheduler.add_noise(target_images, noise, timesteps)

                # Classifier-free training strategy
                context_mask = torch.bernoulli(torch.zeros(bsz) + args.drop_prob)
                for i, mask_value in enumerate(context_mask):
                    if mask_value==1:
                        content_images[i, :, :, :] = 1
                        style_images[i, :, :, :] = 1

                # Predict the noise residual and compute loss
                noise_pred, offset_out_sum = model(
                    x_t=noisy_target_images, 
                    timesteps=timesteps, 
                    style_images=style_images,
                    content_images=content_images,
                    content_encoder_downsample_size=args.content_encoder_downsample_size)
                diff_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                offset_loss = offset_out_sum / 2
                
                # output processing for content perceptual loss
                pred_original_sample_norm = x0_from_epsilon(
                    scheduler=noise_scheduler,
                    noise_pred=noise_pred,
                    x_t=noisy_target_images,
                    timesteps=timesteps)
                pred_original_sample = reNormalize_img(pred_original_sample_norm)
                norm_pred_ori = normalize_mean_std(pred_original_sample)
                norm_target_ori = normalize_mean_std(nonorm_target_images)
                percep_loss = perceptual_loss.calculate_loss(
                    generated_images=norm_pred_ori,
                    target_images=norm_target_ori,
                    device=target_images.device)
                
                loss = diff_loss + \
                        args.perceptual_coefficient * percep_loss + \
                            args.offset_coefficient * offset_loss
                
                if args.phase_2:
                    neg_images = samples["neg_images"]
                    # sc loss
                    sample_style_embeddings, pos_style_embeddings, neg_style_embeddings = scr(
                        pred_original_sample_norm, 
                        target_images, 
                        neg_images, 
                        nce_layers=args.nce_layers)
                    sc_loss = scr.calculate_nce_loss(
                        sample_s=sample_style_embeddings,
                        pos_s=pos_style_embeddings,
                        neg_s=neg_style_embeddings)
                    loss += args.sc_coefficient * sc_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    if global_step % args.ckpt_interval == 0:
                        save_dir = f"{args.output_dir}/global_step_{global_step}"
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(model.unet.state_dict(), f"{save_dir}/unet.pth")
                        torch.save(model.style_encoder.state_dict(), f"{save_dir}/style_encoder.pth")
                        torch.save(model.content_encoder.state_dict(), f"{save_dir}/content_encoder.pth")
                        torch.save(model, f"{save_dir}/total_model.pth")
                        logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}] Save the checkpoint on global step {global_step}")
                        print("Save the checkpoint on global step {}".format(global_step))

                        # validation step
                        if accelerator.is_main_process:
                            run_validation(global_step, args, accelerator)

            logs = {
                "step_loss": loss.detach().item(),
                "diff_loss": diff_loss.detach().item(),
                "percep_loss": percep_loss.detach().item(),
                "offset_loss": offset_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            } # MR

            # logs = {
            #     "step_loss": loss.detach().item(),
            #     "diff_loss": diff_loss.detach().item(),
            #     "percep_loss": percep_loss.detach().item(),
            #     "offset_loss": offset_out_sum / 2,
            # } # lack module
            if args.phase_2:
                logs["sc_loss"] = sc_loss.detach().item()

            if global_step % args.log_interval == 0:
                logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}] Global Step {global_step} => train_loss = {loss}")
                if accelerator.is_main_process:
                    accelerator.log(logs, step=global_step)
            progress_bar.set_postfix(**logs)

            if global_step % (args.log_interval * 20) == 0:
                if accelerator.is_main_process:
                    accelerator.log(logs, step=global_step)

                    # 上传图片（前4组 content-style-generated-target）
                    try:
                        num_samples_to_log = min(4, pred_original_sample.shape[0])
                        images_to_log = []
                        for i in range(num_samples_to_log):
                            content_img = reNormalize_img(content_images[i].detach().cpu())
                            style_img = reNormalize_img(style_images[i].detach().cpu())
                            gen_img = pred_original_sample[i].detach().cpu()
                            target_img = reNormalize_img(nonorm_target_images[i].detach().cpu())

                            # 拼接为四拼图：content | style | generated | target
                            img_quad = torch.cat([content_img, style_img, gen_img, target_img], dim=2)
                            images_to_log.append(wandb.Image(img_quad, caption=f"Sample {i}"))

                        wandb.log({"Quad: C | S | Gen | GT": images_to_log}, step=global_step)

                    except Exception as e:
                        print(f"WandB image logging failed at step {global_step}: {e}")
            
            # Quit
            if global_step >= args.max_train_steps:
                break

    accelerator.end_training()

if __name__ == "__main__":
    main()
