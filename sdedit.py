# References:
    # https://nn.labml.ai/diffusion/ddpm/index.html
    # https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/08_diffusion/01_ddm/ddm.ipynb
    # https://huggingface.co/blog/annotated-diffusion

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import imageio
from tqdm import tqdm
import contextlib

from celeba import CelebADS
from utils import image_to_grid
from stroke import simulate_user_stroke_input

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

transform = A.Compose(
    [
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2(),
    ]
)


class DDPM(nn.Module):
    def linearly_schedule_beta(self):
        self.beta = torch.linspace(
            self.init_beta,
            self.fin_beta,
            self.n_diffusion_steps,
            device=self.device,
        )
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def __init__(
        self,
        model,
        img_size,
        device,
        image_channels=3,
        n_diffusion_steps=1000,
        init_beta=0.0001,
        fin_beta=0.02,
    ):
        super().__init__()

        self.img_size = img_size
        self.device = device
        self.image_channels = image_channels
        self.n_diffusion_steps = n_diffusion_steps
        self.init_beta = init_beta
        self.fin_beta = fin_beta

        self.model = model.to(device)

        self.linearly_schedule_beta()

    @staticmethod
    def index(x, diffusion_step):
        return x[diffusion_step][:, None, None, None]

    def sample_noise(self, batch_size):
        return torch.randn(
            size=(batch_size, self.image_channels, self.img_size, self.img_size),
            device=self.device,
        )

    def sample_diffusion_step(self, batch_size):
        return torch.randint(
            0, self.n_diffusion_steps, size=(batch_size,), device=self.device,
        )

    def batchify_diffusion_steps(self, diffusion_step_idx, batch_size):
        return torch.full(
            size=(batch_size,),
            fill_value=diffusion_step_idx,
            dtype=torch.long,
            device=self.device,
        )

    def perform_diffusion_process(self, ori_image, diffusion_step, rand_noise=None):
        alpha_bar_t = self.index(self.alpha_bar, diffusion_step=diffusion_step)
        mean = (alpha_bar_t ** 0.5) * ori_image
        var = 1 - alpha_bar_t
        if rand_noise is None:
            rand_noise = self.sample_noise(batch_size=ori_image.size(0))
        noisy_image = mean + (var ** 0.5) * rand_noise
        return noisy_image

    def forward(self, noisy_image, diffusion_step):
        return self.model(
            noisy_image=noisy_image.to(self.device), diffusion_step=diffusion_step,
        )

    @torch.inference_mode()
    def take_denoising_step(self, noisy_image, diffusion_step_idx):
        diffusion_step = self.batchify_diffusion_steps(
            diffusion_step_idx=diffusion_step_idx, batch_size=noisy_image.size(0),
        )
        alpha_t = self.index(self.alpha, diffusion_step=diffusion_step)
        beta_t = self.index(self.beta, diffusion_step=diffusion_step)
        alpha_bar_t = self.index(self.alpha_bar, diffusion_step=diffusion_step)
        pred_noise = self(
            noisy_image=noisy_image.detach(), diffusion_step=diffusion_step,
        )
        model_mean = (1 / (alpha_t ** 0.5)) * (
            noisy_image - ((beta_t / ((1 - alpha_bar_t) ** 0.5)) * pred_noise)
        )
        model_var = beta_t

        if diffusion_step_idx > 0:
            rand_noise = self.sample_noise(batch_size=noisy_image.size(0))
        else:
            rand_noise = torch.zeros(
                size=(noisy_image.size(0), self.image_channels, self.img_size, self.img_size),
                device=self.device,
            )
        return model_mean + (model_var ** 0.5) * rand_noise

    @staticmethod
    def _get_frame(x):
        grid = image_to_grid(x, n_cols=int(x.size(0) ** 0.5))
        frame = np.array(grid)
        return frame

    def perform_denoising_process(self, noisy_image, start_diffusion_step_idx, n_frames=None):
        if n_frames is not None:
            frames = list()

        x = noisy_image
        pbar = tqdm(range(start_diffusion_step_idx, -1, -1), leave=False)
        for diffusion_step_idx in pbar:
            pbar.set_description("Denoising...")

            x = self.take_denoising_step(x, diffusion_step_idx=diffusion_step_idx)

            if n_frames is not None and (
                diffusion_step_idx % (self.n_diffusion_steps // n_frames) == 0
            ):
                frames.append(self._get_frame(x))
        return frames if n_frames is not None else x


class SDEdit(DDPM):
    def select_and_batchify_ref(self, data_dir, ref_idx, batch_size):
        # if dataset == "celeba":
        ds = CelebADS(
            data_dir=data_dir, split="test", img_size=self.img_size, hflip=False,
        )
        # return ds[ref_idx][None, ...].to(self.device).repeat(batch_size, 1, 1, 1)
        return ds[ref_idx].to(self.device)

    def sample_from_stroke(self, data_dir, ref_idx, diffusion_step_idx, n_colors, batch_size=1):
        ori_image = self.select_and_batchify_ref(
            data_dir=data_dir,
            ref_idx=ref_idx,
            batch_size=batch_size,
        )
        stroke = simulate_user_stroke_input(ori_image, n_colors=n_colors)
        stroke = transform(image=np.array(stroke))["image"][None, ...].to(self.device)
        # print(ori_image.shape, stroke.shape, stroke.device)

        diffusion_step = self.batchify_diffusion_steps(
            diffusion_step_idx=diffusion_step_idx - 1, batch_size=1,
        )
        noisy_stroke = self.perform_diffusion_process(
            ori_image=stroke,
            diffusion_step=diffusion_step,
        )

        denoised_image =  self.perform_denoising_process(
            noisy_image=noisy_stroke,
            start_diffusion_step_idx=diffusion_step_idx - 1,
        )
        return torch.cat([ori_image[None, ...], stroke[: 1, ...], denoised_image], dim=0)
