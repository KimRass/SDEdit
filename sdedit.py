# References:
    # https://nn.labml.ai/diffusion/ddpm/index.html
    # https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/08_diffusion/01_ddm/ddm.ipynb
    # https://huggingface.co/blog/annotated-diffusion

import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import numpy as np

from celeba import CelebADS
from utils import image_to_grid
from stroke import StrokeSimulator


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

        self.signal_rate = self.alpha_bar ** 0.5 # "$\alpha(t)$"
        self.noise_rate = (1 - self.alpha ** 2) ** 0.5 # "$\sigma(t)$"

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
            size=(
                batch_size, self.image_channels, self.img_size, self.img_size,
            ),
            device=self.device,
        )

    def sample_diffusion_step(self, batch_size):
        return torch.randint(
            0, self.n_diffusion_steps, size=(batch_size,), device=self.device,
        )

    def batchify_diffusion_steps(self, step, batch_size):
        return torch.full(
            size=(batch_size,),
            fill_value=step,
            dtype=torch.long,
            device=self.device,
        )

    def perform_diffusion_process(
        self, ori_image, diffusion_step, rand_noise=None,
    ):
        """
        $\mathbf{x}(t)
        = \alpha(t)\mathbf{x}(0) + \sigma(t)\mathbf{z},
        \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        """
        signal_rate_t = self.index(
            self.signal_rate, diffusion_step=diffusion_step,
        )
        noise_rate_t = self.index(self.noise_rate, diffusion_step=diffusion_step)
        if rand_noise is None:
            rand_noise = self.sample_noise(batch_size=ori_image.size(0))
        noisy_image = signal_rate_t * ori_image + noise_rate_t * rand_noise
        return noisy_image

    def forward(self, noisy_image, diffusion_step):
        return self.model(
            noisy_image=noisy_image.to(self.device), diffusion_step=diffusion_step,
        )

    @torch.inference_mode()
    def take_denoising_step(self, noisy_image, step):
        diffusion_step = self.batchify_diffusion_steps(
            step=step, batch_size=noisy_image.size(0),
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

        if step > 0:
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

    def perform_denoising_process(self, noisy_image, start_step, n_frames=None):
        if n_frames is not None:
            frames = list()

        x = noisy_image
        pbar = tqdm(range(start_step, -1, -1), leave=False)
        for step in pbar:
            pbar.set_description("Denoising...")

            x = self.take_denoising_step(x, step=step)

            if n_frames is not None and (
                step % (self.n_diffusion_steps // n_frames) == 0
            ):
                frames.append(self._get_frame(x))
        return frames if n_frames is not None else x


class SDEdit(DDPM):
    def __init__(self, model, data_dir, img_size, device, kernel_size=3):
        super().__init__(model=model, img_size=img_size, device=device)

        self.stroke_sim = StrokeSimulator(kernel_size=kernel_size)

        self.ds = CelebADS(
            data_dir=data_dir, split="test", img_size=self.img_size, hflip=False,
        )

    def select_and_batchify_ref(self, ref_idx, batch_size):
        # if dataset == "celeba":
        return self.ds[ref_idx][None, ...].to(self.device).repeat(batch_size, 1, 1, 1)

    def time_to_step(self, time):
        return int(time * self.n_diffusion_steps)

    def sample_from_stroke(self, ref_idx, interm_time, n_colors, batch_size):
        ref = self.select_and_batchify_ref(
            ref_idx=ref_idx, batch_size=batch_size - 2,
        )
        stroke = self.stroke_sim(ref, n_colors=n_colors).to(self.device)

        interm_step = self.time_to_step(interm_time)
        diffusion_step = self.batchify_diffusion_steps(
            step=interm_step - 1, batch_size=batch_size - 2,
        )
        noisy_stroke = self.perform_diffusion_process(
            ori_image=stroke,
            diffusion_step=diffusion_step,
        )

        denoised_stroke = self.perform_denoising_process(
            noisy_image=noisy_stroke,
            start_step=interm_step - 1,
        )
        return torch.cat([ref[: 1, ...], stroke[: 1, ...], denoised_stroke], dim=0)
