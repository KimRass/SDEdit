# 1. Pretrained Models
- Download 'ddpm_celeba_64×64.pth' from [DDPM](https://github.com/KimRass/DDPM)

# 2. Samples
| `inter_time=0.15`, `ref_idx=132` | `inter_time=0.15`, `ref_idx=135` |
|:-:|:-:|
| <img src="https://github.com/KimRass/SDEdit/assets/67457712/a0eea975-0928-4351-b058-00e58288207e" width="350"> | <img src="https://github.com/KimRass/SDEdit/assets/67457712/61444428-eeb7-490b-8458-a0d6569ce15d" width="350"> |

| `inter_time=0.60`, `ref_idx=132` | `inter_time=0.60`, `ref_idx=135` |
|:-:|:-:|
| <img src="https://github.com/KimRass/SDEdit/assets/67457712/d9642e3f-39c4-4039-9f34-cd4aaebaf640" width="350"> | <img src="" width="350"> |

# 3. Theoretical Backgrounds
$$\mathbf{x}(t) = \alpha(t)\mathbf{x}(0) + \sigma(t)\mathbf{z}, \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$