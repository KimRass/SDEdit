# 1. Pretrained Models
- Download 'ddpm_celeba_64×64.pth' from [DDPM](https://github.com/KimRass/DDPM)

# 2. Samples
## 1) From Simulated Stroke
| `mode="from_sim_stroke"`,<br>`inter_time=0.30`, `ref_idx=132` | `mode="from_sim_stroke"`,<br>`inter_time=0.30`, `ref_idx=135` |
|:-:|:-:|
| <img src="https://github.com/KimRass/SDEdit/assets/67457712/a0eea975-0928-4351-b058-00e58288207e" width="350"> | <img src="https://github.com/KimRass/SDEdit/assets/67457712/61444428-eeb7-490b-8458-a0d6569ce15d" width="350"> |

| `mode="from_sim_stroke"`,<br>`inter_time=0.45`, `ref_idx=132` | `mode="from_sim_stroke"`,<br>`inter_time=0.45`, `ref_idx=135` |
|:-:|:-:|
| <img src="https://github.com/KimRass/SDEdit/assets/67457712/39481860-2bdc-4876-9d59-1bb49a20f525" width="350"> | <img src="https://github.com/KimRass/SDEdit/assets/67457712/0bac0db2-a240-4cd5-a4b8-903461059777" width="350"> |

| `mode="from_sim_stroke"`,<br>`inter_time=0.60`, `ref_idx=132` | `mode="from_sim_stroke"`,<br>`inter_time=0.60`, `ref_idx=135` |
|:-:|:-:|
| <img src="https://github.com/KimRass/SDEdit/assets/67457712/d9642e3f-39c4-4039-9f34-cd4aaebaf640" width="350"> | <img src="https://github.com/KimRass/SDEdit/assets/67457712/329d76fe-4d1c-41b4-9906-4316652e495d" width="350"> |

# 3. Theoretical Backgrounds
$$\mathbf{x}(t) = \alpha(t)\mathbf{x}(0) + \sigma(t)\mathbf{z}, \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

# 4. To-Dos
- [x] Stroke input simulation.
- [x] Sampling from stroke.
- [ ] Total repeats.
- [ ] VE SDEdit.
- [ ] Sampling from scribble.
- [ ] Image editing only on masked regions.
