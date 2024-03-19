import torch
import argparse
from pathlib import Path
import re
import gc

from utils import get_device, image_to_grid, save_image
from unet import UNet
from sdedit import SDEdit


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_params", type=str, required=True)
    parser.add_argument("--img_size", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ref_idx", type=int, required=True)
    parser.add_argument("--diffusion_step_idx", type=int, required=True)
    parser.add_argument("--n_colors", type=int, required=True)

    # For single_ref, denoising_process modes only
    parser.add_argument("--batch_size", type=int, required=False)

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["from_stroke"],
    )

    args = parser.parse_args()

    args_dict = vars(args)
    new_args_dict = dict()
    for k, v in args_dict.items():
        new_args_dict[k.upper()] = v
    args = argparse.Namespace(**new_args_dict)
    return args


def get_sample_num(x, pref):
    match = re.search(pattern=rf"{pref}-\s*(.+)", string=x)
    return int(match.group(1)) if match else -1


def get_max_sample_num(samples_dir, pref):
    stems = [path.stem for path in Path(samples_dir).glob("**/*") if path.is_file()]
    if stems:
        return max([get_sample_num(stem, pref=pref) for stem in stems])
    else:
        return -1


def pref_to_save_path(samples_dir, pref, suffix):
    max_sample_num = get_max_sample_num(samples_dir, pref=pref)
    save_stem = f"{pref}-{max_sample_num + 1}"
    return str((Path(samples_dir)/save_stem).with_suffix(suffix))


def get_save_path(samples_dir, mode, dataset, ref_idx, diffusion_step_idx):
    pref = f"mode={mode}/dataset={dataset}/ref_idx={ref_idx}"
    if mode in ["from_stroke"] and diffusion_step_idx != 0:
        pref += f"-diffusion_step_idx={diffusion_step_idx}"
    return pref_to_save_path(samples_dir=samples_dir, pref=pref, suffix=".jpg")


def main():
    torch.set_printoptions(linewidth=70)

    args = get_args()
    DEVICE = get_device()
    print(f"[ DEVICE: {DEVICE} ]")

    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    SAMPLES_DIR = Path(__file__).resolve().parent/"samples"

    net = UNet()
    model = SDEdit(model=net, img_size=args.IMG_SIZE, device=DEVICE)
    state_dict = torch.load(str(args.MODEL_PARAMS), map_location=DEVICE)
    model.load_state_dict(state_dict)

    save_path = get_save_path(
        samples_dir=SAMPLES_DIR,
        mode=args.MODE,
        dataset=args.DATASET,
        ref_idx=args.REF_IDX,
        diffusion_step_idx=args.DIFFUSION_STEP_IDX,
    )
    if args.MODE == "from_stroke":
        gen_image = model.sample_from_stroke(
            data_dir=args.DATA_DIR,
            ref_idx=args.REF_IDX,
            diffusion_step_idx=args.DIFFUSION_STEP_IDX,
            n_colors=args.N_COLORS,
        )
        gen_grid = image_to_grid(gen_image, n_cols=3)
        # gen_grid.show()
        save_image(gen_grid, save_path=save_path)


if __name__ == "__main__":
    main()
