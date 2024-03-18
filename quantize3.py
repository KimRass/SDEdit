import timeit
from collections import Counter
from typing import Callable, Tuple, Optional, Union, List, Type

from PIL import Image
import cv2
import numpy as np

MAX_SIZE = 500


def get_img_data(
    img_input: Image.Image,
    mini: bool = False,
    conversion_method: int = cv2.COLOR_RGB2BGR,
) -> Tuple[np.ndarray, int, np.ndarray]:
    img: np.ndarray = cv2.cvtColor(np.array(img_input), conversion_method)
    ratio: float = min(
        MAX_SIZE / img.shape[0], MAX_SIZE / img.shape[1]
    )  # calculate ratio
    if mini:
        ratio /= 6
    # img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)

    n_pixels: int = img.size

    flat_img: np.ndarray = img.reshape((-1, 3))
    flat_img: np.ndarray = np.float32(flat_img)
    return img, n_pixels, flat_img


def update_n_colors(
    label: np.ndarray,
    n_pixels: int,
    threshold_pixel_percentage: float,
    n_colors: int,  # , flat_img: np.ndarray
) -> Tuple[int, int]:
    n_colors_under_threshold: int = 0
    label = label.flatten()
    color_count: Counter[int] = Counter(label)
    for (pixel, count) in color_count.items():
        if count / n_pixels < threshold_pixel_percentage:
            n_colors_under_threshold += 1
    # silhouette = sklearn.metrics.silhouette_score(flat_img, label, metric='euclidean', sample_size=1000)
    # print(f'n_colors = {n_colors}, silhouette_score = {silhouette}')
    n_colors -= -(-n_colors_under_threshold // 2)  # ceil integer division
    return n_colors, n_colors_under_threshold


def process_result(
    center: np.ndarray,
    label: np.ndarray,
    shape: Tuple[int, int, int],
    conversion_method: int = cv2.COLOR_BGR2RGB,
) -> Tuple[Type[Image.Image], np.ndarray]:
    center: np.ndarray = np.uint8(center)
    quantized_img: np.ndarray = center[label]
    quantized_img = quantized_img.reshape(shape)
    quantized_img = cv2.cvtColor(quantized_img, conversion_method)
    center = cv2.cvtColor(np.expand_dims(center, axis=0), conversion_method)[0]
    # return Image.fromarray(quantized_img), center
    return Image.fromarray(quantized_img)


def test_opencv(
    img_input: Image.Image, method1: int, method2: int, n_colors: int = 4
) -> Tuple[Type[Image.Image], np.ndarray]:
    img, n_pixels, flat_img = get_img_data(img_input, False, method1)

    threshold_pixel_percentage: float = 0.01
    n_colors_under_threshold: int = n_colors
    criteria: Tuple[int, int, float] = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        10,
        1.0,
    )
    center: Optional[np.ndarray] = None
    label: Optional[np.ndarray] = None

    while n_colors_under_threshold > 0:
        ret: float
        ret, label, center = cv2.kmeans(
            flat_img, n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        n_colors, n_colors_under_threshold = update_n_colors(
            label, n_pixels, threshold_pixel_percentage, n_colors  # , flat_img
        )

    return process_result(center, label, img.shape, method2)


def test_opencv_rgb(img_input: Image.Image) -> Tuple[Type[Image.Image], np.ndarray]:
    return test_opencv(img_input, cv2.COLOR_RGB2BGR, cv2.COLOR_BGR2RGB)


def test_opencv_hsv(img_input: Image.Image) -> Tuple[Type[Image.Image], np.ndarray]:
    return test_opencv(img_input, cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2RGB)


def test_opencv_lab(img_input: Image.Image) -> Tuple[Type[Image.Image], np.ndarray]:
    return test_opencv(img_input, cv2.COLOR_RGB2Lab, cv2.COLOR_Lab2RGB)


def create_images_results(imgs: [str], quantize_functions: [str]) -> None:
    for img_name in imgs:
        for quantize_function_name in quantize_functions:
            func_name = "test_" + quantize_function_name
            quantized_img, colors_list = eval(f"{func_name}(_img)")
            print(
                f"{img_name} - {quantize_function_name} nb colors : {len(colors_list)}"
            )
            quantized_img.save(
                f"./imgs_results/test_img_{img_name}_{quantize_function_name}.png"
            )


if __name__ == "__main__":
    quantize_functions = [
        # "pillow_median_cut",
        # "pillow_maximum_coverage",
        # "pillow_fast_octree",
        "opencv_rgb",
        "opencv_hsv",
        "opencv_lab",
        # "scipy",
        # "scipy2",
        # "sklearn_kmeans",
        # "sklearn_kmeans2",
        # "sklearn_mini_batch_kmeans",
        # "sklearn_mean_shift",
        # "pycluster_bsas",
        # "pycluster_mbsas",
        # # "pycluster_dbscan",
        # # "pycluster_optics",
        # "pycluster_syncnet",
        # "pycluster_syncsom",
        # "pycluster_ttsas",
        # "pycluster_xmeans",
        # "pycluster_kmeans",
        # "pycluster_kmedians",
    ]
    create_images_results(imgs=[1], quantize_functions=quantize_functions)
