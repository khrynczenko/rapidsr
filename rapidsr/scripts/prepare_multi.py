"""
This scripts creates artificial multi frame image datasets out of signle
image datasets.
"""
import os
import cv2
import numpy as np
from scipy import ndimage
from typing import NamedTuple


def align_image(lr: np.ndarray, hr: np.ndarray, alignment_matrix: np.ndarray):
    pass


def get_alignment_matrix(lr: np.ndarray, hr: np.ndarray):
    if [lr.ndim, hr.ndim] != [2, 2]:
        raise ValueError("Both low-resolution and high-resolution images"
                         "should be in grayscale.")
    motion_model = cv2.MOTION_EUCLIDEAN
    warp_matrix = np.identity((2,3), dtype=np.float32)
    iterations =  5000;
    threshold = 1e-10
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, threshold, iterations)
    cv2.findTransformECC(hr, lr, warp_matrix, motion_model, criteria)
    return warp_matrix

def crop(image, x, y):
    cropped_image = image
    if x >= 0:
        cropped_image = cropped_image[y:, :]
    else:
        cropped_image = cropped_image[:y, :]
    if y >= 0:
        cropped_image = cropped_image[:, x:]
    else:
        cropped_image = cropped_image[:, :x]
    return cropped_image

def take_inner_image(image, x, y):
    rows = image.shape[0]
    cols = image.shape[1]
    inner = image[y:rows - y, x: cols - x]
    return inner

def make_image_shape_divisible_by(image, factor: int, reserve: int=0):
    rows = image.shape[0]
    cols = image.shape[1]
    rows_to_remove = (rows - reserve*2)% factor
    cols_to_remove = (cols - reserve*2)% factor
    return image[:rows-rows_to_remove, :cols-cols_to_remove]

class SubPixelShift:
    """
    Perform sub-pixel shift.
    First image is shifted by a given amount of pixels.
    Then inner image (image -  (shift amount) pixels on borders)
    At last image is being down-scaled. By doing so we achieve the artificial sub-pixel shift.

    """

    class Shift(NamedTuple):
        x_direction: int
        y_direction: int

    def __init__(self, shift: Shift, scale: float):
        self._shift = shift
        self._scale = scale

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Shifts an image.

        :param image: Image to shift.
        :return: Shifted image.
        """
        shift = [self._shift.y_direction, self._shift.x_direction]
        if image.ndim == 3:
            # Do not perform any shift on 3 dimension if exists
            shift.append(0) 

        shifted_image = np.roll(image, (self._shift.y_direction, self._shift.x_direction), axis=(0, 1))
        inner = take_inner_image(shifted_image, 1, 1)
        return inner

datasets_path = r"D:\projects\rapidsr\modcrop2"
out_datasets_path = r"D:\projects\rapidsr\out3"
scale = 0.5

Shift = SubPixelShift.Shift
shifts = [Shift(1, 1), Shift(1, 0), Shift(0, 1)]
shift_methods = [SubPixelShift(shift, 0.5) for shift in shifts]
for dataset_entry in os.scandir(datasets_path):
    os.makedirs(os.path.join(out_datasets_path, dataset_entry.name),
                exist_ok=True)
    for i, image_entry in enumerate(os.scandir(dataset_entry.path), start=1):
        scene_path = os.path.join(out_datasets_path, dataset_entry.name,
                                  str(i))
        lrs_path = os.path.join(out_datasets_path, dataset_entry.name, str(i),
                                "lrs")
        os.makedirs(scene_path, exist_ok=True)
        os.makedirs(lrs_path, exist_ok=True)
        image = cv2.imread(image_entry.path, cv2.IMREAD_UNCHANGED)
        image = make_image_shape_divisible_by(image, 2, 1)
        print(image.shape)
        lrs = [method(image) for method in shift_methods]
        lrs = [cv2.resize(lr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4) for lr in lrs]
        hr = take_inner_image(image, 1, 1)
        for lr in lrs:
            print(f"{hr.shape} == {lr.shape}")
            x = hr.shape[0]
            y = int(lr.shape[0] * 2)
            assert hr.shape[0] == lr.shape[0] * 2
            assert hr.shape[1] == lr.shape[1] * 2
        cv2.imwrite(os.path.join(scene_path, "hr.png"), hr)
        for i, lr in enumerate(lrs, start=1):
            cv2.imwrite(os.path.join(lrs_path, f"lr{i}.png"), lr)
