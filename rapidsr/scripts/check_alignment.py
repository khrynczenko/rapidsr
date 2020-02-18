"""
This scripts creates artificial multi frame image datasets out of signle
image datasets.
"""
import os
import cv2
import numpy as np
from scipy import ndimage


def align_image(lr: np.ndarray, hr: np.ndarray, alignment_matrix: np.ndarray):
    pass


def get_alignment_matrix(lr: np.ndarray, hr: np.ndarray):
    if [lr.ndim, hr.ndim] != [2, 2]:
        raise ValueError("Both low-resolution and high-resolution images"
                         "should be in grayscale.")
    motion_model = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2,3, dtype=np.float32)
    iterations = 5000
    threshold = 1e-10
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, iterations, threshold)
    cv2.findTransformECC(hr, lr, warp_matrix, motion_model, criteria, inputMask=None, gaussFiltSize=5)
    return warp_matrix

# lr = cv2.imread(r"D:\projects\rapidsr\out3\BSDS100\1\lrs\lr2.png", cv2.IMREAD_GRAYSCALE)
# lr = cv2.resize(lr, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
# hr = cv2.imread(r"D:\projects\rapidsr\out3\BSDS100\1\hr.png", cv2.IMREAD_GRAYSCALE)
# hr = cv2.resize(hr, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LANCZOS4)
# print(get_alignment_matrix(lr, hr))