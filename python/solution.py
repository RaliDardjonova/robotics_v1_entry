#!/usr/bin/env python3
import numpy as np

from typing import (
    List,
    Tuple,
    Union
)

from utils.image import (
    ImageType,
    PackedImage,
    StrideImage,
)

from utils.function_tracer import FunctionTracer

eye_pattern1 = np.array([[True, True, True, True, True],
                         [True, False, False, False, True],
                         [True, True, True, True, True],
                         [True, False, False, False, True],
                         [True, True, True, True, True]])
eye_pattern2 = np.array([[True, True, True, True, True],
                         [True, False, True, False, True],
                         [True, False, True, False, True],
                         [True, False, True, False, True],
                         [True, True, True, True, True]])
eye_pattern3 = np.array([[True, True, True, True, True],
                         [True, False, True, False, True],
                         [True, True, True, True, True],
                         [True, False, True, False, True],
                         [True, True, True, True, True]])
eye_pattern4 = np.array([[True, True, True, True, True],
                         [True, True, False, True, True],
                         [True, False, True, False, True],
                         [True, True, False, True, True],
                         [True, True, True, True, True]])


eye_patterns = [eye_pattern3, eye_pattern2, eye_pattern1, eye_pattern4]


def reduce_red(image_slice: np.array):
    masked_common = image_slice[common_pattern]
    if all(i >= 200 for i in masked_common):
        for eye_pattern in eye_patterns:
            masked = image_slice[eye_pattern]
            if all(i >= 200 for i in masked):
                image_slice[eye_pattern] -= 150
                break

    return image_slice


common_pattern = np.array([[True, True, True, True, True],
                           [True, False, False, False, True],
                           [True, False, True, False, True],
                           [True, False, False, False, True],
                           [True, True, True, True, True]])


def compute_solution(images: List[Union[PackedImage, StrideImage]]) -> List[Union[PackedImage, StrideImage]]:
    ft = FunctionTracer("compute_solution", "seconds")

    for idx, image in enumerate(images):

        # take only the red variable of a pixel and convert it to a matrix
        image_matrix = np.array(image.pixels_red)
        shape = (image.resolution.height, image.resolution.width)
        image_matrix = image_matrix.reshape(shape)

        # iterate over all pixels in an image
        for i in range(image.resolution.height - 4):
            for j in range(image.resolution.width - 4):
                if image_matrix[i, j] >= 200:
                    # check if the 5x5 part of the matrix is a red eye and reduce the red component
                    image_matrix[i:i + 5, j: j + 5] = reduce_red(image_matrix[i:i + 5, j: j + 5])

        image_matrix = image_matrix.reshape(len(image.pixels_red))

        image.pixels_red = list(image_matrix)

    del ft
    return images
