
import os

import SimpleITK as sitk
import numpy as np
from monai.utils.type_conversion import convert_data_type


def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    result = result.transpose(2, 3).transpose(1, 2)
    return result


def padding_id(strings, length=10):
    padded_strings = []
    for s in strings:
        if len(s) < length:
            s = '0' * (length - len(s)) + s
        padded_strings.append(s)
    return padded_strings

def save_as_nii(array, origin, direction, spacing, save_name=None):
    Img = sitk.GetImageFromArray(array)
    Img.SetOrigin(origin)
    Img.SetDirection(direction)
    Img.SetSpacing(spacing)
    sitk.WriteImage(Img, save_name)


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_information(path):
    image = sitk.ReadImage(path)
    return image.GetOrigin(), image.GetDirection(), image.GetSpacing()

def check_file(path):
    if not os.path.exists(path):
        return True
    else:
        return False
