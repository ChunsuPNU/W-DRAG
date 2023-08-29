import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import random

def original(img):
    x = [img]
    return x


def rotation(img, degs):
    x_rot = []

    for deg in degs:
        if deg > 0:
            if deg == 90:
                angle = random.randint(1, 90)
            elif deg == 180:
                angle = random.randint(91, 180)
            elif deg == 270:
                angle = random.randint(181, 270)
            elif deg == 360:
                angle = random.randint(271, 360)

            rotated_img = transforms.functional.rotate(img=img, angle=angle,
                                                              interpolation=transforms.InterpolationMode.NEAREST, fill=-1)

            x_rot.append(rotated_img)
        else:
            x_rot.append(img)

    return x_rot

def flip(img):
    x_flip = []

    i = random.randint(0, 4)


    if i == 0:
        x_flip.append(img.flip(2))
    elif i == 1:
        x_flip.append(img.flip(3))
    elif i == 2:
        x_flip.append(torch.rot90(img.flip(2), dims=[2, 3]))
    elif i == 3:
        x_flip.append(torch.rot90(img.flip(2), k=-1, dims=[2, 3]))
    elif i == 4:
        x_flip.append(img.flip(2).flip(3))


    return x_flip


def cropping(img):
    x_crop = []

    b, c, h, w = img.shape
    rescale_weight = random.uniform(1.2, 1.5)
    resized_h, resized_w = int(h * rescale_weight), int(w * rescale_weight)

    resizing = transforms.Resize((resized_h, resized_w))
    img = resizing(img)

    boxes = [[0, 0, h, w],
             [0, 0, h, w],
             [0, resized_w - w, h, resized_w],
             [resized_h - h, 0, resized_h, w],
             [resized_h - h, resized_w - w, resized_h, resized_w]]

    i = random.randint(1, np.shape(boxes)[0]-1)

    cropped_img = img[:, :, int(boxes[i][0]):int(boxes[i][2]), int(boxes[i][1]):int(boxes[i][3])].clone()

    x_crop.append(cropped_img)

    # x_crop_img.append(F.interpolate(cropped_img, (h, w), mode='bicubic'))
    # x_crop_fake.append(F.interpolate(cropped_fake, (h, w), mode='bicubic'))

    #x_crop = torch.cat(x_crop,0)
    return x_crop


def scaling(img):
    x_scale = []

    scaling_factor = random.uniform(0.8, 1.2)

    x_scale.append(transforms.functional.affine(img, angle=0, translate=[0, 0], scale=scaling_factor, shear=0, fill=-1))

    return x_scale


def translation(img):
    x_translated = []

    i_lr = random.randint(-5, 5)
    i_ud = random.randint(-5, 5)

    translated_img_img = transforms.functional.affine(img, angle=0, translate=[i_lr, i_ud], scale=1.0, shear=0, fill=-1)
    x_translated.append(translated_img_img)

    return x_translated


def augmenting_data(img, aug, aug_list):
    if aug == 'rotation':
        return rotation(img, aug_list)
    elif aug == 'fliprot':
        return flip(img)
    elif aug == 'cropping':
        return cropping(img)
    elif aug == 'scaling':
        return scaling(img)
    elif aug == 'translation':
        return translation(img)
    elif aug == 'original':
        return original(img)

