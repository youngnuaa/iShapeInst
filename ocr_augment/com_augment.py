# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:52
# @Author  : zhoujun
import math
import numbers
import random
import cv2
import numpy as np
from skimage.util import random_noise
from albumentations import (
    Blur, CLAHE, GaussNoise, HueSaturationValue, MedianBlur, IAASharpen, RandomContrast,
    RandomBrightness, OneOf, Compose, CoarseDropout, GaussianBlur, Cutout, )


class RandomALBU:
    def __init__(self, random_rate):
        self.aug = self.aug_funcs(p=random_rate)

    def __call__(self, data: dict):
        """
        对图片加噪声
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        im = data['img']
        im_aug = self.aug(image=im)["image"].astype(im.dtype)
        data['img'] = im_aug
        return data

    @staticmethod
    def aug_funcs(p=0.5):
        return Compose([
            GaussNoise(p=0.2),
            OneOf([
                # MotionBlur(p=0.5),
                MedianBlur(blur_limit=3, p=0.2),
                Blur(blur_limit=3, p=0.5),
                GaussianBlur(blur_limit=7, p=0.2),
            ], p=0.5),
            OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(p=0.2),
                # IAAEmboss(),
                RandomBrightness(p=0.2),
                RandomContrast(limit=0.2, p=0.2),
            ], p=0.2),
            HueSaturationValue(p=0.2),
            CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=0.2),
            Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, p=0.2),
            # RandomGridShuffle(grid=(3, 3), always_apply=False, p=0.1),
        ], p=p)


class RandomNoise:
    def __init__(self, random_rate):
        self.random_rate = random_rate

    def __call__(self, data: dict):
        """
        对图片加噪声
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        im = data['img']
        data['img'] = (random_noise(data['img'], mode='gaussian', clip=True) * 255).astype(im.dtype)
        return data


class RandomScale:
    def __init__(self, scales, random_rate):
        """
        :param scales: 尺度
        :param ramdon_rate: 随机系数
        :return:
        """
        self.random_rate = random_rate
        self.scales = scales

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        im = data['img']
        text_polys = data['text_polys']

        tmp_text_polys = text_polys.copy()
        rd_scale = float(np.random.choice(self.scales))
        im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
        tmp_text_polys *= rd_scale

        data['img'] = im
        data['text_polys'] = tmp_text_polys
        return data


class RandomMultiScale:
    def __init__(self, short_sizes, max_size=960, sample_style="range"):
        """
        :param scales: 尺度
        :param ramdon_rate: 随机系数
        :return:
        """
        self.short_sizes = short_sizes
        self.max_size = max_size
        self.sample_style = sample_style

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        #print("self.short_sizes ", self.short_sizes)
        if self.sample_style == "range":
            short_size = np.random.randint(self.short_sizes[0], self.short_sizes[1] + 1)
        else:
            short_size = self.short_sizes

        im = data['img']
        text_polys = data['text_polys']
        tmp_text_polys = text_polys.copy()

        h, w, _ = im.shape
        short_edge = min(h, w)
        max_edge = max(h, w)
        new_max_edge = short_size / short_edge * max_edge
        if new_max_edge > self.max_size:
            scale = self.max_size / max_edge
        else:
            scale = short_size / short_edge

        im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
        #print("short size", short_size)
        #print("short edge:", im.shape)
        #im = cv2.resize(im, (int(w*scale), int(h*scale)))
        tmp_text_polys = tmp_text_polys * scale

        data['img'] = im
        data['text_polys'] = tmp_text_polys
        return data


class RandomRotateImgBox:
    def __init__(self, degrees, random_rate, same_size=False):
        """
        :param degrees: 角度，可以是一个数值或者list
        :param ramdon_rate: 随机系数
        :param same_size: 是否保持和原图一样大
        :return:
        """
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            degrees = (-degrees, degrees)
        elif isinstance(degrees, list) or isinstance(degrees, tuple) or isinstance(degrees, np.ndarray):
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            degrees = degrees
        else:
            raise Exception('degrees must in Number or list or tuple or np.ndarray')
        self.degrees = degrees
        self.same_size = same_size
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        im = data['img']
        text_polys = data['text_polys']

        # ---------------------- 旋转图像 ----------------------
        w = im.shape[1]
        h = im.shape[0]
        angle = np.random.uniform(self.degrees[0], self.degrees[1])

        if self.same_size:
            nw = w
            nh = h
        else:
            # 角度变弧度
            rangle = np.deg2rad(angle)
            # 计算旋转之后图像的w, h
            nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w))
            nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w))
        # 构造仿射矩阵
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, 1)
        # 计算原图中心点到新图中心点的偏移量
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # 更新仿射矩阵
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(im, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        # ---------------------- 矫正bbox坐标 ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rot_text_polys = list()
        for bbox in text_polys:
            point1 = np.dot(rot_mat, np.array([bbox[0, 0], bbox[0, 1], 1]))
            point2 = np.dot(rot_mat, np.array([bbox[1, 0], bbox[1, 1], 1]))
            point3 = np.dot(rot_mat, np.array([bbox[2, 0], bbox[2, 1], 1]))
            point4 = np.dot(rot_mat, np.array([bbox[3, 0], bbox[3, 1], 1]))
            rot_text_polys.append([point1, point2, point3, point4])
        data['img'] = rot_img
        data['text_polys'] = np.array(rot_text_polys)

        # import uuid
        # cv2.drawContours(rot_img, np.array([rot_text_polys], dtype=np.int32), -1, (0, 255, 0), 1)
        # cv2.imwrite(f"tmp/{uuid.uuid4()}.jpg", rot_img)

        return data


class EastRandomRotate:
    def __init__(self, degrees, random_rate, same_size=False):
        """
        :param degrees: 角度，可以是一个数值或者list
        :param ramdon_rate: 随机系数
        :param same_size: 是否保持和原图一样大
        :return:
        """
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            degrees = (-degrees, degrees)
        elif isinstance(degrees, list) or isinstance(degrees, tuple) or isinstance(degrees, np.ndarray):
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            degrees = degrees
        else:
            raise Exception('degrees must in Number or list or tuple or np.ndarray')
        self.degrees = degrees
        self.same_size = same_size
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        im = data['img']
        text_polys = data['text_polys']
        # ---------------------- 旋转图像 ----------------------
        w = im.shape[1]
        h = im.shape[0]

        angle = np.random.uniform(self.degrees[0], self.degrees[1])

        if self.same_size:
            nw = w
            nh = h
        else:
            # 角度变弧度
            rangle = np.deg2rad(angle)
            # 计算旋转之后图像的w, h
            nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w))
            nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w))
        # 构造仿射矩阵
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, 1)
        # 计算原图中心点到新图中心点的偏移量
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # 更新仿射矩阵
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(im, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        # ---------------------- 矫正bbox坐标 ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        text_polys = np.array(text_polys)
        object_num, point_num, _ = text_polys.shape
        text_polys = text_polys.reshape(-1, 2)
        z_ = np.ones(shape=(text_polys.shape[0], 1))
        text_polys_z = np.hstack((text_polys, z_)).T

        text_polys = np.dot(rot_mat, text_polys_z).T.reshape(object_num, point_num, 2)

        data['img'] = rot_img
        data['text_polys'] = text_polys
        return data


class RandomResize:
    def __init__(self, size, random_rate, keep_ratio=False):
        """
        :param input_size: resize尺寸,数字或者list的形式，如果为list形式，就是[w,h]
        :param ramdon_rate: 随机系数
        :param keep_ratio: 是否保持长宽比
        :return:
        """
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError("If input_size is a single number, it must be positive.")
            size = (size, size)
        elif isinstance(size, list) or isinstance(size, tuple) or isinstance(size, np.ndarray):
            if len(size) != 2:
                raise ValueError("If input_size is a sequence, it must be of len 2.")
            size = (size[0], size[1])
        else:
            raise Exception('input_size must in Number or list or tuple or np.ndarray')
        self.size = size
        self.keep_ratio = keep_ratio
        self.random_rate = random_rate


    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        im = data['img']
        text_polys = data['text_polys']

        if self.keep_ratio:
            # 将图片短边pad到和长边一样
            h, w, c = im.shape
            max_h = max(h, self.size[0])
            max_w = max(w, self.size[1])
            im_padded = np.zeros((max_h, max_w, c), dtype=np.uint8)
            im_padded[:h, :w] = im.copy()
            im = im_padded
        text_polys = text_polys.astype(np.float32)
        h, w, _ = im.shape
        im = cv2.resize(im, self.size)
        w_scale = self.size[0] / float(w)
        h_scale = self.size[1] / float(h)
        text_polys[:, :, 0] *= w_scale
        text_polys[:, :, 1] *= h_scale

        data['img'] = im
        data['text_polys'] = text_polys
        return data


def resize_image(img, short_size):
    height, width, _ = img.shape
    if height < width:
        new_height = short_size
        new_width = new_height / height * width
    else:
        new_width = short_size
        new_height = new_width / width * height
    new_height = int(round(new_height / 32) * 32)
    new_width = int(round(new_width / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img, (new_width / width, new_height / height)


class ResizeShortSize:
    def __init__(self, short_size, resize_text_polys=True):
        """
        :param size: resize尺寸,数字或者list的形式，如果为list形式，就是[w,h]
        :return:
        """
        self.short_size = short_size
        self.resize_text_polys = resize_text_polys

    def __call__(self, data: dict) -> dict:
        """
        对图片和文本框进行缩放
        :param data: {'img':,'text_polys':}
        :return:
        """
        im = data['img']
        text_polys = data['text_polys']

        h, w, _ = im.shape
        short_edge = min(h, w)
        if short_edge < self.short_size:
            # 保证短边 >= short_size
            scale = self.short_size / short_edge
            im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
            scale = (scale, scale)
            # im, scale = resize_image(im, self.short_size)
            if self.resize_text_polys:
                # text_polys *= scale
                text_polys[:, 0] *= scale[0]
                text_polys[:, 1] *= scale[1]

        data['img'] = im
        data['text_polys'] = text_polys
        return data


class HorizontalFlip:
    def __init__(self, random_rate):
        """

        :param random_rate: 随机系数
        """
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        im = data['img']
        text_polys = data['text_polys']

        flip_text_polys = text_polys.copy()
        flip_im = cv2.flip(im, 1)
        h, w, _ = flip_im.shape
        flip_text_polys[:, :, 0] = w - flip_text_polys[:, :, 0]

        data['img'] = flip_im
        data['text_polys'] = flip_text_polys
        return data


class VerticallFlip:
    def __init__(self, random_rate):
        """

        :param random_rate: 随机系数
        """
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        im = data['img']
        text_polys = data['text_polys']

        flip_text_polys = text_polys.copy()
        flip_im = cv2.flip(im, 0)
        h, w, _ = flip_im.shape
        flip_text_polys[:, :, 1] = h - flip_text_polys[:, :, 1]
        data['img'] = flip_im
        data['text_polys'] = flip_text_polys
        return data


class MyEastRandomCropData:
    def __init__(self, random_rate=0.5, max_tries=50, min_crop_side_ratio=0.2,
                 require_original_image=False, keep_ratio=True):
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio
        self.require_original_image = require_original_image
        self.keep_ratio = keep_ratio
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data

        im = data['img']
        text_polys = data['text_polys']
        # 计算crop区域
        crop_x, crop_y, crop_w, crop_h = self.crop_area(im, text_polys)

        img = im[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        h, w, _ = img.shape

        text_polys_crop = []
        for poly in text_polys:
            poly[:, 0] = poly[:, 0] - crop_x
            poly[:, 1] = poly[:, 1] - crop_y

            if not self.is_poly_outside_rect(poly, 0, 0, w, h):
                text_polys_crop.append(poly)

        data['img'] = img
        data['text_polys'] = np.float32(text_polys_crop)
        return data

    def is_poly_in_rect(self, poly, x, y, w, h):
        poly = np.array(poly)
        if poly[:, 0].min() < x or poly[:, 0].max() > x + w:
            return False
        if poly[:, 1].min() < y or poly[:, 1].max() > y + h:
            return False
        return True

    def is_poly_outside_rect(self, poly, x, y, w, h):
        poly = np.array(poly)
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False

    def split_regions(self, axis):
        regions = []
        min_axis = 0
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i - 1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        return regions

    def random_select(self, axis, max_size):
        xx = np.random.choice(axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)
        return xmin, xmax

    def region_wise_random_select(self, regions, max_size):
        selected_index = list(np.random.choice(len(regions), 2))
        selected_values = []
        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)
        xmin = min(selected_values)
        xmax = max(selected_values)
        return xmin, xmax

    def crop_area(self, im, text_polys):
        h, w = im.shape[:2]
        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)
        for points in text_polys:
            points = np.round(points, decimals=0).astype(np.int32)
            minx = np.min(points[:, 0])
            maxx = np.max(points[:, 0])
            w_array[minx:maxx] = 1
            miny = np.min(points[:, 1])
            maxy = np.max(points[:, 1])
            h_array[miny:maxy] = 1
        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if len(h_axis) == 0 or len(w_axis) == 0:
            return 0, 0, w, h

        h_regions = self.split_regions(h_axis)
        w_regions = self.split_regions(w_axis)

        for i in range(self.max_tries):
            if len(w_regions) > 1:
                xmin, xmax = self.region_wise_random_select(w_regions, w)
            else:
                xmin, xmax = self.random_select(w_axis, w)

            if len(h_regions) > 1:
                ymin, ymax = self.region_wise_random_select(h_regions, h)
            else:
                ymin, ymax = self.random_select(h_axis, h)

            if xmax - xmin < self.min_crop_side_ratio * w or ymax - ymin < self.min_crop_side_ratio * h:
                # area too small
                continue
            num_poly_in_rect = 0
            for poly in text_polys:
                if not self.is_poly_outside_rect(poly, xmin, ymin, xmax - xmin, ymax - ymin):
                    num_poly_in_rect += 1
                    break

            if num_poly_in_rect > 0:
                return xmin, ymin, xmax - xmin, ymax - ymin

        return 0, 0, w, h