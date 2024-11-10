# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

from ..augmentations import CropImageWithMask, RandomResizedCrop, CenterCrop
import matplotlib.pyplot as plt
from PIL import Image
from .transforms import fliplr_joints, crop, generate_target, transform_pixel
import random
import torch.utils.data as data
import pandas as pd
import numpy as np


__all__ = ["MaskFormerBinaryFullDatasetMapper"]


class MaskFormerCOFWMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        modes = None,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        input_size,
        scale_factor,
        rot_factor,
        flip,
        landmark_boundary_cofw,
        boundary,
        landmark_index_300W,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.modes = modes
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        # self.transform = transform
        self.input_size = input_size
        self.output_size = input_size
        # self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.flip = flip
        self.landmark_boundary_COFW = landmark_boundary_cofw
        self.boundary = boundary
        self.landmark_index_COFW = landmark_index_300W
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(
            f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}"
        )

    @classmethod
    def from_config(cls, cfg, is_train=True, modes = None):
        # Build augmentation
        # before augmentation, we have to crop the image around the selected mask with a expand ratio
        augs = [CropImageWithMask(cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO)]
        if is_train:
            # augs.append(RandomResizedCrop(cfg.INPUT.MIN_SIZE_TRAIN))
            # augs.append(T.RandomFlip())

            # Assume always applies to the training set.
            dataset_names = cfg.DATASETS.TRAIN
        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            dataset_names = cfg.DATASETS.TEST
            meta = MetadataCatalog.get(dataset_names[0])
            ignore_label = meta.ignore_label
            augs.append(CenterCrop(min_size, seg_ignore_label=ignore_label))

        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            'modes': modes,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            'input_size':cfg.DATASETS.INPUT_SIZE,
            'scale_factor': cfg.DATASETS.SCALE_FACTOR,
            'rot_factor': cfg.DATASETS.ROT_FACTOR,
            'flip': cfg.DATASETS.FLIP,
            'landmark_boundary_cofw': cfg.DATASETS.LANDMAKR_BOUNDARY_COFW,
            'boundary': cfg.DATASETS.BOUNDARY,
            'landmark_index_300W': cfg.DATASETS.LANDMARK_INDEX_300W,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # assert self.is_train, "MaskFormerSemanticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below wanjun_dataset
        # if dataset_dict["category_id"]>255:
        #     print(dataset_dict["category_id"])
        #     return None

        img = dataset_dict['image']
        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)
            img = np.repeat(img, 3, axis=2)
        # print(dataset_dict["file_name"])
        scale = dataset_dict["scale"]
        center = dataset_dict["center"]
        pts = dataset_dict["landmarks"]
        scale *= 1.25
        nparts = pts.shape[0]

        r = 0
        dataset_dict["meta"] = 'COFW_train'
        if self.is_train:
            scale = scale * (random.uniform(1 - self.scale_factor,
                                            1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor) \
                if random.random() <= 0.6 else 0
            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)
                pts = fliplr_joints(pts, width=img.shape[1], dataset='COFW')
                center[0] = img.shape[1] - center[0]

        img = crop(img, center, scale, self.input_size, dataset_dict["file_name"], rot=r)
        # plt.imshow(img)
        # plt.show()

        # target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()

        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2] + 1, center, scale, self.output_size, rot=r)
                # target[i] = generate_target(target[i], tpts[i]-1, self.sigma,
                #                             label_type=self.label_type)

        # ---------------------- update coord target -------------------------------



        # 边界mask生成
        if self.input_size[0] <= 224:
            radius = 8
        else:
            radius = 16
        sto_maps = []
        num_boundary = len(self.boundary) #取边界数量 8
        instances = Instances(self.input_size)
        if self.is_train and self.modes == 'prompt':
            iidx = 1 #缺少右边界
            while iidx==1:
                iidx = random.randint(0, num_boundary-1)  # 随机一个边界 [0, 7]
            boundary_name = self.boundary[iidx]
            boundary_index = self.landmark_boundary_COFW[iidx]
            bound_land_num = len(boundary_index)
            maps = np.zeros([bound_land_num, self.input_size[0], self.input_size[0]])
            map_index = 0
            for idx in boundary_index:
                x, y = tpts[idx]
                if x + radius < 0 or y + radius < 0 or x - radius > self.input_size[0] or y - radius > self.input_size[0]:
                    continue
                x_low = x - radius if x - radius > 0 else 0
                x_high = x + radius if x + radius < self.input_size[0] else self.input_size[0]
                y_low = y - radius if y - radius > 0 else 0
                y_high = y + radius if y + radius < self.input_size[0] else self.input_size[0]
                maps[map_index][int(y_low):int(y_high), int(x_low):int(x_high)] = 1
                map_index = map_index + 1
            maps = np.max(maps, 0)
            instances.gt_classes = torch.tensor([iidx], dtype=torch.int64)
            masks = []
            masks.append(maps == 1)
            # plt.imshow(masks[0])
            # plt.show()
            masks = BitMasks(torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks]))
            instances.gt_masks = masks.tensor
            # plt.imshow(masks.tensor.squeeze(0).numpy())
            # plt.show()
            instances.gt_boxes = masks.get_bounding_boxes()
        else:
            if not self.is_train:
                dataset_dict["meta"] = 'COFW_val'
            classes = []
            masks = []
            sto_maps = []
            for iidx in range(num_boundary):
                boundary_index = self.landmark_boundary_COFW[iidx]
                bound_land_num = len(boundary_index)
                if bound_land_num == 0: #边界没有点
                    continue
                maps = np.ones([bound_land_num, self.input_size[0], self.input_size[0]]) * (-1) # bound_land_num：每个边界特征点的数量
                map_index = 0
                for idx in boundary_index: #对属于边界的每个特征点进行处理，生成对应的mask
                    x, y = tpts[idx]
                    if x + radius < 0 or y + radius < 0 or x - radius > self.input_size[0] or y - radius > \
                            self.input_size[0]:
                        continue
                    x_low = x - radius if x - radius > 0 else 0
                    x_high = x + radius if x + radius < self.input_size[0] else self.input_size[0]
                    y_low = y - radius if y - radius > 0 else 0
                    y_high = y + radius if y + radius < self.input_size[0] else self.input_size[0]
                    maps[map_index][int(y_low):int(y_high), int(x_low):int(x_high)] = iidx
                    map_index = map_index + 1
                maps = np.max(maps, 0)
                sto_maps.append(torch.Tensor(maps))
                classes.append(iidx)
                masks.append(maps == iidx)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            else:
                masks = BitMasks(torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks]))
                instances.gt_masks = masks.tensor
                instances.gt_boxes = masks.get_bounding_boxes()
            # maps = torch.stack(sto_maps).numpy()
            # maps = np.max(maps, 0)
            # plt.imshow(maps)
            # plt.show()
            # temp = instances.gt_masks.numpy()
            # temp = np.max(temp, 0)
            # plt.imshow(temp)
            # plt.show()
        # plt.imshow(maps)
        # plt.show()
        # plt.imshow(temp)
        # plt.show()

        # temp = np.max(maps, 0)
        # plt.imshow(temp)
        # plt.show()


        target = tpts[:, 0:2] / self.input_size[0]

        img = img.astype(np.float32)
        img = (img / 255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)

        target_weight = np.ones((nparts, 1), dtype=np.float32)
        target_weight = torch.from_numpy(target_weight)

        dataset_dict['image']=torch.Tensor(img)
        dataset_dict['target']=target
        dataset_dict['landmarks']=torch.Tensor(pts)
        dataset_dict['rotate'] = r
        dataset_dict["scale"] = scale
        dataset_dict['target_weight'] = target_weight
        dataset_dict['task'] = 'COFW'
        dataset_dict['instances'] = instances
        return dataset_dict
