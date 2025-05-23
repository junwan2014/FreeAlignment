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



__all__ = ["MaskFormerBinaryFullDatasetMapper"]


class MaskFormerBinaryFullDatasetMapper:
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
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
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
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(
            f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}"
        )

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        # before augmentation, we have to crop the image around the selected mask with a expand ratio
        augs = [CropImageWithMask(cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO)]
        if is_train:
            augs.append(RandomResizedCrop(cfg.INPUT.MIN_SIZE_TRAIN))
            augs.append(T.RandomFlip())

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
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
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
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError( "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format( dataset_dict["file_name"]))

        if "annotations" in dataset_dict:
            raise ValueError( "Semantic segmentation dataset should not have 'annotations'.")
            

        if dataset_dict["task"] == "sem_seg":
            aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
            aug_input.category_id = dataset_dict["category_id"]
            aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
            image = aug_input.image
            # plt.imshow(image)
            # plt.show()
            sem_seg_gt = aug_input.sem_seg
            # plt.imshow(sem_seg_gt)
            # plt.show()

            
            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            if sem_seg_gt is not None:
                sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

            if self.size_divisibility > 0:
                image_size = (image.shape[-2], image.shape[-1])
                padding_size = [ 0, self.size_divisibility - image_size[1], 0, self.size_divisibility - image_size[0],]
                image = F.pad(image, padding_size, value=128).contiguous()
                if sem_seg_gt is not None:
                    sem_seg_gt = F.pad( sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

            image_shape = (image.shape[-2], image.shape[-1])  # h, w

            dataset_dict["image"] = image

            if sem_seg_gt is not None:
                dataset_dict["sem_seg"] = sem_seg_gt.long()

            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            instances.gt_classes = torch.tensor( [dataset_dict["category_id"]], dtype=torch.int64)

            masks = []
            masks.append(sem_seg_gt == dataset_dict["category_id"])
            # tmp = masks[0].astype(int)
            # plt.imshow(tmp)
            # plt.show()
            if masks[0].sum() == 0:
                return None
            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros( (0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks( torch.stack([ torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks]))
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances
        

        elif dataset_dict["task"] == "pan_seg":

            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB").astype("double")

            from panopticapi.utils import rgb2id
            
            pan_seg_gt = rgb2id(pan_seg_gt)

            aug_input = T.AugInput(image, sem_seg=pan_seg_gt)
            aug_input.category_id = dataset_dict["id"]
            aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
            image = aug_input.image
            pan_seg_gt = aug_input.sem_seg

            # plt.imshow(image)
            # plt.show()
            #
            # plt.imshow(sem_seg_gt)
            # plt.show()

            
            pan_seg_gt = torch.as_tensor(pan_seg_gt.astype("long"))
            
            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

            if self.size_divisibility > 0:
                image_size = (image.shape[-2], image.shape[-1])
                padding_size = [ 0, self.size_divisibility - image_size[1], 0, self.size_divisibility - image_size[0],]
                image = F.pad(image, padding_size, value=128).contiguous()
                if pan_seg_gt is not None:
                    pan_seg_gt = F.pad( pan_seg_gt, padding_size, value=self.ignore_label).contiguous()

            image_shape = (image.shape[-2], image.shape[-1])  # h, w

            dataset_dict["image"] = image

            pan_seg_gt = pan_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = []
            masks = []
            
            class_id = dataset_dict["category_id"]
            if not dataset_dict["iscrowd"]:
                classes.append(class_id)
                masks.append(pan_seg_gt == dataset_dict["id"])
                # tmp = masks[0].astype(int)
                # plt.imshow(tmp)
                # plt.show()
            
            # classes = np.array(classes)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            else:
                masks = BitMasks(torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks]))
                instances.gt_masks = masks.tensor
                instances.gt_boxes = masks.get_bounding_boxes()
            
            dataset_dict["instances"] = instances

        
        # ### for instance segmentation
        elif dataset_dict["task"] == "ins_seg":

            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB").astype("double")

            # plt.imshow(image)
            # plt.show()

            from panopticapi.utils import rgb2id
            
            pan_seg_gt = rgb2id(pan_seg_gt)

            aug_input = T.AugInput(image, sem_seg=pan_seg_gt)
            aug_input.category_id = dataset_dict["id"]
            aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
            image = aug_input.image
            pan_seg_gt = aug_input.sem_seg
            #
            # plt.imshow(sem_seg_gt)
            # plt.show()

            pan_seg_gt = torch.as_tensor(pan_seg_gt.astype("long"))

            
            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

            if self.size_divisibility > 0:
                image_size = (image.shape[-2], image.shape[-1])
                padding_size = [ 0, self.size_divisibility - image_size[1], 0, self.size_divisibility - image_size[0],]
                image = F.pad(image, padding_size, value=128).contiguous()
                if pan_seg_gt is not None:
                    pan_seg_gt = F.pad( pan_seg_gt, padding_size, value=self.ignore_label).contiguous()

            image_shape = (image.shape[-2], image.shape[-1])  # h, w

            dataset_dict["image"] = image

            pan_seg_gt = pan_seg_gt.numpy()
            ins_instances = Instances(image_shape)
            classes = []
            masks = []

            class_id = dataset_dict["category_id"]
            if not dataset_dict["iscrowd"] and dataset_dict["isthing"]:
                classes.append(class_id)
                masks.append(pan_seg_gt == dataset_dict["id"])
                # tmp = masks[0].astype(int)
                # plt.imshow(tmp)
                # plt.show()
            
            # classes = np.array(classes)
            ins_instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                ins_instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
                ins_instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            else:
                masks = BitMasks(torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks]))
                ins_instances.gt_masks = masks.tensor
                ins_instances.gt_boxes = masks.get_bounding_boxes()
            
            dataset_dict["instances"] = ins_instances

        return dataset_dict
