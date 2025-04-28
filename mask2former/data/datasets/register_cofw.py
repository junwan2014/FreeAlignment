# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from .utils import load_binary_mask
from detectron2.data.datasets.coco import load_coco_json, load_sem_seg_cofw

CLASS_NAMES = ('left contour', 'right contour', 'left brow', 'right brow', 'nose', 'left eye', 'right eye',  'mouth')
CLASS_NAMES_VAL = ('left contour', 'right contour', 'left brow', 'right brow', 'nose', 'left eye', 'right eye',  'mouth')
# CLASS_NAMES_VAL = ('left contour',  'left brow', 'right brow', 'nose', 'left eye', 'right eye',  'mouth')

BASE_CLASS_NAMES = [
    c for i, c in enumerate(CLASS_NAMES) if i not in [15, 16, 17, 18, 19]
]
NOVEL_CLASS_NAMES = [c for i, c in enumerate(CLASS_NAMES) if i in [15, 16, 17, 18, 19]]


def _get_voc_meta(cat_list):
    ret = {
        "thing_classes": cat_list,
        "stuff_classes": cat_list,
    }
    return ret


def register_all_voc_11k(root):
    root = os.path.join(root, "VOC2012")
    meta = _get_voc_meta(CLASS_NAMES)
    base_meta = _get_voc_meta(BASE_CLASS_NAMES)

    novel_meta = _get_voc_meta(NOVEL_CLASS_NAMES)

    for name, image_dirname, sem_seg_dirname in [
        ("train", "JPEGImages", "annotations_detectron2/train"),
        ("test", "JPEGImages", "annotations_detectron2/val"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        all_name = f"voc_sem_seg_{name}"
        DatasetCatalog.register(
            all_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(all_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )
        MetadataCatalog.get(all_name).set(
            evaluation_set={
                "base": [
                    meta["stuff_classes"].index(n) for n in base_meta["stuff_classes"]
                ],
            },
            trainable_flag=[
                1 if n in base_meta["stuff_classes"] else 0
                for n in meta["stuff_classes"]
            ],
        )
        # classification
        DatasetCatalog.register(
            all_name + "_classification",
            lambda x=image_dir, y=gt_dir: load_binary_mask(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(all_name + "_classification").set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="classification",
            ignore_label=255,
            evaluation_set={
                "base": [
                    meta["stuff_classes"].index(n) for n in base_meta["stuff_classes"]
                ],
            },
            trainable_flag=[
                1 if n in base_meta["stuff_classes"] else 0
                for n in meta["stuff_classes"]
            ],
            **meta,
        )
        # zero shot
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname + "_base")
        base_name = f"voc_base_sem_seg_{name}"

        DatasetCatalog.register(
            base_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(base_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **base_meta,
        )
        # classification
        DatasetCatalog.register(
            base_name + "_classification",
            lambda x=image_dir, y=gt_dir: load_binary_mask(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(base_name + "_classification").set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="classification",
            ignore_label=255,
            **base_meta,
        )
        # zero shot
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname + "_novel")
        novel_name = f"voc_novel_sem_seg_{name}"
        DatasetCatalog.register(
            novel_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(novel_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **novel_meta,
        )


def register_cofw(root, max_num):
    root = root+'OCFW/'
    meta = _get_voc_meta(CLASS_NAMES)
    image_dir = root
    gt_dir = root
    train_json = root + '/COFW_train_color.mat'
    val_json = root + '/COFW_test_color.mat'
    all_name = 'COFW_train'
    DatasetCatalog.register(all_name, lambda: load_sem_seg_cofw(root, train_json, max_num, 'train')) #D:\python_work\lib\detectron_repo\detectron2\data\datasets\coco.py

    MetadataCatalog.get(all_name).set(
        image_root=image_dir,
        sem_seg_root=gt_dir,
        evaluator_type="sem_seg",
        ignore_label=255,
        **meta,
    )

    image_dir = root
    gt_dir = root
    all_name = 'COFW_val'
    panoptic_root = 'E:/datasets/UFA/annotations/COFW_val/'
    panoptic_json = 'E:/datasets/UFA/annotations/COFW_gt_ibug.json'
    meta = _get_voc_meta(CLASS_NAMES_VAL)
    DatasetCatalog.register(all_name,lambda: load_sem_seg_cofw(root, val_json, max_num, 'test'),)

    MetadataCatalog.get(all_name).set(
        image_root=image_dir,
        sem_seg_root=gt_dir,
        panoptic_root=panoptic_root,
        panoptic_json=panoptic_json,
        evaluator_type="sem_seg",
        ignore_label=255,
        **meta,
    )


