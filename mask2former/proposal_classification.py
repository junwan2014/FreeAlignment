# Copyright (c) Facebook, Inc. and its affiliates.
import logging

import torch
import numpy as np
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.utils.logger import log_every_n, log_first_n
from detectron2.utils.events import get_event_storage
from detectron2.utils.visualizer import Visualizer
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import cv2

from .modeling.clip_adapter import (
    ClipAdapter,
    PredefinedPromptExtractor,
    LearnablePromptExtractor,
)
from .modeling.clip_adapter.clip import CLIP


@META_ARCH_REGISTRY.register()
class ProposalClipClassifier(nn.Module):
    @configurable
    def __init__(self, clip_adapter, task_names):
        super().__init__()
        self.clip_adapter = clip_adapter
        # store text features
        self.text_features = dict()
        self.task_names = task_names
        self.register_buffer(
            "pixel_mean", torch.Tensor(CLIP.PIXEL_MEAN).view(1, -1, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(CLIP.PIXEL_STD).view(1, -1, 1, 1), False
        )
        names = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                names.append(name)
        log_first_n(logging.INFO, names)


    @classmethod
    def from_config(cls, cfg):

        if cfg.MODEL.CLIP_ADAPTER.PROMPT_LEARNER == "predefined":
            prompt_learner = PredefinedPromptExtractor(
                cfg.MODEL.CLIP_ADAPTER.PREDEFINED_PROMPT_TEMPLATES
            )
        elif cfg.MODEL.CLIP_ADAPTER.PROMPT_LEARNER == "learnable":
            prompt_learner = LearnablePromptExtractor(
                prompt_dim=cfg.MODEL.CLIP_ADAPTER.PROMPT_DIM,
                prompt_shape=cfg.MODEL.CLIP_ADAPTER.PROMPT_SHAPE,
                task_prompt_shape=cfg.MODEL.CLIP_ADAPTER.TASK_PROMPT_SHAPE,
                task_names=cfg.INPUT.TASK_NAME,

            )

        else:
            raise NotImplementedError(
                "Prompt learner {} is not supported".format(
                    cfg.MODEL.CLIP_ADAPTER.PROMPT_LEARNER
                )
            )
        clip_adapter = ClipAdapter(
            cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME, prompt_learner
        )
        return {
            "clip_adapter": clip_adapter,
            "task_names": cfg.INPUT.TASK_NAME,
        }

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        """#这个时候数据已经读取出来了
        dataset_name = [x["meta"] for x in batched_inputs]  #mask2former/data/dataset_mappers/mask_former_binary_full_dataset_mapper.py 108
        assert len(set(dataset_name)) == 1
        dataset_name = dataset_name[0]
        images = [x["image"] for x in batched_inputs]
        images = torch.stack(images).cuda() #[bs, 3, 224, 224]
        #
        # ttmp = images[8].permute(1, 2, 0).cpu().detach().numpy()
        # plt.imshow(ttmp)
        # plt.show()

        # for x in batched_inputs:
        #     tmp = x["instances"]
        #     temp = x["instances"].gt_masks[0]

        masks = (
            torch.stack([x["instances"].gt_masks[0] for x in batched_inputs])
            .to(self.device)
            .type(images.dtype)
        )#[bs, 224, 224] GT——mask

        #
        # tmp = masks[10].cpu().numpy()
        # plt.imshow(tmp)
        # plt.show()

        if "task" in batched_inputs[0].keys():#每次只计算一个任务
            # print(batched_inputs[0]["task"])
            if batched_inputs[0]["task"] == "300W":
                task_name = "300W."
                # print(task_name)
            elif batched_inputs[0]["task"] == "COFW":
                task_name = "COFW."
                # print(task_name)
            elif batched_inputs[0]["task"] == "AFLW":
                task_name = "AFLW."
                # print(task_name)
            elif batched_inputs[0]["task"] == "WFLW":
                task_name = "WFLW."
                # print(task_name)
        else:
            task_name = "semantic segmentation."

        # task_name = "300W." #与数据集无关，这样才能捕捉不同数据集同一边界的互补信息

        class_names = [
            c.strip() for c in MetadataCatalog.get(dataset_name).stuff_classes
        ]#所有的类别，总共156类，基准类别。
        # normalize
        images = ( images * masks[:, None, ...] + (1 - masks[:, None, ...]) * self.pixel_mean)#对图像进行mask处理
        # img_temp = images[8].permute(1, 2, 0).cpu().detach().numpy()
        # plt.imshow(img_temp)
        # plt.show()
        #clip
        logits = self.clip_adapter(images, class_names, task_name) #[32, 8] 32张遮罩之后的图像和8类之间的关系   mask2former/modeling/clip_adapter/adapter.py line 20
        metadata = MetadataCatalog.get(dataset_name)

        if self.training:#计算损失
            target = torch.cat([x["instances"].gt_classes for x in batched_inputs]) #[bs] 每个图像对应的类别
            loss_cls = F.cross_entropy(logits, target.to(self.device))#计算交叉熵损失
            storage = get_event_storage()
            if storage.iter % 1000 == 0:
                vis = Visualizer(batched_inputs[0]["image"].permute(1, 2, 0).cpu().numpy().copy(),metadata,)
                vis_mask = target.new_ones(batched_inputs[0]["image"].shape[1:]) * 255
                vis_mask[batched_inputs[0]["instances"].gt_masks[0]] = batched_inputs[0]["instances"].gt_classes[0]
                vis.draw_sem_seg(vis_mask)
                # cv2.imshow('sample', vis.get_output().get_image())
                # cv2.waitKey(-1)
                pvis = Visualizer(batched_inputs[0]["image"].permute(1, 2, 0).cpu().numpy().copy(),metadata,)
                vis_mask = target.new_ones(batched_inputs[0]["image"].shape[1:]) * 255
                vis_mask[batched_inputs[0]["instances"].gt_masks[0]] = ( logits[0].argmax().detach().cpu())
                # ttmp2 = logits[0].argmax().detach().cpu()
                pvis.draw_sem_seg(vis_mask)
                storage.put_image( "train_data", np.concatenate([vis.get_output().get_image(), pvis.get_output().get_image()], axis=1,),)
                storage.put_scalar( "train_acc", 100.0 * (logits.detach().argmax(dim=1).cpu() == target).sum() / len(target),)
            # import pdb; pdb.set_trace()
            return loss_cls
        else:
            sim = [{"classification": logit[None].detach()} for logit in logits]
            return sim

    @property
    def device(self):
        return self.pixel_mean.device
