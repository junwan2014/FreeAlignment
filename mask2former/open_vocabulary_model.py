# Copyright (c) Facebook, Inc. and its affiliates.
from cgitb import text
import logging
import copy
import random
import os
from typing import Tuple
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.logger import log_first_n
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data.transforms import ResizeTransform
from .modeling.clip_adapter import (
    ClipAdapter,
    MaskFormerClipAdapter,
    build_prompt_learner,
)
from .mask_former_model import MaskFormer
from lib.evaluate import get_transformer_coords, compute_nme, compute_nme_io
from lib.inference import get_final_preds_match


@META_ARCH_REGISTRY.register()
class OpenVocabulary(MaskFormer):

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        clip_adapter: nn.Module,
        region_clip_adapter: nn.Module = None,
        task_names: list,
        criterion: nn.Module,
        criterion_land: nn.Module,
        num_queries: int,
        semantic_on: bool,
        instance_on: bool,
        panoptic_on: bool,
        landmark_index_cofw: list,
        landmark_index_aflw: list,
        landmark_index_wflw: list,
        landmark_index_300w: list,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        clip_ensemble: bool,
        clip_ensemble_weight: float,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        test_topk_per_image: int,
        cfg,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            clip_adapter: adapter for clip-based mask classification
            num_queries: int, number of queries
            panoptic_on: bool, whether to output panoptic segmentation prediction
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            criterion_land=criterion_land,
            num_queries=num_queries,
            semantic_on=semantic_on,
            instance_on=instance_on,
            panoptic_on=panoptic_on,
            landmark_index_cofw =landmark_index_cofw,
            landmark_index_aflw=landmark_index_aflw,
            landmark_index_wflw=landmark_index_wflw,
            landmark_index_300w=landmark_index_300w,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            metadata=metadata,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        self.clip_adapter: ClipAdapter = clip_adapter
       
        self._region_clip_adapter = region_clip_adapter

        self.task_names = task_names
        self.clip_ensemble: bool = clip_ensemble
        self.clip_ensemble_weight: float = clip_ensemble_weight

        self.test_topk_per_image = test_topk_per_image
        self.landmark_index_300w = landmark_index_300w
        self.landmark_index_aflw = landmark_index_aflw
        self.landmark_index_wflw = landmark_index_wflw
        self.landmark_index_cofw = landmark_index_cofw


    @classmethod
    def from_config(cls, cfg):
        init_kwargs = MaskFormer.from_config(cfg)
        prompt_learner = build_prompt_learner(cfg.MODEL.CLIP_ADAPTER, cfg.INPUT.TASK_NAME)
        region_clip_adapter = None
        if cfg.MODEL.CLIP_ADAPTER.SEPERATE_ADAPTER:
            log_first_n(
                logging.WARNING,
                "Using different head for region classification and query classification",
            )
            cls_prompt_learner = build_prompt_learner(
                cfg.MODEL.CLIP_ADAPTER, cfg.INPUT.TASK_NAME
            )
            region_clip_adapter = MaskFormerClipAdapter(
                cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.CLIP_MODEL_NAME,
                cls_prompt_learner,
                mask_fill=cfg.MODEL.CLIP_ADAPTER.MASK_FILL,
                mask_expand_ratio=cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO,
                mask_thr=0.4,
                mask_matting=cfg.MODEL.CLIP_ADAPTER.MASK_MATTING,
                region_resized=cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED,
            )

        clip_adapter = MaskFormerClipAdapter(
            cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME,
            prompt_learner,
            mask_fill=cfg.MODEL.CLIP_ADAPTER.MASK_FILL,
            mask_expand_ratio=cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO,
            mask_thr=cfg.MODEL.CLIP_ADAPTER.MASK_THR,
            mask_matting=cfg.MODEL.CLIP_ADAPTER.MASK_MATTING,
            region_resized=cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED,
        )
        
        init_kwargs["clip_adapter"] = clip_adapter
        init_kwargs["region_clip_adapter"] = region_clip_adapter
        init_kwargs["task_names"] = cfg.INPUT.TASK_NAME

        init_kwargs["landmark_index_300w"] = cfg.DATASETS.LANDMARK_INDEX_300W
        init_kwargs["landmark_index_aflw"] = cfg.DATASETS.LANDMARK_INDEX_AFLW
        init_kwargs["landmark_index_wflw"] = cfg.DATASETS.LANDMARK_INDEX_WFLW
        init_kwargs["landmark_index_cofw"] = cfg.DATASETS.LANDMARK_INDEX_COFW

        init_kwargs["clip_ensemble"] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE
        init_kwargs[
            "clip_ensemble_weight"
        ] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT
        init_kwargs["test_topk_per_image"] = cfg.TEST.DETECTIONS_PER_IMAGE
        init_kwargs["metadata"] = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        init_kwargs["semantic_on"] = "semantic segmentation." in cfg.INPUT.TASK_NAME
        init_kwargs["instance_on"] = "instance segmentation." in cfg.INPUT.TASK_NAME
        init_kwargs["panoptic_on"] = "panoptic segmentation." in cfg.INPUT.TASK_NAME

        init_kwargs["cfg"] = cfg

        return init_kwargs


    def forward(self, batched_inputs, flip=None, text_labels=None):
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
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        if text_labels == None:
            dataset_name = [x["meta"] for x in batched_inputs]
            assert len(set(dataset_name)) == 1
            dataset_name = dataset_name[0]
        else:
            dataset_name = " " 
        
        
        images = [x["image"].to(self.device) for x in batched_inputs] #mask2former/data/dataset_mappers/coco_full_task_new_baseline_dataset_mapper.py line 152
        image_size = images[0].size()[1:]
        # images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)#[bs, 3, 640, 640]
        #使用resnet提取图像特征
        features = self.backbone(images.tensor)#[bs, 256, 160, 160], [bs, 512, 80, 80], [bs, 1024, 40, 40], [bs, 2048, 20, 20]

        land_target = batched_inputs[0]["target"].to(self.device)
        if land_target.size(0) == 68:
            landmark_index = self.landmark_index_300w
        elif land_target.size(0) == 29:
            landmark_index = self.landmark_index_cofw
        elif land_target.size(0) == 19:
            landmark_index = self.landmark_index_aflw
        else:
            landmark_index = self.landmark_index_wflw

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

        if text_labels == None:
            class_names = self.get_class_name_list(dataset_name) #156
        else:
            class_names = text_labels

        if self.training:
            # task_name = random.choice(self.task_names)#这里为什么随机选择一个任务，因为3个标记都有
            # task_name = "300W."
            text_features = self.clip_adapter.get_text_features(class_names, task_name)#得到已知类（156类）和另外1类的 文本提示编码[157, 512]
            # fused_text_features [157, 512] 嵌入文本特征
            outputs, fused_text_features = self.sem_seg_head(features, text_features) # mask2former/modeling/heads/mask_former_interaction_head.py line 136
            # outputs["pred_logits"]: [bs, 157, 512]：提示的目标  [bs, 100, 512]：推测的100个目标  [bs, 100, 512]x[bs, 512, 157]=[bs, 100, 157] 推测的100个目标的取值概率
            outputs["pred_logits"] = self.clip_adapter.get_sim_logits(text_features, self.clip_adapter.normalize_feature(outputs["pred_logits"]))

            if "aux_outputs" in outputs.keys(): #分别计算每个解码器输出的目标的概率（提示的目标与生成的目标之间的相关性）
                for i in range(len(outputs["aux_outputs"])):
                    outputs["aux_outputs"][i]["pred_logits"] = self.clip_adapter.get_sim_logits(text_features, self.clip_adapter.normalize_feature(outputs["aux_outputs"][i]["pred_logits"]),)
            # mask classification target
            # if task_name == "semantic segmentation.":
            #     gt_instances = [x["sem_instances"].to(self.device) for x in batched_inputs]
            # elif task_name == "instance segmentation.":
            #     gt_instances = [x["ins_instances"].to(self.device) for x in batched_inputs]
            # elif task_name == "panoptic segmentation.":
            #     gt_instances = [x["pan_instances"].to(self.device) for x in batched_inputs]

            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            # temp = gt_instances[0]
            # temp1 = temp.gt_masks
            # temp2 = temp.gt_classes
            # for ii in range(temp1.size(0)):
            #     tmp = temp1[ii].detach().cpu().numpy()
            #     plt.imshow(tmp)
            #     plt.show()
            targets = self.prepare_targets(gt_instances, images)
            losses = self.criterion(outputs, targets) #mask2former/modeling/criterion.py line 194

            land_target = [x["target"].to(self.device) for x in batched_inputs]
            land_target = torch.stack(land_target)
            land_target_weight = [x["target_weight"].to(self.device) for x in batched_inputs]
            land_target_weight = torch.stack(land_target_weight)

            # output = outputs[dataset]
            loss_dict, pred =  self.criterion_land(outputs, land_target, land_target_weight, landmark_index)
            pred = pred*image_size[0]
            # pred_1 = land_target.detach().cpu()*image_size[0]
            preds = get_transformer_coords(pred, batched_inputs,image_size)
            # preds_1 = get_transformer_coords(pred_1, batched_inputs, image_size)
            dt_name = batched_inputs[0]['task']
            boxes = None
            if dt_name == "AFLW":
                boxes = [x["box_size"] for x in batched_inputs]
            meta = [x["landmarks"] for x in batched_inputs]
            meta = torch.stack(meta)
            nme_batch = np.mean(compute_nme(preds, meta, boxes))
            # nme_batch_1 = np.mean(compute_nme(preds_1, meta, boxes))
            # print(dt_name, nme_batch, nme_batch_1)
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            for ii in list(loss_dict.keys()):
                if ii in self.criterion_land.weight_dict:
                    loss_dict[ii] *= self.criterion_land.weight_dict[ii]
                    losses[ii] = loss_dict[ii]
                else:
                    # remove this loss if not specified in `weight_dict`
                    loss_dict.pop(ii)


            losses['nme_batch'] = nme_batch
            # losses = dict(losses.items() + loss_dict.items())
            return losses
        else:
            # task_name = "panoptic segmentation."

            text_features = self.clip_adapter.get_text_features(class_names, task_name)

            outputs, fused_text_features = self.sem_seg_head(features, text_features)
            # outputs["pred_logits"]: [1, 100, 134]   outputs["pred_masks"] [1, 100, w, h]
            outputs["pred_logits"] = self.clip_adapter.get_sim_logits(text_features,
                                                                      self.clip_adapter.normalize_feature(
                                                                          outputs["pred_logits"]))

            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            mask_pred_results = F.interpolate(mask_pred_results,
                                              size=(images.tensor.shape[-2], images.tensor.shape[-1]), mode="bilinear",
                                              align_corners=True, )
            processed_results = []
            if flip!=True:
                for i, (mask_cls_result, mask_pred_result, input_per_image, image_size) in enumerate(zip(mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes)):
                    height = image_size[0]
                    width = image_size[1]
                    mask_pred_result = sem_seg_postprocess(mask_pred_result, image_size, height, width)  # 线性插值，没用
                    image = input_per_image["image"].to(self.device)

                    panoptic_r = self.panoptic_inference(mask_cls_result, mask_pred_result, image, class_names, task_name, dataset_name)

                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])

                    # process results
                    cur_device = panoptic_r[0].device
                    panoptic_mask = panoptic_r[0].cpu().numpy().astype(np.uint8)
                    ori_h, ori_w = panoptic_mask.shape[0], panoptic_mask.shape[1]
                    transform = ResizeTransform(ori_h, ori_w, height, width)
                    panoptic_mask = transform.apply_segmentation(panoptic_mask)
                    panoptic_r[0] = torch.tensor(panoptic_mask).to(cur_device)

                    segment_info = panoptic_r[1]
                    cur_seg_ids = list(torch.unique(panoptic_r[0]))
                    segment_info = [seg_info for seg_info in segment_info if seg_info["id"] in cur_seg_ids]
                    panoptic_r[1] = segment_info
                    processed_results.append({"panoptic_seg": panoptic_r})
            processed_results.append({"pred_land_logits": outputs['pred_land_logits']})
            processed_results.append({"pred_land_pts": outputs['pred_land_pts']})
            processed_results.append({"landmark_index": landmark_index})
            return processed_results

    def semantic_inference(self, mask_cls, mask_pred, image, class_names, task_name, dataset_name):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1] #[100, cls-1]去掉背景类
        mask_pred = mask_pred.sigmoid()
        
        # get the classification result from clip model
        
        if self.clip_ensemble:
            #clip_cls [valid_num, 134] 预测的88个mask对应的目标分别属于134类中的哪一类的概率
            clip_cls, valid_flag = self.region_clip_adapter(image, class_names, task_name, mask_pred, normalize=True) # mask2former/modeling/clip_adapter/adapter.py Line 111
            
            if clip_cls is None:
                clip_cls = torch.empty(0, mask_cls.shape[-1] + 1, device=self.device)
            
            clip_cls = F.softmax(clip_cls[:, :-1], dim=-1) #去掉背景类
            if self.clip_ensemble_weight > 0:
                map_back_clip_cls = mask_cls.new_ones(mask_cls.shape) #[100, 133] 全1
                map_back_clip_cls[valid_flag] = clip_cls # #[100, 133] valid替换成clip_cls
                if hasattr(MetadataCatalog.get(dataset_name), "trainable_flag"):
                    trained_mask = torch.Tensor( MetadataCatalog.get(dataset_name).trainable_flag).to(mask_cls.device)[None, :] #[1, 133]
                else:
                    trained_mask = mask_cls.new_zeros(mask_cls.shape)
                # mask_cls [100, 133]
                mask_cls = trained_mask * torch.pow(mask_cls, self.clip_ensemble_weight) * torch.pow(map_back_clip_cls, 1 - self.clip_ensemble_weight) +(1 - trained_mask) * torch.pow( mask_cls, 1 - self.clip_ensemble_weight) * torch.pow( map_back_clip_cls, self.clip_ensemble_weight)
            else:
                mask_cls = clip_cls
                mask_pred = mask_pred[valid_flag]
        #[133, w, h]
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred, image, class_names, task_name, dataset_name):
        
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        
        
        if self.clip_ensemble:
            clip_cls, valid_flag = self.region_clip_adapter(
                image, class_names, task_name, mask_pred, normalize=True
            ) #返回有用的查询对应的匹配结果 [valid_query, num_class+1]
            if clip_cls is None:
                clip_cls = torch.empty(0, mask_cls.shape[-1] + 1, device=self.device)
            
            clip_cls = F.softmax(clip_cls[:, :-1], dim=-1) #去掉背景类，对概率进行softmax处理，归一化，和为1
            if self.clip_ensemble_weight > 0:
                map_back_clip_cls = mask_cls.new_ones(mask_cls.shape) #无用的置为1
                map_back_clip_cls[valid_flag] = clip_cls #有用的置为真实的概率
                if hasattr(MetadataCatalog.get(dataset_name), "trainable_flag"):
                    trained_mask = torch.Tensor(
                        MetadataCatalog.get(dataset_name).trainable_flag
                    ).to(mask_cls.device)[None, :]
                else:
                    trained_mask = mask_cls.new_zeros(mask_cls.shape)
                
                mask_cls = trained_mask * torch.pow(
                    mask_cls, self.clip_ensemble_weight
                ) * torch.pow(map_back_clip_cls, 1 - self.clip_ensemble_weight) + (
                    1 - trained_mask
                ) * torch.pow(
                    mask_cls, 1 - self.clip_ensemble_weight
                ) * torch.pow(
                    map_back_clip_cls, self.clip_ensemble_weight
                )
            else:
                mask_cls = clip_cls
                mask_pred = mask_pred[valid_flag]
        #mask_cls [100, 133], mask_pred [100, w, h], sem_maps [133, w, h]
        sem_maps = torch.einsum("qc,qhw->chw", mask_cls, mask_pred).argmax(0)
        # temp = sem_maps.detach().cpu().numpy()
        # plt.imshow(temp)
        # plt.show()
        #这个应该表示分割的结果图
        #scores 对应最大的类别的概率， labels对应最大类别的编号
        scores, labels = F.softmax(mask_cls / 0.01, dim=-1).max(-1)
        keep = labels.ne(self.sem_seg_head.num_classes)  # 返回与输入具有相同形状的张量数组，若对应位置上的元素不相等，则该位置上的元素是True，否则是False。
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device) #初始化为全零
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            return [panoptic_seg, segments_info]
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                pred_class_name = class_names[pred_class]
                isthing = pred_class_name in self.metadata.thing_classes

                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_masks[k] >= 0.5) & (sem_maps == pred_class)  # sem_maps为预测的语义分割结果
                mask_area = mask.sum().item()

                if original_area > 0 and mask.sum().item() > 0:  # 单个实力预测mask值大于0.5的总数>0, 并且这个预测和语义分割的预测一致
                    if mask_area / original_area < self.overlap_threshold:  # 如果重叠比例小于0.5
                        continue
                    if isthing and cur_scores[k] < 0.5:  # 如果当前query的分数小于0.5
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1  # 否则当前的实例号+1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

                    
            panoptic_res = [panoptic_seg, segments_info]
            return panoptic_res

    def instance_inference(self, mask_cls, mask_pred, image, class_names, task_name, dataset_name):
        
        image_size = mask_pred.shape[-2:]
        num_classes = mask_cls.shape[-1]
        
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        if self.clip_ensemble:
            clip_cls, valid_flag = self.region_clip_adapter( image, class_names, task_name, mask_pred.sigmoid(), normalize=True)
            if clip_cls is None:
                clip_cls = torch.empty(0, mask_cls.shape[-1] + 1, device=self.device)
           
            clip_cls = F.softmax(clip_cls[:, :-1], dim=-1)
            if self.clip_ensemble_weight > 0:
                map_back_clip_cls = mask_cls.new_ones(mask_cls.shape)
                map_back_clip_cls[valid_flag] = clip_cls
                if hasattr(MetadataCatalog.get(dataset_name), "trainable_flag"):
                    trained_mask = torch.Tensor(
                        MetadataCatalog.get(dataset_name).trainable_flag
                    ).to(mask_cls.device)[None, :]
                else:
                    trained_mask = mask_cls.new_zeros(mask_cls.shape)
                
                mask_cls = trained_mask * torch.pow(
                    mask_cls, self.clip_ensemble_weight
                ) * torch.pow(map_back_clip_cls, 1 - self.clip_ensemble_weight) + (
                    1 - trained_mask
                ) * torch.pow(
                    mask_cls, 1 - self.clip_ensemble_weight
                ) * torch.pow(
                    map_back_clip_cls, self.clip_ensemble_weight
                )
            else:
                mask_cls = clip_cls
                mask_pred = mask_pred[valid_flag]
        # sem_maps[w, h]
        sem_maps = torch.einsum("qc,qhw->chw", mask_cls, mask_pred.sigmoid()).argmax(0)


        scores = F.softmax(mask_cls / 0.01, dim=-1)[:, :-1]
        scores_per_image, labels_per_image = scores.max(-1)


        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                pred_class_name = class_names[lab]
                keep[i] = pred_class_name in self.metadata.thing_classes 

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        class_mask_memory = {}
        keep = torch.zeros_like(scores_per_image).bool()
    
        for k in range(labels_per_image.shape[0]):
            
            pred_class = labels_per_image[k]
            original_area = (mask_pred[k] >= 0.5).sum().item()
            
            mask = (mask_pred[k] >= 0.5) & (sem_maps == pred_class)
            mask_area = mask.sum().item()

            if mask_area > 0 and original_area > 0 and scores_per_image[k] > 0.5:
                if mask_area / original_area > self.overlap_threshold:
                    keep[k] = True

                    if lab in class_mask_memory.keys():
                        class_mask_memory[lab].append(k)
                    else: 
                        class_mask_memory[lab] = [k]
        
        for cls_id, idx_list in class_mask_memory.items():
            mask_area_list = [(mask_pred[i] >= 0.5).sum().item() for i in idx_list]
            max_area = np.max(np.array(mask_area_list))
            max_idx = np.argmax(np.array(mask_area_list))
            union_mask = torch.zeros_like(mask_pred[0]).bool()
            for i, idx in enumerate(idx_list):
                if i != max_idx:
                    union_mask = (union_mask ==True) | (mask_pred[idx] >= 0.5) 
            union_mask_area = union_mask.sum().item()
            if union_mask_area / max_area > 0.8:
                keep[idx_list[max_idx]] = False

        
        scores_per_image = scores_per_image[keep]
        labels_per_image = labels_per_image[keep]
        mask_pred = mask_pred[keep]
        
        result = Instances(image_size)
        
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

    def get_class_name_list(self, dataset_name):
        class_names = [
            c.strip() for c in MetadataCatalog.get(dataset_name).stuff_classes
        ]
        return class_names


    @property
    def region_clip_adapter(self):
        if self._region_clip_adapter is None:
            return self.clip_adapter
        return self._region_clip_adapter
