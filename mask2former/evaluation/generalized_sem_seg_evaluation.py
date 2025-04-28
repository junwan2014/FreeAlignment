# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch
import matplotlib.pyplot as plt

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from detectron2.utils import comm

from lib.inference import get_final_preds_match, get_coords
from lib.evaluate import compute_nme

from detectron2.evaluation import SemSegEvaluator


class GeneralizedSemSegEvaluator(SemSegEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        num_classes=None,
        ignore_label=None,
        post_process_func=None,
    ):
        super().__init__(
            dataset_name,
            distributed=distributed,
            output_dir=output_dir,
            num_classes=num_classes,
            ignore_label=ignore_label,
        )
        meta = MetadataCatalog.get(dataset_name)
        try:
            self._evaluation_set = meta.evaluation_set
        except AttributeError:
            self._evaluation_set = None
        self.post_process_func = (
            post_process_func
            if post_process_func is not None
            else lambda x, **kwargs: x
        )

    def process(self, inputs, outputs, outputs_flip):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        temp = inputs[0]["instances"]
        gt_classes = temp.gt_classes
        gt_masks = temp.gt_masks
        nums = len(gt_classes)
        gt = np.ones((gt_masks.size(1), gt_masks.size(2),), dtype=int)*self._num_classes
        for ii in range(nums):
            cur_mask = gt_masks[ii]
            gt[cur_mask] = gt_classes[ii]
        # plt.imshow(gt)
        # plt.show()
        output_tmp = outputs[0]["sem_seg"].argmax(dim=0).to(self._cpu_device)
        pred = np.array(output_tmp, dtype=np.int)
        # plt.imshow(pred)
        # plt.show()
        # import pdb; pdb.set_trace()
        gt[gt == self._ignore_label] = self._num_classes
        self._conf_matrix += np.bincount((self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1), minlength=self._conf_matrix.size,).reshape(self._conf_matrix.shape)
        # temp = self.encode_json_sem_seg(pred, input["file_name"])
        self._predictions.extend(self.encode_json_sem_seg(pred, inputs[0]["file_name"]))
        if inputs[0]["task"] == 'AFLW':
            self.land_predictions.append(
                {
                    "scale": inputs[0]["scale"],
                    "center": inputs[0]["center"],
                    "rotate": inputs[0]["rotate"],
                    "landmarks": inputs[0]["landmarks"],
                    "task": inputs[0]["task"],
                    "image_size": inputs[0]["image"].shape[1:],
                    "target": inputs[0]["target"],
                    "box_size": inputs[0]["box_size"],
                    "pred_land_logits": outputs[1]["pred_land_logits"],
                    "pred_land_pts": outputs[2]["pred_land_pts"],
                    "pred_land_logits_flip": outputs_flip[1]["pred_land_logits"],
                    "pred_land_pts_flip": outputs_flip[2]["pred_land_pts"],
                    "landmark_index": outputs[3]["landmark_index"],
                }
            )
        else:
            self.land_predictions.append(
                {
                    "scale": inputs[0]["scale"],
                    "center": inputs[0]["center"],
                    "rotate": inputs[0]["rotate"],
                    "landmarks": inputs[0]["landmarks"],
                    "task": inputs[0]["task"],
                    "image_size": inputs[0]["image"].shape[1:],
                    "target": inputs[0]["target"],
                    "pred_land_logits": outputs[1]["pred_land_logits"],
                    "pred_land_pts": outputs[2]["pred_land_pts"],
                    "pred_land_logits_flip": outputs_flip[1]["pred_land_logits"],
                    "pred_land_pts_flip": outputs_flip[2]["pred_land_pts"],
                    "landmark_index": outputs[3]["landmark_index"],
                }
            )
        # print(".....")

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=np.float)
        iou = np.full(self._num_classes, np.nan, dtype=np.float)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res["IoU-{}".format(name)] = 100 * iou[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res["ACC-{}".format(name)] = 100 * acc[i]
        if self._evaluation_set is not None:
            for set_name, set_inds in self._evaluation_set.items():
                iou_list = []
                set_inds = np.array(set_inds, np.int)
                mask = np.zeros((len(iou),)).astype(np.bool)
                mask[set_inds] = 1
                miou = np.sum(iou[mask][acc_valid[mask]]) / np.sum(iou_valid[mask])
                pacc = np.sum(tp[mask]) / np.sum(pos_gt[mask])
                res["mIoU-{}".format(set_name)] = 100 * miou
                res["pAcc-{}".format(set_name)] = 100 * pacc
                iou_list.append(miou)
                miou = np.sum(iou[~mask][acc_valid[~mask]]) / np.sum(iou_valid[~mask])
                pacc = np.sum(tp[~mask]) / np.sum(pos_gt[~mask])
                res["mIoU-un{}".format(set_name)] = 100 * miou
                res["pAcc-un{}".format(set_name)] = 100 * pacc
                iou_list.append(miou)
                res["hIoU-{}".format(set_name)] = (
                    100 * len(iou_list) / sum([1 / iou for iou in iou_list])
                )
        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)

        self.land_predictions = comm.gather(self.land_predictions)
        self.land_predictions = list(itertools.chain(*self.land_predictions))
        task = self.land_predictions[0]['task']
        nme_count = 0
        nme_batch_sum = 0
        nme_batch_sum_gt = 0
        nme_batch_sum_mean = 0
        for _prediction in self.land_predictions:
            num_joints = _prediction["landmarks"].shape[0]
            outputs = {}
            outputs['pred_logits'] = _prediction['pred_land_logits']
            outputs['pred_coords'] = _prediction['pred_land_pts']
            outputs['pred_logits_flip'] = _prediction['pred_land_logits_flip']
            outputs['pred_coords_flip'] = _prediction['pred_land_pts_flip']
            outputs['center'] = _prediction['center']
            outputs['scale'] = _prediction['scale']
            outputs['rotate'] = _prediction['rotate']
            landmark_index = _prediction['landmark_index']
            image_size = _prediction['image_size']
            preds, _, pred = get_final_preds_match(landmark_index, outputs, num_joints, image_size,
                                                   _prediction['center'], _prediction['scale'],
                                                   _prediction['rotate'])

            preds_flip, _, pred = get_final_preds_match(landmark_index, outputs, num_joints, image_size,
                                                        _prediction['center'], _prediction['scale'],
                                                        _prediction['rotate'], flip=True)
            pred_raw = _prediction['target'].numpy()
            pred_raw *= np.array(image_size)
            pred_gt = get_coords(pred_raw, outputs['center'], outputs['scale'], outputs['rotate'], image_size)
            pred_gt = torch.Tensor(pred_gt).unsqueeze(0)

            preds_mean = (preds + preds_flip) / 2
            # boxes = None
            boxes = []
            if _prediction['task'] == 'AFLW':
                boxes.append(_prediction['box_size'])
            nme_temp = compute_nme(preds, _prediction["landmarks"].unsqueeze(0), boxes)
            nme_batch_sum += np.sum(nme_temp)
            nme_temp_gt = compute_nme(pred_gt, _prediction["landmarks"].unsqueeze(0), boxes)
            nme_batch_sum_gt += np.sum(nme_temp_gt)
            nme_temp_mean = compute_nme(preds_mean, _prediction["landmarks"].unsqueeze(0), boxes)
            nme_batch_sum_mean += np.sum(nme_temp_mean)
            nme_count = nme_count + preds.shape[0]
        nme = nme_batch_sum / nme_count
        nme_mean = nme_batch_sum_mean / nme_count
        nme_gt = nme_batch_sum_gt / nme_count
        nme_results = {}
        nme_results[task] = nme
        task_1 = task + '_mean'
        nme_results[task_1] = nme_mean
        task_2 = task + '_gt'
        nme_results[task_2] = nme_gt
        self._logger.info(nme_results)
        return nme_results

