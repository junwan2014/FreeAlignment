# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import torch
import io
import itertools
import json
import logging
import numpy as np
import os
import tempfile
from collections import OrderedDict
from typing import Optional
from PIL import Image
from tabulate import tabulate
from lib.inference import get_final_preds_match
from lib.evaluate import compute_nme
import matplotlib.pyplot as plt

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from lib.inference import get_coords
from panopticapi.utils import id2rgb
from panopticapi.evaluation import pq_compute, pq_compute_multi_core

from detectron2.evaluation.evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

class GeneralizedPanopticEvaluator(DatasetEvaluator):
    """
    Evaluate Panoptic Quality metrics on COCO using PanopticAPI.
    It saves panoptic segmentation prediction in `output_dir`

    It contains a synchronize call and has to be called from all workers.
    """

    def __init__(self, dataset_name: str, output_dir: Optional[str] = None):
        """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        # self._thing_contiguous_id_to_dataset_id = {
        #     v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
        # }
        # self._stuff_contiguous_id_to_dataset_id = {
        #     v: k for k, v in self._metadata.stuff_dataset_id_to_contiguous_id.items()
        # }

        self._output_dir = output_dir
        if self._output_dir is not None:
            PathManager.mkdirs(self._output_dir)

    def reset(self):
        self._predictions = []
        self.land_predictions = []

    def _convert_category_id(self, segment_info):
        isthing = segment_info.pop("isthing", None)
        if isthing is None:
            # the model produces panoptic category id directly. No more conversion needed
            return segment_info
        if isthing is True:
            segment_info["category_id"] = self._thing_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = self._stuff_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        return segment_info

    def process(self, inputs, outputs, outputs_flip):
        # tmp = inputs[0]["image"].size()[1:]
        # temp = inputs[0]["image"].shape[1:]
        if inputs[0]["task"]=='AFLW':
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
                    "pred_land_logits_flip": outputs_flip[0]["pred_land_logits"],
                    "pred_land_pts_flip": outputs_flip[1]["pred_land_pts"],
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
                    "pred_land_logits_flip": outputs_flip[0]["pred_land_logits"],
                    "pred_land_pts_flip": outputs_flip[1]["pred_land_pts"],
                    "landmark_index": outputs[3]["landmark_index"],
                }
            )

        panoptic_img, segments_info = outputs[0]["panoptic_seg"]
        panoptic_img = panoptic_img.cpu().numpy()
        # plt.imshow(panoptic_img)
        # plt.show()
        # import pdb; pdb.set_trace()
        if segments_info is None:
            # If "segments_info" is None, we assume "panoptic_img" is a
            # H*W int32 image storing the panoptic_id in the format of
            # category_id * label_divisor + instance_id. We reserve -1 for
            # VOID label, and add 1 to panoptic_img since the official
            # evaluation script uses 0 for VOID label.
            label_divisor = self._metadata.label_divisor
            segments_info = []
            for panoptic_label in np.unique(panoptic_img):
                if panoptic_label == -1 :
                        # VOID region.
                    continue
                pred_class = panoptic_label // label_divisor

                isthing = ( pred_class in self._metadata.thing_dataset_id_to_contiguous_id.values())
                segments_info.append(
                    {
                        "id": int(panoptic_label),
                        "category_id": int(pred_class),
                        "isthing": bool(isthing),
                        'instance': input['instance'],
                    }
                )
            # Official evaluation script uses 0 for VOID label.

        file_name = os.path.basename(inputs[0]["file_name"])
        file_name_png = os.path.splitext(file_name)[0] + ".png"

        image_id = os.path.splitext(file_name)[0]
        with io.BytesIO() as out:
            # temp = id2rgb(panoptic_img)
            # temp1 = Image.fromarray(temp)
            # temp1.save(out, format="PNG")

            Image.fromarray((id2rgb(panoptic_img.astype(np.int32))).astype(np.uint8)).save(out, format="PNG")
            # segments_info = [self._convert_category_id(x) for x in segments_info]
            self._predictions.append(
                {
                    "image_id": image_id,
                    "file_name": file_name_png,
                    "png_string": out.getvalue(),
                    "segments_info": segments_info,
                    # 'instances': inputs[0]['instances'],
                }
            )

    def evaluate(self, name_tmp):
        comm.synchronize()

        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        # pred_annotations = {el['image_id']: el for el in self._predictions}
        gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)

        pred_dir = self._output_dir+  "/panoptic_seg/" + str(name_tmp)
        os.makedirs(pred_dir, exist_ok=True)
        logger.info("Writing all panoptic predictions to {} ...".format(pred_dir))
        for p in self._predictions:
            with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                f.write(p.pop("png_string"))

        with open(gt_json, "r") as f:
            json_data = json.load(f)
        json_data["annotations"] = self._predictions

        output_dir = pred_dir
        temp = self.land_predictions[0]['task']+"_predictions.json"
        predictions_json = os.path.join(output_dir, temp)
        with PathManager.open(predictions_json, "w") as f:
            f.write(json.dumps(json_data))

        with contextlib.redirect_stdout(io.StringIO()):
            pq_res = pq_compute( #C:\anaconda3\envs\py38\Lib\site-packages\panopticapi\evaluation.py
                gt_json,
                PathManager.get_local_path(predictions_json),
                gt_folder=gt_folder,
                pred_folder=pred_dir,
            )

        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        results = OrderedDict({"panoptic_seg": res})
        print(results)
        _print_panoptic_results(pq_res)
        print(pq_res["per_class"])

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
            outputs['pred_logits'] =  _prediction['pred_land_logits']
            outputs['pred_coords'] = _prediction['pred_land_pts']
            outputs['pred_logits_flip'] = _prediction['pred_land_logits_flip']
            outputs['pred_coords_flip'] = _prediction['pred_land_pts_flip']
            outputs['center'] = _prediction['center']
            outputs['scale'] = _prediction['scale']
            outputs['rotate'] = _prediction['rotate']
            landmark_index = _prediction['landmark_index']
            image_size = _prediction['image_size']
            preds, _, pred = get_final_preds_match(landmark_index, outputs, num_joints, image_size, _prediction['center'], _prediction['scale'],
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
            if _prediction['task']=='AFLW':
                boxes.append(_prediction['box_size'])
            nme_temp = compute_nme(preds, _prediction["landmarks"].unsqueeze(0), boxes)
            nme_batch_sum += np.sum(nme_temp)

            nme_temp_gt = compute_nme(pred_gt, _prediction["landmarks"].unsqueeze(0), boxes)
            nme_batch_sum_gt += np.sum(nme_temp_gt)



            nme_temp_mean = compute_nme(preds_mean, _prediction["landmarks"].unsqueeze(0), boxes)
            nme_batch_sum_mean += np.sum(nme_temp_mean)

            # nme_batch_loss0 += np.sum(nme_temp_loss0)
            nme_count = nme_count + preds.shape[0]
        nme = nme_batch_sum / nme_count
        nme_mean = nme_batch_sum_mean / nme_count
        nme_gt = nme_batch_sum_gt/nme_count
        nme_results = {}
        nme_results[task] = nme
        task_1 = task + '_mean'
        nme_results[task_1] = nme_mean
        task_2 = task + '_gt'
        nme_results[task_2] = nme_gt
        print(nme_results)
        return nme_results


def _print_panoptic_results(pq_res):
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All"]:
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    logger.info("Panoptic Evaluation Results:\n" + table)


if __name__ == "__main__":
    from detectron2.utils.logger import setup_logger

    logger = setup_logger()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json")
    parser.add_argument("--gt-dir")
    parser.add_argument("--pred-json")
    parser.add_argument("--pred-dir")
    args = parser.parse_args()

    from panopticapi.evaluation import pq_compute

    with contextlib.redirect_stdout(io.StringIO()):
        pq_res = pq_compute(
            args.gt_json, args.pred_json, gt_folder=args.gt_dir, pred_folder=args.pred_dir
        )
        _print_panoptic_results(pq_res)