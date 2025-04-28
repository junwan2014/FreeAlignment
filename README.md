## Installation

### Environment
* python>=3.7
* torch>=1.10.0
* torchvision>=0.11.1
* timm>=0.6.12
* detectron2>=0.6.0 follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).
* requirement.txt

### Other dependency

The modified clip package.
```bash
cd third_party/CLIP
python -m pip install -Ue .
```

CUDA kernel for MSDeformAttn
```bash
cd mask2former/modeling/heads/ops
bash make.sh
```

### Datasets
You should download 300W, COFW, AFLW and WFLW datasets first.

## Getting Started

### Training 
To train a model with "train_net.py", first make sure the preparations are done. 
Take the training on COCO as an example.

Training prompts
```bash
python train_net.py --config-file configs/mask2former_learn_prompt_bs32_16k.yaml --num-gpus 8
```

Training model
```bash
python train_net.py --config-file configs/face-align/mask2former_R101c_alldataset_bs32_60k.yaml --num-gpus 8 MODEL.CLIP_ADAPTER.PROMPT_CHECKPOINT ${TRAINED_PROMPT_MODEL}
```

### Evaluation
```bash
python train_net.py --config-file configs/face-align/mask2former_R101c_alldataset_bs32_60k.yaml --num-gpus 8 --eval-only MODEL.WEIGHTS  ${TRAINED_MODEL}
```
