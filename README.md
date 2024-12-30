# AVE-UWSSOD：Adaptive Visual Enhancement and Semi-Supervised Learning for Robust Underwater Object Detection
AVE-UWSSOD is a network model focused on underwater target detection that combines visual enhancement and a semi-supervised object detection framework based on co-training image preprocessing methods and high-quality pseudo-labeling to achieve robust underwater object detection.
![AVE-UWSSOD Image](https://github.com/Fly-fate/AVE-UWSSOD/blob/master/docs/AVE-SSOD.png)
![AVE-UWSSOD workflows](https://github.com/Fly-fate/AVE-UWSSOD/blob/master/docs/workflows.png)

### Installation

Install AVE-UWSSOD from source.
```shell
https://github.com/Fly-fate/AVE-UWSSOD.git
pip3 install -v -e .  # or  python3 setup.py develop
```

### Training

The following example uses labeled data as 10% train and rest 90% train data as unlabeled data, with YOLOX-Nano as the baseline model.

- **Step1.Train model on labeled data**
This step trains a standard detector on labeled data

```shell
python tools/train.py \
    --batch-size  64 \
    --exp_file  ./exps/example/yolox_voc/yolox_voc_nano.py \
    --ckpt /path/to/your/yolox_nano.pth \
```

- **Step2.Generate pseudo labels of unlabeled data**
In this step we use unlabeled data for inference to generate high quality for labeling while Set to use original images rather than resized ones for inference.
```shell
python tools/eval_ssl.py \
    -path /path/to/your/unlabel_dataset \
    --exp_file ./exps/example/yolox_voc/yolox_voc_nano.py \
    --ckpt /path/to/your/ckpt.pth \
```

- **Step3.Train AVE-UWSSOD**
 Use labeled data and unlabeled data with pseudo labels to train a AVE-UWSSOD detector
```shell
python tools/train_ssl.py \
    --batch-size 8 \
    --exp_file ./exps/example/yolox_voc/yolox_voc_nano_ssl.py \
    --batch-size-unlabeled 8
```



### Evaluation

```shell
python tools/eval.py \
    --exp_file ./exps/example/yolox_voc/yolox_voc_nano_ssl.py \
    --batch-size 64 \
    --devices 2 \
    --ckpt /path/to/your/ckpt.pth 
```


## Acknowledgements
For the comparison against other methods, we use the official implementations from the following repositories:
-   [STAC: A Simple Semi-Supervised Learning Framework for Object Detection](https://github.com/google-research/ssl_detection/tree/master)
-   [YOLOX: Exceeding YOLO Series in 2021](https://github.com/Megvii-BaseDetection/YOLOX?tab=readme-ov-file)
