# AVE-UWSSOD：Adaptive Visual Enhancement and Semi-Supervised Learning for Robust Underwater Object Detection
AVE-UWSSOD is a network model focused on underwater target detection that combines visual enhancement and a semi-supervised object detection framework based on co-training image preprocessing methods and high-quality pseudo-labeling to achieve robust underwater object detection.
![AVE-UWSSOD Image](https://github.com/Fly-fate/AVE-UWSSOD/blob/master/docs/AVE-SSOD.png)
![AVE-UWSSOD workflows](https://github.com/Fly-fate/AVE-UWSSOD/blob/master/docs/workflows.png)

## To reproduce the results in the paper
<details>
<summary>Installation</summary>

Install AVE-UWSSOD from source.
```shell
https://github.com/Fly-fate/AVE-UWSSOD.git
pip3 install -v -e .  # or  python3 setup.py develop
```

</details>

<details>
<summary>Download data</summary>

```shell
mkdir -p ${VOCDIR}
cd ${VOCDIR}
wget https://drive.usercontent.google.com/download?id=1PgP7gY1FkcpQ1D6XW_lPzTYCgsMhItbw&export=download&authuser=0&confirm=t&uuid=2f3b1cef-e48f-430d-a015-6ecba5466ebd&at=AENtkXY_kzd_2W_mpgmnLWiPMl4c%3A1731054381954
tar -xf URPC2020_detection.tar
```

</details>

<details>
<summary>Training</summary>

**Step1.Train model on labeled data**
```shell
python tools/train.py \
    --batch-size  64 \
    --exp_file  ./exps/example/yolox_voc/yolox_voc_nano.py \
    --ckpt /path/to/your/yolox_nano.pth \
```

**Step2.Generate pseudo labels of unlabeled data**
```shell
python tools/eval_ssl.py \
    -path /path/to/your/unlabel_dataset \
    --exp_file ./exps/example/yolox_voc/yolox_voc_nano.py \
    --ckpt /path/to/your/ckpt.pth \
```

**Step3.Train AVE-UWSSOD**
```shell
python tools/train_ssl.py \
    --batch-size 8 \
    --exp_file ./exps/example/yolox_voc/yolox_voc_nano_ssl.py \
    --batch-size-unlabeled 8
```
</details>

<details>
<summary>Evaluation</summary>

```shell
python tools/eval.py \
    --exp_file ./exps/example/yolox_voc/yolox_voc_nano_ssl.py \
    --batch-size 64 \
    --devices 2 \
    --ckpt /path/to/your/ckpt.pth 
```

</details>

## Acknowledgements
For the comparison against other methods, we use the official implementations from the following repositories:
-   [STAC: A Simple Semi-Supervised Learning Framework for Object Detection](https://github.com/google-research/ssl_detection/tree/master)
-   [YOLOX: Exceeding YOLO Series in 2021](https://github.com/Megvii-BaseDetection/YOLOX?tab=readme-ov-file)
