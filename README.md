# AVE-UWSSOD：Adaptive Visual Enhancement and Semi-Supervised Learning for Robust Underwater Object Detection
AVE-UWSSOD is a network model focused on underwater target detection that combines visual enhancement and a semi-supervised object detection framework based on co-training image preprocessing methods and high-quality pseudo-labeling to achieve robust underwater object detection.
![AVE-UWSSOD Image](https://github.com/Fly-fate/AVE-UWSSOD/blob/master/docs/AVE-SSOD.png)
![AVE-UWSSOD workflows](https://github.com/Fly-fate/AVE-UWSSOD/blob/master/docs/workflows.png)

## Data Provided
Since the URPC2020 dataset exceeds GitHub's size limit, we provide a download link for the URPC2020 dataset [here](https://drive.google.com/file/d/1PgP7gY1FkcpQ1D6XW_lPzTYCgsMhItbw/view?usp=sharing).The URPC2020 training dataset is randomly divided into training and validation groups with 4,434 and 1,019 images, respectively.
## To reproduce the results in the paper
### Installation

Install AVE-UWSSOD from source.
```shell
https://github.com/Fly-fate/AVE-UWSSOD.git
pip3 install -v -e .  # or  python3 setup.py develop
```




### Download data
To ensure the fairness of the results we use the [STAC](https://github.com/google-research/ssl_detection/tree/master) paradigm for dataset partitioning.  

**step1.Download underwater data set `URPC2020`**
```shell
mkdir -p ${COCODIR}
cd ${COCODIR}
wget https://drive.usercontent.google.com/download?id=1PgP7gY1FkcpQ1D6XW_lPzTYCgsMhItbw&export=download&authuser=0&confirm=t&uuid=2f3b1cef-e48f-430d-a015-6ecba5466ebd&at=AENtkXY_kzd_2W_mpgmnLWiPMl4c%3A1731054381954
tar -xf URPC2020_detection.tar
```
**step2.Generate labeled and unlabeled splits**

- **format converter from pascal voc to coco.**
If you are using a VOC dataset instead of a COCO format dataset you can format it using the following command

```bash
cd ${PRJROOT}/datasets

python3 pascal_voc_xml2json.py --data_dir $VOCDIR

# resulting format
# ${VOCDIR}
#   - VOCdevkit
#       - Annotations
#       - JPEGImages

```
- **Generate labeled and unlabeled splits with different proportions of labeled data**
```
cd ${PRJROOT}/prepare_datasets

# Format:
#  labeled split - <datasetname>.<seed>@<percent_of_labeld>
#  unlabeled split - <datasetname>.<seed>@<percent_of_labeld>-unlabeled
for seed in 1 2 3 4 5; do
  for percent in 1 2 5 10 20; do
    python3 prepare_coco_data.py --percent $percent --seed $seed &
  done
done
```




### Training

The following example uses labeled data as 10% train2017 and rest 90% train2017 data as unlabeled data, with YOLOX-Nano as the baseline model.

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
