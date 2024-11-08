# AVE-UWSSOD：Adaptive Visual Enhancement and Semi-Supervised Learning for Robust Underwater Object Detection
AVE-UWSSOD is a network model focused on underwater target detection that combines visual enhancement and a semi-supervised object detection framework based on co-training image preprocessing methods and high-quality pseudo-labeling to achieve robust underwater object detection.
![AVE-UWSSOD Image](https://github.com/Fly-fate/AVE-UWSSOD/blob/master/docs/AVE-SSOD.png)
![AVE-UWSSOD workflows](https://github.com/Fly-fate/AVE-UWSSOD/blob/master/docs/workflows.png)

## To reproduce the results in the paper
<details>
<summary>Installation</summary>

Step1. Install YOLOX from source.
```shell
git clone git@github.com:Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip3 install -v -e .  # or  python3 setup.py develop
```

</details>
