from curses import keyname
import torch
#import json

dict = torch.load("/home/21131213348/isp_yolox/YOLOX_peleenet_1/weights/peleenet.pth")
print(dict.keys())

model = dict["model"]
#print(model.keys())
for key in model.keys():         
    if "head.cls_preds" in key:
        print(key)

#weight_txt=open("/home/zhouyue/yolox/YOLOX/weights/w.json","w")
#weight_txt.write(json.dumps(model_dict))