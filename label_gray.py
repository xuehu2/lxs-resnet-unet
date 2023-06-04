import os
from PIL import Image



path = "D:/DeepLearning/Model/Resnet50_Unet_xuehu/dataset"
image_path = os.path.join(path, "label_gray")

temp_seg = os.listdir(image_path)

total_seg = []
for seg in temp_seg:
    if seg.endswith(".png"):
        total_seg.append(seg)

for seg in total_seg:
    image = Image.open(os.path.join(image_path,seg))
    print(image.mode)
    # max_temp = max(image.size)
    mask = Image.new('L', (2592, 2592),0)  #创建一个最大的掩码图2592*2592
    mask.paste(image, (0, 0))
    mask.save(os.path.join(path,"resize_label_gray", seg))

