import copy
import os
import random
from turtle import color
import cv2
import numpy as np
from PIL import Image

from model import *

if __name__ == "__main__":
    #---------------------------------------------------#
    #   定义了输入图片的颜色，当我们想要去区分两类的时候
    #   我们定义了两个颜色，分别用于背景和斑马线
    #   [0,0,0], [0,255,0]代表了颜色的RGB色彩
    #---------------------------------------------------#
    class_colors = [(0,0,0),(200,0,0), (250,0,150), (200,150,150), (250,150,150), (0,200,0), (150,250,0), (150,200,150),
                    (200,0,200), (150,0,250), (150,150,250), (250,200,0), (200,200,0), (0,0,200), (0,150,200), (0,200,250)]
    #---------------------------------------------#
    #   定义输入图片的高和宽，以及种类数量
    #---------------------------------------------#
    
    #---------------------------------------------#
    #   背景 + 斑马线 = 2
    #---------------------------------------------#
    

    #---------------------------------------------#
    #   载入模型
    #---------------------------------------------#
    model = Resnet50_Unet()
    model.summary()
    #--------------------------------------------------#
    #   载入权重，训练好的权重会保存在logs文件夹里面
    #   我们需要将对应的权重载入
    #   修改model_path，将其对应我们训练好的权重即可
    #   下面只是一个示例
    #--------------------------------------------------#
    model_path = "logs/ep061-loss0.177-val_loss0.301.h5"
    model.load_weights(model_path)

    with open("D:/BaiduNetdiskDownload/GID/Fine Land-cover Classification_15classes/list/test.txt","r") as f:
        test_lines = f.readlines()
    num = len(test_lines)
    #--------------------------------------------------#
    #   对imgs文件夹进行一个遍历
    #--------------------------------------------------#
    
    for i in range(num):
        #--------------------------------------------------#
        #   打开imgs文件夹里面的每一个图片
        #--------------------------------------------------#
        name = test_lines[i].split()[0]
        img = Image.open("D:/BaiduNetdiskDownload/GID/Fine Land-cover Classification_15classes/img/{}.png".format(name))

        old_img = copy.deepcopy(img)
        orininal_h = np.array(img).shape[0]
        orininal_w = np.array(img).shape[1]

        #--------------------------------------------------#
        #   对输入进来的每一个图片进行Resize
        #   resize成[HEIGHT, WIDTH, 3]
        #--------------------------------------------------#
        img = img.resize((512,512), Image.BICUBIC)
        img = np.array(img)/255
        img = img.reshape(-1, 512, 512, 3)

        #--------------------------------------------------#
        #   将图像输入到网络当中进行预测
        #--------------------------------------------------#
        pr = model.predict(img)[0]
        pr = pr.reshape((512, 512, 16)).argmax(axis=-1)
        
        '''
        #------------------------------------------------#
        #   创建一副新图，并根据每个像素点的种类赋予颜色
        #------------------------------------------------#
        seg_img = np.zeros((512,512,3))
        for c in range(16):
            seg_img[:, :, 0] += ((pr[:,: ] == c) * class_colors[c][0]).astype('uint8')
            seg_img[:, :, 1] += ((pr[:,: ] == c) * class_colors[c][1]).astype('uint8')
            seg_img[:, :, 2] += ((pr[:,: ] == c) * class_colors[c][2]).astype('uint8')
        '''        
        seg_img = Image.fromarray(np.uint8(pr)).resize((orininal_w,orininal_h))
        seg_img.save("predict_15_gray/{}.png".format(name))
        print("save {}.png".format(name))
        #------------------------------------------------#
        #   将新图片和原图片混合
        #------------------------------------------------#
        #image = Image.blend(old_img,seg_img,0.3)
        
        #image.save("./img_out/"+jpg)


