from model import *
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from tensorflow.keras.optimizers import Adam
from PIL import Image

def generate_arrays_from_file(lines,batch_size):
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            #-------------------------------------#
            #   读取输入图片并进行归一化和resize
            #-------------------------------------#
            name = lines[i].split()[0]
            img = Image.open("dataset/resize_image/{}.png".format(name))
            img = img.resize((512,512), Image.BICUBIC)
            img = np.array(img)/255
            X_train.append(img)

            #-------------------------------------#
            #   读取标签图片并进行归一化和resize
            #-------------------------------------#
            name = lines[i].split()[0]
            label = Image.open("dataset/resize_label_gray/{}.png".format(name))
            label = np.array(label)
            one_hot_label = np.eye(2)[label]
            Y_train.append(one_hot_label)

            i = (i+1) % n
        yield (np.array(X_train), np.array(Y_train))

if __name__ == "__main__":

    log_dir = "logs/"

    model = Resnet50_Unet()
    model.summary()
    
    #权重文件
    #weights_path = 'logs/ep036-loss0.202-val_loss0.224.h5'
    #model.load_weights(weights_path,by_name=True,skip_mismatch=True)

    # 打开数据集的txt
    with open("D:/BaiduNetdiskDownload/UDD5/list/train.txt","r") as f:
        train_lines = f.readlines()
    with open("D:/BaiduNetdiskDownload/UDD5/list/val.txt","r") as f:
        val_lines = f.readlines()

    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)

   

    if True:
        lr = 1e-4
        batch_size = 1
        model.compile(loss = 'categorical_crossentropy',
                optimizer = Adam(lr=lr),
                metrics = ['accuracy'])

        model.fit_generator(generate_arrays_from_file(train_lines, batch_size),
                steps_per_epoch=max(1, len(train_lines)//batch_size),
                validation_data=generate_arrays_from_file(val_lines, batch_size),
                validation_steps=max(1, len(val_lines)//batch_size),
                epochs=500,
                initial_epoch=0,
                callbacks=[checkpoint, reduce_lr,early_stopping])
        