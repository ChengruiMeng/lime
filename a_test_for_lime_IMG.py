import tensorflow as tf
slim = tf.contrib.slim
import sys

sys.path.append('.../lime/tf-models/slim')
#You can change that address to fit your pretrain-model location

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib
import skimage
from skimage import io,data,color,morphology,feature,transform
from skimage.segmentation import clear_border,mark_boundaries
from lime import lime_image
import time
from nets import inception
from preprocessing import inception_preprocessing
from datasets import imagenet

def loadDatadet(infile,k):
    f=open(infile,'r')
    sourceInLine=f.readlines()
    dataset=[]
    for line in sourceInLine:
        temp1=line.strip('\n')
        temp2=temp1.split(',')
        dataset.append(temp2)
    for i in range(0,len(dataset)):
        for j in range(k):
            dataset[i].append(int(dataset[i][j]))
        del(dataset[i][0:k])
    return dataset

session = tf.Session()
image_size = inception.inception_v3.default_image_size

def transform_img_fn(path_list):
        out = []
        for f in path_list:
            image_raw = tf.image.decode_jpeg(open(f).read(), channels=3)
            image = inception_preprocessing.preprocess_image(image_raw, image_size, image_size, is_training=False)
            out.append(image)
        return session.run([out])[0]

processed_images = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, _ = inception.inception_v3(processed_images, num_classes=1001, is_training=False)
probabilities = tf.nn.softmax(logits)
checkpoints_dir = '/Users/mengchengrui/Desktop/mytest/lime/tf-models/slim/pretrained/'
init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'inception_v3.ckpt'),
        slim.get_model_variables('InceptionV3'))
init_fn(session)

def predict_fn(images):
        return session.run(probabilities, feed_dict={processed_images: images})

images = transform_img_fn(['origin picture address'])
images2 = transform_img_fn(['target picture address'])
preds = predict_fn(images)
sysnet = np.argmax(preds)
print(sysnet)
preds2 = predict_fn(images2)
sysnet2 = np.argmax(preds2)
print(sysnet2)
diro = 'this experiment folder location'
image = images[0]
image2 = images2[0]
wo =[]
explainer = lime_image.LimeImageExplainer()
explanation= explainer.explain_instance_ot(image, image2, predict_fn, top_labels=1, hide_color=0, num_samples=1000)
# i_t for target percent,i_o for origin percent,this test show all situations that area-ratio below 30%(5% as step)
for i_t in range(0,7):
    for i_o in range(0,7):
        arg1 =5*i_t
        arg2 =5*i_o
        dirr = diro + 'tgt_{}_ori_{}/'.format(arg1,arg2)
        os.mkdir(dirr)
        image_remix_dir = dirr+'{}on{}_tgt_{}_ori_{}.npy'.format(sysnet,sysnet2,arg1,arg2)
        image_mask_dir = dirr+'{}on{}_tgt_{}_ori_{}_mask.npy'.format(sysnet,sysnet2,arg1,arg2)

        tgt_percent = 0.01 * arg1
        ori_percent = 0.01 * arg2
        print("limit is target: {} origin: {} ".format(tgt_percent,ori_percent))
        min_area,logit,iMin,jMin,image_remixMin,maskMin = explanation.find_common_hull(sysnet, sysnet2,predict_fn,tgt_percent,ori_percent)
        if min_area != None:
            wo.append([tgt_percent,ori_percent,min_area,logit])
        np.save(image_remix_dir, image_remixMin)
        np.save(image_mask_dir, maskMin)
        new_result=image_remixMin/2+0.5
        dirr2=dirr+'tgt_{}_ori_{}_strat.jpeg'.format(arg1,arg2)
        plt.imshow(new_result)
        plt.axis('off')
        plt.imsave(dirr2, new_result, format="jpeg")
        plt.clf()
        plt.close()

name=['目标图像面积限制','原始图像面积限制','总替换面积比','logit']
T_wo=pd.DataFrame(columns=name,data=wo)
T_wo.to_csv(diro+'348on181.csv',encoding='gbk')
