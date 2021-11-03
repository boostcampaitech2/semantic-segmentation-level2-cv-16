import numpy as np
import pydensecrf.densecrf as dcrf
from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
from skimage.color import gray2rgb
from skimage.color import rgb2gray
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.util import img_as_ubyte


import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def decode(rle_mask):
    mask = rle_mask.split()
    img = np.zeros(256*256, dtype=np.uint8)
    for i, m, in enumerate(mask):
        img[i] = int(m)
    return img.reshape(256,256)


def encode(im):
    pixels = im.flatten()
    return ' '.join(str(x) for x in pixels)

"""
reading and decoding the submission 

"""
df = pd.read_csv('/opt/ml/segmentation/baseline_code/submission/ensem_0.794.csv')
test_path = '/opt/ml/segmentation/input/data/'

'''
#Deeplab
    ITER_MAX= 10
    POS_W= 3
    POS_XY_STD= 1
    BI_W= 4
    BI_XY_STD= 67
    BI_RGB_STD= 3
#Deeplab Large
    ITER_MAX= 10
    pos_w = 3
    pos_xy_std = 3
    bi_w = 4
    bi_xy_std = 121
    bi_rgb_std = 5
#PSPNET
    POS_W = 3
    POS_XY_STD = 3
    Bi_W = 4
    Bi_XY_STD = 49
    Bi_RGB_STD = 5
#PydenseCRF
    POS_W = 3
    POS_XY_STD = 3
    Bi_W = 10
    Bi_XY_STD = 80
    Bi_RGB_STD = 13
https://github.com/lucasb-eyer/pydensecrf
d.addPairwiseGaussian(sxy=3, compat=3)
d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=im, compat=10)

'''


def crf(original_image, mask_img):
    MAX_ITER = 30
    POS_W = 3
    POS_XY_STD = 3
    Bi_W = 4
    Bi_XY_STD = 49
    Bi_RGB_STD = 5
    labels = mask_img.flatten()

    n_labels = 11
    img = np.ascontiguousarray(original_image)
    
    #Setting up the CRF model
    
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=img, compat=Bi_W)

    Q = d.inference(MAX_ITER)


    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    return MAP.reshape((original_image.shape[0],original_image.shape[1]))

for i in tqdm(range(df.shape[0])):
    if str(df.loc[i,'PredictionString'])!=str(np.nan):        
        decoded_mask = decode(df.loc[i,'PredictionString'])        
        orig_img = imread(test_path+df.loc[i,'image_id'])
        image_resized = resize(orig_img, (orig_img.shape[0] // 2, orig_img.shape[1] // 2), anti_aliasing=True) 
        crf_output = crf(img_as_ubyte(image_resized),decoded_mask)
        df.loc[i,'PredictionString'] = encode(crf_output)
df.to_csv('ensem_crf_iter_30_PSP_setting.csv',index=False)