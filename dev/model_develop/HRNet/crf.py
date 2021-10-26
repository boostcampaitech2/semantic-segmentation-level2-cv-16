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
df = pd.read_csv('/opt/ml/segmentation/baseline_code/submission/30_HRNetV2_OCR_W64_23epoch_rotate+randomsizecrop_0.647.csv')
test_path = '/opt/ml/segmentation/input/data/'

def crf(original_image, mask_img):
    
    labels = mask_img.flatten()

    n_labels = 11
    
    #Setting up the CRF model
    
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3,3), compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(10,10), srgb=(70,70,70), rgbim=original_image, compat=5)

    Q = d.inference(45)

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
df.to_csv('crf_correction.csv',index=False)