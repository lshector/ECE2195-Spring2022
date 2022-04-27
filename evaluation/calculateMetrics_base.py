# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 17:10:23 2022

@author: Diego Wildenstein

This is the base calculateMetrics code. It can be used to compare the quality 
of images with similar names from two different directories.
"""
import sys
import os
import cv2
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as skpsnr
#originalImg = cv2.imread(str(sys.argv[1]))
#compressedImg = cv2.imread(str(sys.argv[2]))

#origDir = "./Validation/"
#compDir = "./Compressed/"

compDir = "./serial_val/"
origDir = "./outputs_val/"


psnrs = []
ssims = []
CRs = []

count = 0
for file in os.listdir(origDir):
    # Read an image from the original directory.
    orig = cv2.imread( origDir+file )
    # now try to find a matching one in the comp directory. Ideally, it should 
    # have a name like "12.png"
    comp = cv2.imread( compDir+str(count)+".png" )
    #comp = cv2.imread( compDir+"0000"+str(count)+".jpg" )

    # We have to fix cv2's weird color thing
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    comp = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)


    
    # If the two are not exactly the same size, we have to resize and 
    # interpolate one of them.
    if comp.size != orig.size:
        newsize = orig.shape
        comp = cv2.resize(comp, dsize=(newsize[1], newsize[0]), \
                          interpolation=cv2.INTER_CUBIC )
        cv2.imwrite("rescaled_D2.jpg", comp)
    

    # Calculate the PSNR between the two images
    """
    meanSqErr = np.mean((orig - comp) ** 2)
    if meanSqErr == 0:
        psnr = 100
    else:
        psnr = 20 * math.log10(255 / math.sqrt(meanSqErr))
    """
    psnr = skpsnr(orig, comp)
    psnrs.append(psnr)
    #print(psnr)


    # calculate the structural similarity index (SSIM) between the images.
    """
    uX = np.average(orig)
    uY = np.average(comp)
    
    varX = np.var(orig)
    varY = np.var(comp)
    covXY = np.cov(orig, comp)
    # constants, L is 255 for uint8 images
    k1 = 0.01
    k2 = 0.03
    L = 255
    
    c1 = (k1*L)^2
    c2 = (k2*L)^2
    ssim = ( (2*uX*uY+c1)*(2*covXY+c2) ) / ( (uX^2+uY^2+c1)*(varX+varY+c2) )
    """
    similarity = ssim( orig, comp, channel_axis=2)
    ssims.append(similarity)


    # Calculate the compression ratio
    origSize = os.path.getsize(origDir+file)
    compSize = os.path.getsize(compDir+str(count)+".png")
    cRatio = origSize / compSize
    CRs.append(cRatio)
    #print("Compression ratio: ", cRatio)


    count = count + 1


# Calculate the average metrics and display them.
avgPSNR = np.average(psnrs)
avgSSIM = np.average(ssims)
avgCR = np.average(CRs)
print("Average PSNR: ", avgPSNR)
print("Average SSIM: ", avgSSIM)
print("Average CR: ", avgCR)

