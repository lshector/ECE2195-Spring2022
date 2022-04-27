# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 17:10:23 2022

@author: Diego Wildenstein

This version of calculateMetrics compares the original uncompressed images to 
the output produced by the CNN-JPEG quantized accelerator.
"""
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as skpsnr


# Point origDir to the location of the uncompressed images. point compDir to 
# the location of the output from the CNN accelerator.
origDir = "./inputs_val/"
compDir = "./outputs_val/"

psnrs = []
ssims = []
CRs = []

count = 0
for file in os.listdir(origDir):
    # Read an image from the original directory.
    orig = cv2.imread( origDir+file )
    # now try to find a matching one in the comp directory. Since the 
    # accelerator adds some zeros to the name, we have to account for those.
    if count < 10:
        comp = cv2.imread( compDir+"00000"+str(count)+".jpg" )
    else:   
        comp = cv2.imread( compDir+"0000"+str(count)+".jpg" )

        
    # We have to fix cv2's weird color thing
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    comp = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)


    # If the two are not exactly the same size, we have to resize and 
    # interpolate one of them just so we can calculate SSIM and PSNR.
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
    # Same as before here, look for the extra zeros.
    if count < 10:
        compSize = os.path.getsize(compDir+"00000"+str(count)+".jpg")
    else:
        compSize = os.path.getsize(compDir+"0000"+str(count)+".jpg")

    
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

