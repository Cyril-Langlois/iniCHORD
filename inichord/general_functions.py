# -*- coding: utf-8 -*-
"""
Fonctions générales
"""
import os
import h5py
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog
# import cupy as cp
import time
from skimage.morphology import disk, ball
import scipy.ndimage as sci
from skimage.restoration import denoise_nl_means
from skimage import img_as_float
import cv2

import sklearn.preprocessing as pp
import matplotlib.pyplot as plt
import bm3d
from pyvsnr import vsnr2d

import tifffile as tf

#______________________________________________________
# h5py functions
def get_dataset_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys

def get_group_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Group) else None)
    return keys

def openH5file(text = 'theoretical database (*.crddb)'):

    '''
    Open a dialog to look for the H5 file to open
    
    Returns (in this order):
        - H5file itself, to be used to get datasets
        - directory of the file
        - file path
        - the list of keys in the H5 file
    '''
   
    filePath, dirFile = getFilePathDialog(text)

    # lecture du fichier de profils theoriques
    f = h5py.File(filePath[0], 'r')

    return f, dirFile, filePath[0], get_dataset_keys(f)

def Saving_img_or_stack(stack):
    """
    Save an image or a stack of images in tif format,
    in its original type (uint8 or float32...)

    Parameters
    ----------
    stack : numpy array 2D or 3D or color image
        DESCRIPTION.

    Returns nothing
    -------
    None.

    """
    
    SavePath = filedialog.asksaveasfilename(title="Save the aligned stack", filetypes=[("TIFF Files", "*.tif")])

    tf.imwrite(SavePath, stack)    
    
def savingMTEXfromHDF5():
    # Ouverture du fichier h5
    
    f, dirFile, filePath, listKeys = openH5file()
    
    
    for i in listKeys:
        if "nScoresOri" in i:
            nScoresOri = np.asarray(f[i])
            
    Quat=nScoresOri[-1,:,:,:]
    x = len(Quat[0])
    y = len(Quat[0][0])

    ti = time.strftime("%Y-%m-%d__%Hh-%Mm")

    with open(dirFile + '\indexGPU_'+ ti + '.quatCHORDv3-CTFxyConv.txt', 'w') as file:
    
        for i in range(x):
            for j in range(y):

                index = 1    
                if Quat[0,i,j] ==0 :
                    index = 0
                file.write(str(index) + '\t' + str(j) + '\t' + str(i) + '\t' + str(Quat[0, i, j]) +
                '\t' + str(Quat[1, i, j])  + '\t' + str(Quat[2, i, j])  + '\t' + str(Quat[3, i, j]) + '\n') 

#________________________________________________________
# type conversion

def scale_to(x, x_min, x_max, t_min, t_max): # Conversion en 16bits
    """
    Scales x to lie between t_min and t_max
    Links:
         https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
         https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
    """
    r = x_max - x_min
    r_t = t_max - t_min
    assert(math.isclose(0,r, abs_tol=np.finfo(float).eps) == False)
    x_s = r_t * (x - x_min) / r + t_min
    return x_s

#________________________________________________________
# GPU function with cupy

def isCupy(a):
    '''
    Parameters
    ----------
    a : any python object
        the object which type to be determined, cupy or not cupy.

    Returns
    -------
    Boolean
        True if the object in parameter is a cupy object, False for any other type.

    '''
    return str(type(a)) == "<class 'cupy.ndarray'>"

#________________________________________________________
# general functions

def getFilePathDialog(text):
    '''
    
    Parameters
    ----------
    text : Ttype 'string'
        the text to be displayed to describe what type of file should be opened by the dialog

    Returns
    -------
    filePath
        the full path to the selected file

    '''
    window = tk.Tk()
    window.wm_attributes('-topmost', 1)
    window.withdraw()

    # Pour utiliser le programme de traitement stack
    filePath = filedialog.askopenfilenames(title=text)
    dirFile = os.path.dirname(filePath[0])
    
    # Pour indexation ces lignes fonctionne (jusque V16 d'indexation sur GPU)
    # filePath = filedialog.askopenfilename(title=text, multiple=True, parent=window)[0]
    # dirFile = os.path.dirname(filePath)
    
    return filePath, dirFile


#______________________________
# image treatment

def remove_outliers(image, radius=2, threshold=50):
    footprint_function = disk if image.ndim == 2 else ball
    footprint = footprint_function(radius=radius)
    median_filtered = sci.median_filter(image, footprint=footprint)
    outliers = (
        (image > median_filtered + threshold)
        | (image < median_filtered - threshold)
    )
    output = np.where(outliers, median_filtered, image)
    return outliers, output

def NonLocalMeanDenoising(image, param_h = 0.06, fastAlgo = True, size = 5, distance = 6):

    denoisedImage = denoise_nl_means(image, h= param_h, fast_mode = True,
                                     patch_size = size,
                                     patch_distance = distance)
    return denoisedImage

def BM3D(image, psd, isFast):
    if isFast:
        denoisedImage = bm3d.bm3d(image, sigma_psd = psd, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    else:
        denoisedImage = bm3d.bm3d(image, sigma_psd = psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)

    return denoisedImage

def VSNR_funct(image, filter_):
    
    denoisedImage = vsnr2d(image, filter_)
    
    return denoisedImage

def convertToUint8(stack):
      mn = stack.min()
      mx = stack.max()
    
      mx -= mn
    
      stack = ((stack - mn)/mx) * 255.0
      return np.round(stack).astype(np.uint8)
  
def convertToUfloat32(stack):
      mn = stack.min()
      mx = stack.max()
    
      mx -= mn
    
      stack = ((stack - mn)/mx) * 2_000_000_000
      return np.round(stack).astype(np.float32)
   
def sliceinfo(image):
    mn = image.min()
    mx = image.max()
    print(f"\n _________________\nle type de donnée est {type(mn)} \n le min est {mn} \n et le max est {mx}")

def bleach_ratio(stack):
    Mean_intensity = np.zeros(len(stack))
    Ratio = np.zeros(len(stack))
    Bleached_stack = np.zeros((len(stack),len(stack[0]),len(stack[0][0])))

    for i in range(0,len(stack)):
        Mean_intensity[i] = np.mean(stack[i,:,:])
        Ratio[i] = Mean_intensity[i]/Mean_intensity[0]   
        
        if Ratio[i]<= 1 :  
            Bleached_stack[i,:,:] = stack[i,:,:]*Ratio[i]
        else : 
            Bleached_stack[i,:,:] = stack[i,:,:]/Ratio[i]
    
    return Bleached_stack

def cropped_img_or_stack(data,Pj1,Pi1,Pj2,Pi2):
    if len(np.shape(data)) == 3:
        Cropped_data = data[:,Pi1:Pi2,Pj1:Pj2]
    else :
        Cropped_data = data[Pi1:Pi2,Pj1:Pj2]
    
    return Cropped_data

#______________________________
# Denoising characterization

def Pixels_transformation(arr1, arr2): #Permet de connaitre le nombre de pixel ayant été modifiés
    diff = arr2-arr1 #Différences des deux cartes
    
    Unmodified_pixels_number = np.count_nonzero(diff) # Nombre de pixels n'ayant pas été modifiés
    Total_pixels_number = len(diff)*len(diff[0]) # Nombre total de pixels
    Modified_pixels_proportion = Unmodified_pixels_number * 100 / Total_pixels_number
    
    Var = diff
    Modified_pixels_intensity = Var
    del(Var)
    
    Modified_pixels_intensity[Modified_pixels_intensity == 0] = np.nan
    mean_modified_pixels_intensity = np.nanmean(abs(Modified_pixels_intensity), axis=None)
    
    max_intensity_arr = np.max(arr1)
    mean_proportion_intensity = mean_modified_pixels_intensity * 100 / max_intensity_arr
    
    return Modified_pixels_proportion, mean_modified_pixels_intensity, mean_proportion_intensity

# Resize_stack : pour réduire le nombre de pixels d'une image 
def Binning_stack(Raw_Stack, bin_value = 2, inter = cv2.INTER_CUBIC):
    """
    

    Parameters
    ----------
    Raw_Stack : numpy array
        DESCRIPTION: profiles in dimension, then image height in dimension 1 and image width in dimension 2
    bin_value : TYPE, optional
        DESCRIPTION. The default is 2.
    inter : TYPE, optional
        DESCRIPTION. Type of binning interpolation. The default is cv2.INTER_CUBIC.

    Returns
    -------
    Resize_stk : numpy array
        DESCRIPTION: 

    """
    bin_value = bin_value
    X_binned = int(np.round((len(Raw_Stack[0][0])/bin_value)))
    Y_binned = int(np.round((len(Raw_Stack[0])/bin_value)))
    
    Resize_stk = []              
            
    for i in range(0,len(Raw_Stack)):
        Var = Raw_Stack[i,:,:]
        Resized_img = cv2.resize(Var, dsize=(X_binned, Y_binned), interpolation=inter)
        Resize_stk.append(Resized_img)
    
    Resize_stk = np.dstack(Resize_stk)
    Resize_stk = np.swapaxes(Resize_stk, 2, 0)
    Resize_stk = np.rot90(Resize_stk, k=1, axes=(1, 2))
    Resize_stk = np.fliplr(Resize_stk)
    
    return Resize_stk

#_______________________________
# Grain boundaries highlightment 

def canny_edge_determination(img, threshold1, threshold2): # Permet d'obtenir les contours pour une image
    
    img_blur = cv2.GaussianBlur(img, (3,3), 0) # Blur the image for better edge detection
    edge = cv2.Canny(image=img_blur, threshold1=threshold1, threshold2=threshold2) # Canny Edge Detection
    
    return edge

def Sum_canny_edge_determination(stack, threshold1, threshold2): #Permet d'obtenir les contour d'une stack
    edges = np.zeros((len(stack),len(stack[0]),len(stack[0][0])))
    edges_stack = convertToUint8(stack)

    # Determination des contours sur chaque image 
    for i in range(0,len(edges_stack)):
        edges[i,:,:] = canny_edge_determination(edges_stack[i,:,:],threshold1,threshold2) #Stack, threshold 1, threshold 2
    
    # Sommation des contours de chaque slices
    Sum_edges = edges == 255
    Sum_edges = np.sum(Sum_edges,0)
    
    return edges, Sum_edges

def Sum_canny_edge_proportion_determination(Sum_edges, stack, seuil_JDG): #Permet de resortir un filtre de JDG et un taux de pixels appartement à un JDG

    # Seuillage de "Sum_edges" et calcul du taux de l'image étant une interface
    seuil_JDG = seuil_JDG
    thresh_sum_edges = Sum_edges > seuil_JDG
    edges_proportion = np.round(thresh_sum_edges.sum() * 100 / (len(stack[0])*len(stack[0][0])))

    return thresh_sum_edges, edges_proportion