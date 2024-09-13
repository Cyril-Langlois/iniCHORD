# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 23:26:20 2023

@author: clanglois1
"""
import os

from inspect import getsourcefile
from os.path import abspath

import numpy as np
import tifffile as tf
import time
import cv2
import largestinteriorrectangle as lir

import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QPixmap

from . import general_functions as gf

from skimage import exposure
from scipy import ndimage as ndi

path2thisFile = abspath(getsourcefile(lambda:0))
uiclass, baseclass = pg.Qt.loadUiType(os.path.dirname(path2thisFile) + "/Stitching_v4.ui")

class MainWindow(uiclass, baseclass):
    def __init__(self,parent):
        super().__init__()
        
        self.setWindowIcon(QtGui.QIcon('icons/Stitch_icon.png'))
        
        self.setupUi(self)
        self.parent = parent

        self.x = 0
        self.y = 0
    
        self.defaultIV() # Hide the PlotWidget until a data has been loaded
        
        # Buttons disabled until first importation
        self.KADBox.setEnabled(False)
        self.col_spinBox.setEnabled(False)
        self.row_spinBox.setEnabled(False)
        self.Range_Edit.setEnabled(False)
        self.Tresh_spinBox.setEnabled(False)
        self.full_range.setEnabled(False)
        self.ComputeClass_bttn.setEnabled(False)
        self.Push_export.setEnabled(False)
        self.Save_bttn.setEnabled(False)
        
        self.OpenData.clicked.connect(self.loaddata) # Open the KAD maps
        self.ComputeClass_bttn.clicked.connect(self.Stitching) # Run the stitching program
        self.Push_export.clicked.connect(self.export_data)
        self.Save_bttn.clicked.connect(self.Save_results) # Save maps and informations
        
        self.KADBox.valueChanged.connect(self.Change_KAD_display)
        self.col_spinBox.valueChanged.connect(self.set_up)
        self.row_spinBox.valueChanged.connect(self.set_up)
        self.flag_KAD = False
        self.flag_affine = True # Affine transformation used as default 
        
        self.col_nbr = self.col_spinBox.value()
        self.row_nbr = self.row_spinBox.value()
        
        self.val_tresh = self.Tresh_spinBox.value()
        self.Tresh_spinBox.valueChanged.connect(self.Change_treshold)
        
        self.Range_Edit.setText("300")
        self.changeRange() # Modify the searching range 
        self.Range_Edit.editingFinished.connect(self.changeRange) # Modify the searching range 
        
        app = QApplication.instance()
        screen = app.screenAt(self.pos())
        geometry = screen.availableGeometry()
        
        # Control window position and dimensions
        self.move(int(geometry.width() * 0.05), int(geometry.height() * 0.05))
        self.resize(int(geometry.width() * 0.9), int(geometry.height() * 0.6))
        self.screen = screen
        
        # Icons sizes management for QMessageBox
        self.pixmap = QPixmap("icons/Stitch_icon.png")
        self.pixmap = self.pixmap.scaled(100, 100)

#%% Functions
    def data_choice(self): # Allow to apply other treatment depending if the data is a KAD one
        msg = QMessageBox.question(self, 'Image stitching', 'Is it a KAD data ?')
        if msg == QMessageBox.Yes:
            self.flag_KAD = True
        if msg == QMessageBox.No:
            self.flag_KAD = False

    def loaddata(self): # Must be a sequence of 2D array
        self.StackLoc, self.StackDir = gf.getFilePathDialog("data to be stitched (*.tiff)") 
        
        checkimage = tf.TiffFile(self.StackLoc[0]).asarray() # Check for dimension. If 2 dimensions : 2D array. If 3 dimensions : stack of images
        
        if checkimage.ndim != 2: # Check if the data is not a 2D array
            self.parent.popup_message("2D stitching","Please import a sequence of 2D arrays",'icons/Stitch_icon.png')
            return

        else:
            self.data_choice()
    
            self.KAD_base = []
            
            for i in range(0,len(self.StackLoc)): # Create a list of 2D array 
                Var = tf.TiffFile(self.StackLoc[i]).asarray()
                self.KAD_base.append(Var)
                
            for i in range(len(self.KAD_base)): 
                mask = np.isnan(self.KAD_base[i])
                self.KAD_base[i][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), self.KAD_base[i][~mask]) # Allow to convert NaN value with the nearest good value
                
                if self.flag_KAD == True: # Normalization and CLAHE application of every array
                    self.KAD_base[i] = (self.KAD_base[i] - np.min(self.KAD_base[i])) / (np.max(self.KAD_base[i]) - np.min(self.KAD_base[i])) # Normalization step
                    self.KAD_base[i] = exposure.equalize_adapthist(self.KAD_base[i], kernel_size=None, clip_limit=0.01, nbins=256) # CLAHE step
    
                if self.KAD_base[i].dtype != 'uint8': # If data is not a 8bits one, then the conversion is apply
                    self.KAD_base[i] = gf.convertToUint8(self.KAD_base[i])
                
                Search0 = np.where(self.KAD_base[i] == 0.0) # Replace 0.0 by 1 for final crop efficiency
                self.KAD_base[i][Search0] = 1
                
            self.KAD_base_display = self.KAD_base.copy() 
            
            for i in range(len(self.KAD_base_display)): # Flip and rotation for display
                self.KAD_base_display[i] = np.flip(self.KAD_base_display[i], 0)
                self.KAD_base_display[i] = np.rot90(self.KAD_base_display[i], k=1, axes=(1, 0))
            
            self.displayExpKAD(self.KAD_base_display[0])
            self.InfoValues_label.setText(str(self.KAD_base_display[0].shape))
    
            # Buttons disabled until first importation
            self.KADBox.setEnabled(True)
            self.col_spinBox.setEnabled(True)
            self.row_spinBox.setEnabled(True)
            self.Range_Edit.setEnabled(True)
            self.Tresh_spinBox.setEnabled(True)
            self.full_range.setEnabled(True)
            self.ComputeClass_bttn.setEnabled(True)
        
    def set_up(self):
        # Here, nbr of column and nbr of raw are defined. self.Super_listing is create.
        # it allow to work row by row for the next computation step
        self.col_nbr = self.col_spinBox.value()
        self.row_nbr = self.row_spinBox.value()

        self.Super_listing = []
        step = 0

        for i in range(0,self.row_nbr):
            self.Super_listing.append(self.KAD_base[step:self.col_nbr + step])
            
            step = step + self.col_nbr
            
        self.Super_listing = [x for x in self.Super_listing if x != []]      
    
    def Change_KAD_display(self): # Visualization of the different 2D array 
        self.KADBox.setRange(0,len(self.KAD_base)-1)
        self.KADBox.setSingleStep(1)
        self.Value = self.KADBox.value()
        
        self.displayExpKAD(self.KAD_base_display[self.Value])
        self.InfoValues_label.setText(str(self.KAD_base_display[self.Value].shape))

    def make_transfo_choice(self): # Choice of the transformation to be used to stitch images together
        if self.transfo_choice == 'Translation':
            self.flag_translation = True
            self.flag_affine = False
            self.flag_homo = False
        
        elif self.transfo_choice == 'Affine':
            self.flag_translation = False
            self.flag_affine = True
            self.flag_homo = False
            
        elif self.transfo_choice == 'Homography':
            self.flag_translation = False
            self.flag_affine = False
            self.flag_homo = True
            
    def Stitching(self): # Stitch images together
        self.transfo_choice = self.Transfo_box.currentText() # Extract transformation choice
        self.make_transfo_choice() # Define which transformation must be used
        
        self.nbr = self.col_nbr * self.row_nbr # Nbr of total 2D array
        
        self.prgbar = 0 # Progress bar initial value
        self.progressBar.setValue(self.prgbar)
        self.progressBar.setRange(0, self.nbr-1)
        self.increment = 0

        try:
            self.set_up()
            self.res = []

            if self.col_nbr > 1: # If there are more than 1 column
                for i in range(0,len(self.Super_listing)):
                    for j in range(len(self.Super_listing[0])-1,0,-1):
                        # Find matches between images 
                        M = self.find_matches(self.Super_listing[i][j],self.Super_listing[i][j-1],direction = "horizontal")
                        # Apply transformation matrix M
                        dst = self.horizontal_stitching(self.Super_listing[i][j],self.Super_listing[i][j-1],M)
                        # Stitch the two images together
                        self.Super_listing[i][j] = self.Super_listing[i][0:j]
                        self.Super_listing[i][j-1] = dst
                        
                        QApplication.processEvents()    
                        self.increment = self.increment + 1
                        self.ValSlice = self.increment
                        self.progression_bar()
                        
                        # j is modified because the lenght of Super listing is evolving at each step
                        # j = len(self.Super_listing[i])+1
                        j = len(self.Super_listing[i])-1
                        
                        # The twoD stitching is cropped
                        self.cropped_dst = self.cropping_step(dst)
                        
                    # The twoD stitching is stored to be used after 
                    self.res.append(self.cropped_dst)
                    
            else : # If there is only 1 column, self.res is a copy of self.KAD_base 
                self.res = self.KAD_base.copy()
        
            if self.row_nbr > 1: # If the nbr of row is higher than 1
                for i in range(len(self.res)-1,0,-1):
                    # Find matches between images 
                    M = self.find_matches(self.res[i],self.res[i-1],direction = "vertical")
                    # Apply transformation matrix M
                    dst = self.vertical_stitching(self.res[i],self.res[i-1],M)
                    # Stitch the two images together
                    self.res[i] = self.res[0:i]
                    self.res[i-1] = dst
                    
                    QApplication.processEvents()    
                    self.increment = self.increment + 1
                    self.ValSlice = self.increment
                    self.progression_bar()
                    
                    # i is modified because the lenght of self.res is evolving at each step
                    i = len(self.res[i])-1
                    
                    # The twoD stitching is cropped
                    self.cropped_dst = self.cropping_step(dst)
            
            # Apply flip and rotation for display of the stitched 2D arrays
            self.displayed_dst = np.copy(self.cropped_dst)
            self.displayed_dst = np.flip(self.displayed_dst, 0)
            self.displayed_dst = np.rot90(self.displayed_dst, k=1, axes=(1, 0))
                    
            self.displayStitch(self.displayed_dst)
            
            self.Push_export.setEnabled(True) # Allow to export to main GUI
            self.Save_bttn.setEnabled(True) # Allow to save data in a folder
            
        except:
            self.parent.popup_message("2D stitching","Stitch failed. Please check that the [columns - rows] information match the imported data.",'icons/Stitch_icon.png')
            return

    def find_matches(self,data1, data2, direction = "horizontal"):
        sift = cv2.SIFT_create()
        
        # Allow to specifiy if the descriptor determination must be apply on the whole images or only a part of it
        if self.full_range.isChecked():
            self.val_range = len(data2[0])
        else :
            self.val_range = self.val_range

        # create a mask image filled with zeros, the size of original image
        mask = np.zeros(data1.shape[:2], dtype=np.uint8)
        mask2 = np.zeros(data2.shape[:2], dtype=np.uint8)
        
        # Allow to specifiy the range of search to consider the fact that the images are vertical or horizontal
        if direction == 'horizontal':
            cv2.rectangle(mask, (0,0), (self.val_range,len(data1)), (255), thickness = -1)
            cv2.rectangle(mask2, (len(data2[0])-self.val_range,0), (len(data2[0]),len(data2)), (255), thickness = -1)
        if direction == 'vertical':
            cv2.rectangle(mask, (0,0), (len(data1[0]),self.val_range), (255), thickness = -1)
            cv2.rectangle(mask2, (len(data2[0]),len(data2)), (0,len(data2)-self.val_range), (255), thickness = -1)
    
        # Extract descriptor in the two images
        kp1, des1 = sift.detectAndCompute(data1,mask)
        kp2, des2 = sift.detectAndCompute(data2,mask2)
        
        # Look for matches
        match = cv2.BFMatcher()
        matches = match.knnMatch(des1,des2,k=2)
        
        # Matches filtering in order to keep only the best pairs
        good = []
        for m,n in matches:
            if m.distance < self.val_tresh*n.distance:
                good.append(m)
        
        MIN_MATCH_COUNT = 10 # Minimum nbr of descriptor needed to compute
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
            # Look for transformaiton matrix between the two set of descriptors
            if self.flag_translation == True : 
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) #Homography
                
                M[1:3,0] = 0
                M[2,1] = 0
                M[0,1] = 0
                
            elif self.flag_affine == True : 
                M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts) #Affine
            elif self.flag_homo == True : 
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) #Homography
        
        else:
            self.parent.popup_message("2D stitching","Not enough descriptor...",'icons/Stitch_icon.png')
            return
            
        return M # Return the matrix of transformation between the 2 images

    def horizontal_stitching(self,data1,data2,M):
        # Specify the initial shape of dst as a function of the transformation (translation/affine: warpAffine ; else: warpPerspective)
        if self.flag_affine == True : 
            dst = cv2.warpAffine(data1,M,(data2.shape[1] + data1.shape[1], data2.shape[0]))
        else :
            dst = cv2.warpPerspective(data1,M,(data2.shape[1] + data1.shape[1], data2.shape[0]))
        
        dst[0:data2.shape[0],0:data2.shape[1]] = data2
        
        return dst
    
    def vertical_stitching(self,data1,data2,M):   
        # Specify the initial shape of dst as a function of the transformation (translation/affine: warpAffine ; else: warpPerspective)
        if self.flag_affine == True : 
            dst = cv2.warpAffine(data1,M,(data2.shape[1] + data1.shape[1], data2.shape[0] + data1.shape[0]))
        else:
            dst = cv2.warpPerspective(data1,M,(data2.shape[1] + data1.shape[1], data2.shape[0] + data1.shape[0]))

        dst[0:data2.shape[0],0:data2.shape[1]] = data2
        
        return dst
        
    def cropping_step(self,dst): # Crop images to remove black border 
        Min_Aligned_stack = self.Mask_min(dst,0)  
        Min_Aligned_stack = Min_Aligned_stack.astype("bool")
        Min_Aligned_stack = ndi.binary_fill_holes(Min_Aligned_stack).astype("bool")
    
        Selection = lir.lir(Min_Aligned_stack) # array([2, 2, 4, 7])
        Cropped_dst = dst[Selection[1]:Selection[3],Selection[0]:Selection[2]]  
        
        return Cropped_dst
    
    def Mask_min(self,Min_Aligned_stack, threshold):
        Mask = np.zeros((len(Min_Aligned_stack),len(Min_Aligned_stack[0])))
        Mask[Min_Aligned_stack > threshold] = 1
        return Mask
    
    def changeRange(self): # Take into account the searching range modification
        self.val_range = int(self.Range_Edit.text())
        self.full_range.setChecked(False)
        
    def Change_treshold(self): # Take into account the descriptor filtering threshold
        self.val_tresh = self.Tresh_spinBox.value()
    
    def progression_bar(self): # Fonction relative à la barre de progression
        self.prgbar = self.ValSlice
        self.progressBar.setValue(self.prgbar)

    def Save_results(self):
        ti = time.strftime("%Y-%m-%d__%Hh-%Mm-%Ss") # Absolute time 
        
        directory = "Stitched_images_" + ti # Name of the main folder
        PathDir = os.path.join(self.StackDir, directory)  # where to create the main folder
        os.mkdir(PathDir)  # Create main folder

        # Images saving step
        tf.imwrite(PathDir + '/Cropped_stitched_img.tiff', np.rot90(np.flip(self.displayed_dst, 0), k=1, axes=(1, 0)).astype('float32')) 

        # Information (.TXT) step
        with open(PathDir + '\Image stitching.txt', 'w') as file:
            file.write("KAD data: " + str(self.flag_KAD))
            file.write("\nNumber of images: " + str(len(self.KAD_base)))
            file.write("\nNumber of row: " + str(self.row_nbr))   
            file.write("\nNumber of column: " + str(self.col_nbr))  
            if self.full_range.isChecked():
                file.write("\nSearching range (pxls): full images")   
            else:
                file.write("\nSearching range (pxls): "+ str(self.val_range))  
            file.write("\nTransformation: " + str(self.val_tresh)) 
            file.write("\nThreshold: " + str(self.transfo_choice)) 

        # Finished message
        self.parent.popup_message("2D stitching","Saving process is over.",'icons/Stitch_icon.png')

    def export_data(self): # Push stitched image in the main GUI
        self.parent.KAD = np.copy(self.displayed_dst) # Copy in the main GUI
        self.parent.StackList.append(self.displayed_dst) # Add the data in the stack list
        
        self.parent.StackDir = self.StackDir
        
        Combo_text = '\u2022 Stitched KAD'
        Combo_data = self.displayed_dst
        self.parent.choiceBox.addItem(Combo_text, Combo_data) # Add the data in the QComboBox

        self.parent.displayDataview(self.parent.KAD) # Display the labeled grain
        self.parent.choiceBox.setCurrentIndex(self.parent.choiceBox.count() - 1) # Show the last data in the choiceBox QComboBox

        self.parent.Info_box.ensureCursorVisible()
        self.parent.Info_box.insertPlainText("\n \u2022 Stitched KAD map.") 
             
        self.parent.Save_button.setEnabled(True)
        self.parent.Reload_button.setEnabled(True)
        self.parent.choiceBox.setEnabled(True)
        
        # Finished message
        self.parent.popup_message("2D stitching","Stitched KAD has been exported to the main GUI.",'icons/Stitch_icon.png')
    
    def displayExpKAD(self, series): # Display of initial KAD maps
        self.KADSeries.ui.histogram.hide()
        self.KADSeries.ui.roiBtn.hide()
        self.KADSeries.ui.menuBtn.hide()
        
        view = self.KADSeries.getView()
        state = view.getState()        
        self.KADSeries.setImage(series) 
        view.setState(state)
        view.setBackgroundColor(self.parent.color1)
        
        self.KADSeries.autoRange()
        
    def displayStitch(self, series): # Display of initial KAD map
        self.StitchSeries.ui.histogram.show()
        self.StitchSeries.ui.roiBtn.hide()
        self.StitchSeries.ui.menuBtn.hide()
        
        view = self.StitchSeries.getView()
        state = view.getState()        
        self.StitchSeries.setImage(series) 
        view.setState(state)
        view.setBackgroundColor(self.parent.color1)
        
        self.StitchSeries.autoRange()
        
        histplot = self.StitchSeries.getHistogramWidget()
        histplot.setBackground(self.parent.color1)
        
        histplot.region.setBrush(pg.mkBrush(self.parent.color5 + (120,)))
        histplot.region.setHoverBrush(pg.mkBrush(self.parent.color5 + (60,)))
        histplot.region.pen = pg.mkPen(self.parent.color5)
        histplot.region.lines[0].setPen(pg.mkPen(self.parent.color5, width=2))
        histplot.region.lines[1].setPen(pg.mkPen(self.parent.color5, width=2))
        histplot.fillHistogram(color = self.parent.color5)        
        histplot.autoHistogramRange()       

    def defaultIV(self):
        self.KADSeries.ui.histogram.hide()
        self.KADSeries.ui.roiBtn.hide()
        self.KADSeries.ui.menuBtn.hide()
        
        view = self.KADSeries.getView()
        view.setBackgroundColor(self.parent.color1)
        
        ROIplot = self.KADSeries.getRoiPlot()
        ROIplot.setBackground(self.parent.color1)
        
        self.StitchSeries.ui.histogram.hide()
        self.StitchSeries.ui.roiBtn.hide()
        self.StitchSeries.ui.menuBtn.hide()
        
        view = self.StitchSeries.getView()
        view.setBackgroundColor(self.parent.color1)