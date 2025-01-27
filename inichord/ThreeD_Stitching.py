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
from PyQt5.QtWidgets import QApplication, QMessageBox, QLabel, QDialog, QVBoxLayout, QPushButton
from PyQt5 import QtCore, QtGui

from inichord import General_Functions as gf

from skimage import exposure
from scipy import ndimage as ndi

path2thisFile = abspath(getsourcefile(lambda:0))
uiclass, baseclass = pg.Qt.loadUiType(os.path.dirname(path2thisFile) + "/ThreeD_Stitching.ui")

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
        
        self.val_tresh = self.Tresh_spinBox.value()
        self.Tresh_spinBox.valueChanged.connect(self.Change_treshold)
        
        self.Range_Edit.setText("0")
        self.changeRange() # Modify the searching range 
        self.Range_Edit.editingFinished.connect(self.changeRange) # Modify the searching range 
        
        app = QApplication.instance()
        screen = app.screenAt(self.pos())
        geometry = screen.availableGeometry()
        
        # Control window position and dimensions
        self.move(int(geometry.width() * 0.05), int(geometry.height() * 0.05))
        self.resize(int(geometry.width() * 0.9), int(geometry.height() * 0.6))
        self.screen = screen

#%% Functions
    def popup_message(self,title,text,icon):
        msg = QDialog(self) # Create a Qdialog box
        msg.setWindowTitle(title)
        msg.setWindowIcon(QtGui.QIcon(icon))
        
        label = QLabel(text) # Create a QLabel for the text
        
        font = label.font() # Modification of the font
        font.setPointSize(8)  # Font size modification
        label.setFont(font)
        
        label.setAlignment(QtCore.Qt.AlignCenter) # Text centering
        label.setWordWrap(False)  # Deactivate the line return

        ok_button = QPushButton("OK") # Creation of the Qpushbutton
        ok_button.clicked.connect(msg.accept)  # Close the box when pushed
        
        layout = QVBoxLayout() # Creation of the vertical layout
        layout.addWidget(label)       # Add text
        layout.addWidget(ok_button)   # Add button
        
        msg.setLayout(layout) # Apply position 
        msg.adjustSize() # Automatically adjust size of the window
        
        msg.exec_() # Display the message box
        
    def data_choice(self): # Allow to apply other treatment depending if the data is a KAD one
        msg = QMessageBox.question(self, 'Image stitching', 'Is it a KAD data ?')
        if msg == QMessageBox.Yes:
            self.flag_KAD = True
        if msg == QMessageBox.No:
            self.flag_KAD = False

    def loaddata(self): 
        self.StackLoc, self.StackDir = gf.getFilePathDialog("Image series (*.tiff)") 
        
        checkimage = tf.TiffFile(self.StackLoc[0]).asarray() # Check for dimension. If 2 dimensions : 2D array. If 3 dimensions : stack of images
        
        if checkimage.ndim != 3: # Check if the data is not a sequence of 3D array
            self.popup_message("3D stitching","Image series (3D stacks) must be imported",'icons/Stitch_icon.png')
            return
        
        else:
            self.Base_map = []
    
            for i in range(0,len(self.StackLoc)):
                Var = tf.TiffFile(self.StackLoc[i]).asarray()
                # Replace NaN value by the nearest OK value
                mask = np.isnan(Var)
                Var[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Var[~mask])

                if Var.dtype != 'uint8': # Convert to 8bits if needed
                    Var = gf.convertToUint8(Var)
                
                Search0 = np.where(Var == 0.0) # Replace 0.0 by 1 for final crop efficiency
                Var[Search0] = 1
                
                self.Base_map.append(Var)
                
            self.Base_map_display = self.Base_map.copy() 
            
            for i in range(len(self.Base_map_display)): # Flip and rotation for display
                self.Base_map_display[i] = np.flip(self.Base_map_display[i], 1)
                self.Base_map_display[i] = np.rot90(self.Base_map_display[i], k=1, axes=(2, 1))
            
            self.displaySeries(self.Base_map_display[0])
            self.InfoValues_label.setText(str(self.Base_map_display[0].shape))
            
        self.StackLoc, self.StackDir = gf.getFilePathDialog("KAD maps (*.tiff)")  

        checkimage = tf.TiffFile(self.StackLoc[0]).asarray() # Check for dimension. If 2 dimensions : 2D array. If 3 dimensions : stack of images
        
        if checkimage.ndim != 2: # Check if the data is not a sequence of 2D array
            self.popup_message("3D stitching","2D images must be imported",'icons/Stitch_icon.png')
            return
        
        else:
            self.data_choice()
            self.Base_KAD = []
    
            for i in range(0,len(self.StackLoc)):
                Var = tf.TiffFile(self.StackLoc[i]).asarray()
                # Replace NaN value by the nearest OK value
                mask = np.isnan(Var)
                Var[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Var[~mask])
                # Normalization and CLAHE
                Var = (Var - np.min(Var)) / (np.max(Var) - np.min(Var)) # Normalization step
                Var = exposure.equalize_adapthist(Var, kernel_size=None, clip_limit=0.01, nbins=256) # CLAHE step
    
                if Var.dtype != 'uint8': # Convert to 8bits if needed
                    Var = gf.convertToUint8(Var)
                
                Search0 = np.where(Var == 0.0) # Replace 0.0 by 1 for final crop efficiency
                Var[Search0] = 1
                
                self.Base_KAD.append(Var)
                
            self.Base_KAD_display = self.Base_KAD.copy() 
            
            for i in range(len(self.Base_KAD_display)): # Flip and rotation for display
                self.Base_KAD_display[i] = np.flip(self.Base_KAD_display[i], 0)
                self.Base_KAD_display[i] = np.rot90(self.Base_KAD_display[i], k=1, axes=(1, 0))
            
            self.displayExpKAD(self.Base_KAD_display[0])
    
            # Buttons disabled until first importation
            self.KADBox.setEnabled(True)
            self.col_spinBox.setEnabled(True)
            self.row_spinBox.setEnabled(True)
            self.Range_Edit.setEnabled(True)
            self.Tresh_spinBox.setEnabled(True)
            self.full_range.setEnabled(True)
            self.ComputeClass_bttn.setEnabled(True)
            
        range_step = []
        for i in range(0,len(self.Base_KAD)):
            range_step.append(np.mean(self.Base_KAD[i].shape[1]))

        auto_val_range = int(np.round(np.mean(range_step)*0.3)) # 30% of the mean dimensions of images

        self.Range_Edit.setText(str(auto_val_range))
        self.val_range = auto_val_range
        
    def set_up(self):
        # Here, nbr of column and nbr of raw are defined. self.Super_BSE and self.Super_KAD are created.
        # it allow to work row by row for the next computation step
        self.col_nbr = self.col_spinBox.value()
        self.row_nbr = self.row_spinBox.value()
        self.slice_nbr = len(self.Base_map[0])
        
        self.Super_BSE = []
        self.Super_KAD = []
        
        step = 0
        
        for i in range(0,self.row_nbr):
            self.Super_BSE.append(self.Base_map[step:self.col_nbr + step])
            self.Super_KAD.append(self.Base_KAD[step:self.col_nbr + step])
            
            step = step + self.col_nbr
            
        self.Super_BSE = [x for x in self.Super_BSE if x != []]
        self.Super_KAD = [x for x in self.Super_KAD if x != []]        
    
    def Change_KAD_display(self): # Visualization of the different arrays 
        self.KADBox.setRange(0,len(self.Base_KAD)-1)
        self.KADBox.setSingleStep(1)
        self.Value = self.KADBox.value()
        
        self.displayExpKAD(self.Base_KAD_display[self.Value])
        self.displaySeries(self.Base_map_display[self.Value])
        self.InfoValues_label.setText(str(self.Base_map_display[self.Value].shape))

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
        
        try:
            self.set_up() # Prepare data
            self.transformation_determination() # Check for transformation using the KAD data
            self.apply_transformation() # Apply the transformation on the image series
            
            self.Push_export.setEnabled(True)
            self.Save_bttn.setEnabled(True)
        except:
            self.popup_message("3D stitching","Stitch failed. Please check that the [columns - rows] information match the imported data.",'icons/Stitch_icon.png')
            return

    def transformation_determination(self):
        self.res = []
        self.M_hori = []
        self.M_verti = []
        self.Sel1=[]
        self.Sel2=[]
        
        self.nbr = self.col_nbr * self.row_nbr # Nbr of total 2D array
        
        self.prgbar = 0 # Progress bar initial value
        self.progressBar.setValue(self.prgbar)
        self.progressBar.setRange(0, self.nbr-1)
        self.increment = 0

        if self.col_nbr > 1: # If there are more than 1 column
            for i in range(0,len(self.Super_KAD)):
                for j in range(len(self.Super_KAD[0])-1,0,-1):  
                    # Find matches between images 
                    M = self.find_matches(self.Super_KAD[i][j],self.Super_KAD[i][j-1],direction = "horizontal")
                    # Store transformation matrix M
                    self.M_hori.append(M)
                    # Apply transformation matrix M
                    dst = self.horizontal_stitching(self.Super_KAD[i][j],self.Super_KAD[i][j-1],M)
                    # Stitch the two images together
                    self.Super_KAD[i][j] = self.Super_KAD[i][0:j]
                    self.Super_KAD[i][j-1] = dst
                    
                    QApplication.processEvents()    
                    self.increment = self.increment + 1
                    self.ValSlice = self.increment
                    self.progression_bar()
                    
                    # j is modified because the lenght of Super listing is evolving at each step
                    j = len(self.Super_KAD[i])-1
                    
                    # The twoD stitching is cropped
                    self.cropped_dst,selection = self.cropping_step(dst)
                    self.Sel1.append(selection) 
                    
                # The twoD stitching is stored to be used after 
                self.res.append(self.cropped_dst)

        else : # If there is only 1 column, self.res is a copy of self.KAD_base 
            self.res = self.Base_KAD.copy()

        if self.row_nbr > 1: # If the nbr of row is higher than 1
            for i in range(len(self.res)-1,0,-1):
                # Find matches between images 
                M = self.find_matches(self.res[i],self.res[i-1],direction = "vertical")
                # Store transformation matrix M
                self.M_verti.append(M)
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
                self.cropped_dst,selection = self.cropping_step(dst)
                self.Sel2.append(selection)
         
        # Apply flip and rotation for display of the stitched 2D arrays
        self.displayed_dst = np.copy(self.cropped_dst)
        self.displayed_dst = np.flip(self.displayed_dst, 0)
        self.displayed_dst = np.rot90(self.displayed_dst, k=1, axes=(1, 0))
                
        self.displayKADStitch(self.displayed_dst)

    def find_matches(self,data1, data2, direction = "horizontal"):
        sift = cv2.SIFT_create()
        
        # Allow to specifiy if the descriptor determination must be apply on the whole images or only a part of it
        if self.full_range.isChecked():
            self.val_range = len(data2[0])

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
            self.popup_message("3D stitching","Not enough descriptor...",'icons/Stitch_icon.png')
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
        
        return Cropped_dst, Selection
    
    def apply_transformation(self): # Apply the transformation matrices on the 3D array sequence
        self.res_serie = []
        
        self.prgbar = 0 # Progress bar initial value
        self.progressBar.setValue(self.prgbar)
        self.progressBar.setRange(0, self.nbr-1)
        self.increment = 0
        
        for j in range(0,self.slice_nbr):
            k = 0  # Count for apply transformation matrix
            
            self.tempo_res = []
            
            if self.col_nbr > 1: # If there are more than 1 column            
                for a in range(0,len(self.Super_BSE)):
                    for i in range(len(self.Super_BSE[0])-1,0,-1): # 2 then 1
                        if i == len(self.Super_BSE[0])-1:
                            data1 = self.Super_BSE[a][i][j]
                            
                        data2 = self.Super_BSE[a][i-1][j]
                        
                        if self.flag_affine == True : 
                            dst = cv2.warpAffine(data1,self.M_hori[k],(data2.shape[1] + data1.shape[1], data2.shape[0]))
                        else :
                            dst = cv2.warpPerspective(data1,self.M_hori[k],(data2.shape[1] + data1.shape[1], data2.shape[0]))
                        
                        dst[0:data2.shape[0],0:data2.shape[1]] = data2
                        
                        cropped_dst = dst[self.Sel1[k][1]:self.Sel1[k][3],self.Sel1[k][0]:self.Sel1[k][2]]
                        
                        k = k+1 # Count is increased
                        data1 = dst
                        
                        QApplication.processEvents()    
                        self.increment = self.increment + 1
                        self.ValSlice = self.increment
                        self.progression_bar()
                        
                    self.tempo_res.append(cropped_dst)
            else :
                self.X = self.Super_BSE.copy()
                            
                self.tempo_res = []
                for y in range(0,len(self.X)):
                    var = self.X [y][0][j]
                
                    self.tempo_res.append(var)
            
            k = 0 
                
            for i in range(len(self.tempo_res)-1,0,-1):
                if i == len(self.Super_BSE)-1:
                    data1 = self.tempo_res[i]
                    
                data2 = self.tempo_res[i-1]
                
                if self.flag_affine == True : 
                    dst = cv2.warpAffine(data1,self.M_verti[k],(data2.shape[1] + data1.shape[1], data2.shape[0] + data1.shape[0]))
                    dst[0:data2.shape[0],0:data2.shape[1]] = data2
                else : 
                    dst = cv2.warpPerspective(data1,self.M_verti[k],(data2.shape[1] + data1.shape[1], data2.shape[0] + data1.shape[0]))
                    dst[0:data2.shape[0],0:data2.shape[1]] = data2
                
                cropped_dst = dst[self.Sel2[k][1]:self.Sel2[k][3],self.Sel2[k][0]:self.Sel2[k][2]]
                
                k = k+1 # Count is increased
                
                data1 = dst
                
                QApplication.processEvents()    
                self.increment = self.increment + 1
                self.ValSlice = self.increment
                self.progression_bar()
            
            self.res_serie.append(cropped_dst)
        
        self.res_serie_cropped = np.copy(self.res_serie)
        
        self.displayed_res_serie = np.copy(self.res_serie_cropped)
        self.displayed_res_serie = np.flip(self.displayed_res_serie, 1)
        self.displayed_res_serie = np.rot90(self.displayed_res_serie, k=1, axes=(2, 1))
                
        self.displaySeriesStitch(self.displayed_res_serie)
    
    def changeRange(self): # Take into account the searching range modification
        self.val_range = int(self.Range_Edit.text())
        self.full_range.setChecked(False)
        
    def Change_treshold(self): # Take into account the descriptor filtering threshold
        self.val_tresh = self.Tresh_spinBox.value()
        
    def Mask_min(self,Min_Aligned_stack, threshold):
        Mask = np.zeros((len(Min_Aligned_stack),len(Min_Aligned_stack[0])))
        Mask[Min_Aligned_stack > threshold] = 1
        return Mask
    
    def progression_bar(self): # Fonction relative à la barre de progression
        self.prgbar = self.ValSlice
        self.progressBar.setValue(self.prgbar)

    def Save_results(self):
        ti = time.strftime("%Y-%m-%d__%Hh-%Mm-%Ss") # Absolute time 
        
        directory = "Stitched_map_Series_" + ti # Name of the main folder
        PathDir = os.path.join(self.StackDir, directory)  # where to create the main folder
        os.mkdir(PathDir)  # Create main folder

        # Images saving step
        tf.imwrite(PathDir + '/Stitched_map.tiff', np.rot90(np.flip(self.displayed_dst, 0), k=1, axes=(1, 0)).astype('float32')) 
        tf.imwrite(PathDir + '/Stitched_serie.tiff', np.rot90(np.flip(self.displayed_res_serie, 1), k=1, axes=(2, 1)).astype('float32')) 

        # Information (.TXT) step
        with open(PathDir + '\map and serie stitching.txt', 'w') as file:
            file.write("Number of images: " + str(self.slice_nbr))
            file.write("\nNumber of row: " + str(self.row_nbr))   
            file.write("\nNumber of column: " + str(self.col_nbr))  
            if self.full_range.isChecked():
                file.write("\nSearching range (pxls): full images")   
            else:
                file.write("\nSearching range (pxls): "+ str(self.val_range))  
            file.write("\nTransformation: " + str(self.val_tresh)) 
            file.write("\nThreshold: " + str(self.transfo_choice)) 

        # Finished message
        self.popup_message("3D stitching","Saving process is over.",'icons/Stitch_icon.png')

    def export_data(self): # Push stitched image in the main GUI
        # Stitched map (KAD or contour)
        if self.flag_KAD == True: # If stitching has been applied on KAD data 
            self.parent.flag_stitchKAD = True
            
            self.parent.KAD = np.copy(self.displayed_dst) # Copy in the main GUI
            self.parent.StackList.append(self.displayed_dst) # Add the data in the stack list
            
            self.parent.StackDir = self.StackDir
            
            self.parent.displayDataview(self.parent.KAD) # Display the labeled grain
            self.parent.choiceBox.setCurrentIndex(self.parent.choiceBox.count() - 1) # Show the last data in the choiceBox QComboBox
        else :
            self.parent.flag_stitchKAD = False
            
            self.parent.Contour_map = np.copy(self.displayed_dst) # Copy in the main GUI
            self.parent.StackList.append(self.displayed_dst) # Add the data in the stack list
            
            self.parent.StackDir = self.StackDir
            
            self.parent.displayDataview(self.parent.Contour_map) # Display the labeled grain
            self.parent.choiceBox.setCurrentIndex(self.parent.choiceBox.count() - 1) # Show the last data in the choiceBox QComboBox
            
        Combo_text = '\u2022 Stitched map'
        Combo_data = self.displayed_dst
        self.parent.choiceBox.addItem(Combo_text, Combo_data) # Add the data in the QComboBox
     
        self.parent.Info_box.ensureCursorVisible()
        self.parent.Info_box.insertPlainText("\n \u2022 Stitched map.") 
        
        # Stitched serie
        self.parent.flag = False # For auto-denoising access
        self.parent.flag_image = True # For GRDD - GDS
        self.parent.Current_stack = np.copy(self.displayed_res_serie) # Copy in the main GUI
        self.parent.image = np.copy(self.displayed_res_serie) # Copy for the GROD-GOS computation
        self.parent.StackList.append(self.displayed_res_serie) # Add the data in the stack list
        
        self.parent.StackDir = self.StackDir
        
        Combo_text = '\u2022 Stitched serie'
        Combo_data = self.displayed_res_serie
        self.parent.choiceBox.addItem(Combo_text, Combo_data) # Add the data in the QComboBox

        self.parent.displayExpStack(self.parent.Current_stack) # Display the labeled grain
        self.parent.label_Treatment.setText("Stitched serie") # Title of the data
     
        self.parent.Info_box.ensureCursorVisible()
        self.parent.Info_box.insertPlainText("\n \u2022 Stitched serie.")
             
        # Activation of buttons in the main GUI
        self.parent.Edit_tools_button.setEnabled(True)
        self.parent.Registration_button.setEnabled(True)
        self.parent.Background_remover_button.setEnabled(True)
        self.parent.Remove_outliers_button.setEnabled(True)
        self.parent.Manual_denoising_button.setEnabled(True)
        self.parent.Auto_denoising_button.setEnabled(True)
        self.parent.Save_button.setEnabled(True)
        self.parent.Reload_button.setEnabled(True)
        self.parent.Choice_denoiser.setEnabled(True)
        self.parent.choiceBox.setEnabled(True)
        
        for i in range(0,3): # enables [Bleach correction ; STD map ; KAD map]
            self.parent.Tool_choice.model().item(i).setEnabled(True)
        
        # Finished message
        self.popup_message("3D stitching","Stitched KAD has been exported to the main GUI.",'icons/Stitch_icon.png')
        
    def displaySeries(self, series): # Display of initial KAD maps
        self.Series.ui.histogram.hide()
        self.Series.ui.roiBtn.hide()
        self.Series.ui.menuBtn.hide()
        
        view = self.Series.getView()
        state = view.getState()        
        self.Series.setImage(series) 
        view.setState(state)
        
        view.setBackgroundColor(self.parent.color1)
        
        self.Series.autoRange()
        
        ROIplot = self.Series.getRoiPlot()
        ROIplot.setBackground(self.parent.color1)
        
        font=QtGui.QFont('Noto Sans Cond', 8)
        ROIplot.getAxis("bottom").setTextPen('k') # Apply size of the ticks label
        ROIplot.getAxis("bottom").setTickFont(font)
        
        self.Series.timeLine.setPen(color=self.parent.color3, width=15)
        self.Series.frameTicks.setPen(color=self.parent.color1, width=5)
        self.Series.frameTicks.setYRange((0, 1))

        s = self.Series.ui.splitter
        s.handle(1).setEnabled(True)
        s.setStyleSheet("background: 5px white;")
        s.setHandleWidth(5) 
    
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
        
    def displayKADStitch(self, series): # Display of initial KAD map
        self.StitchKAD.ui.histogram.show()
        self.StitchKAD.ui.roiBtn.hide()
        self.StitchKAD.ui.menuBtn.hide()
        
        view = self.StitchKAD.getView()
        state = view.getState()        
        self.StitchKAD.setImage(series) 
        view.setState(state)
        view.setBackgroundColor(self.parent.color1)
        
        self.StitchKAD.autoRange()
        
        histplot = self.StitchKAD.getHistogramWidget()
        histplot.setBackground(self.parent.color1)
        
        histplot.region.setBrush(pg.mkBrush(self.parent.color5 + (120,)))
        histplot.region.setHoverBrush(pg.mkBrush(self.parent.color5 + (60,)))
        histplot.region.pen = pg.mkPen(self.parent.color5)
        histplot.region.lines[0].setPen(pg.mkPen(self.parent.color5, width=2))
        histplot.region.lines[1].setPen(pg.mkPen(self.parent.color5, width=2))
        histplot.fillHistogram(color = self.parent.color5)        
        histplot.autoHistogramRange()       
        
    def displaySeriesStitch(self, series): # Display of initial KAD map
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
        
        ROIplot = self.StitchSeries.getRoiPlot()
        ROIplot.setBackground(self.parent.color1)
        
        font=QtGui.QFont('Noto Sans Cond', 8)
        ROIplot.getAxis("bottom").setTextPen('k') # Apply size of the ticks label
        ROIplot.getAxis("bottom").setTickFont(font)
        
        self.StitchSeries.timeLine.setPen(color=self.parent.color3, width=15)
        self.StitchSeries.frameTicks.setPen(color=self.parent.color1, width=5)
        self.StitchSeries.frameTicks.setYRange((0, 1))

        s = self.StitchSeries.ui.splitter
        s.handle(1).setEnabled(True)
        s.setStyleSheet("background: 5px white;")
        s.setHandleWidth(5) 

    def defaultIV(self):
        self.Series.ui.histogram.hide()
        self.Series.ui.roiBtn.hide()
        self.Series.ui.menuBtn.hide()
        
        view = self.Series.getView()
        view.setBackgroundColor(self.parent.color1)
        
        ROIplot = self.Series.getRoiPlot()
        ROIplot.setBackground(self.parent.color1)
        
        self.KADSeries.ui.histogram.hide()
        self.KADSeries.ui.roiBtn.hide()
        self.KADSeries.ui.menuBtn.hide()
        
        view = self.KADSeries.getView()
        view.setBackgroundColor(self.parent.color1)
        
        ROIplot = self.KADSeries.getRoiPlot()
        ROIplot.setBackground(self.parent.color1)
        
        self.StitchKAD.ui.histogram.hide()
        self.StitchKAD.ui.roiBtn.hide()
        self.StitchKAD.ui.menuBtn.hide()
        
        view = self.StitchKAD.getView()
        view.setBackgroundColor(self.parent.color1)
        
        self.StitchSeries.ui.histogram.hide()
        self.StitchSeries.ui.roiBtn.hide()
        self.StitchSeries.ui.menuBtn.hide()
        
        view = self.StitchSeries.getView()
        view.setBackgroundColor(self.parent.color1)
        
        ROIplot = self.StitchSeries.getRoiPlot()
        ROIplot.setBackground(self.parent.color1)