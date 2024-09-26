# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:18:21 2024

@author: glhote1
"""

#%% Imports

import os
from os.path import abspath

from inspect import getsourcefile
import tifffile as tf
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from PyQt5.QtWidgets import QApplication

from LibrairiesCyril import general_functions as gf # USED
from LibrairiesCyril import Alignement_GUI_pg_12_TSG as align # USED
from LibrairiesCyril import remFFT_pg_v2_TSG as RemFFT # USED
from LibrairiesCyril import remOutliersGUI_pg_v3_TSG as rO # USED
from LibrairiesCyril import Denoise_GUI_pg_03_TSG as autoden # USED
from LibrairiesCyril import KAD_function_02 as KADfunc # USED
from LibrairiesCyril import Contour_map_pg_01_TSG as Contour

import tkinter as tk
from tkinter import filedialog

path2thisFile = abspath(getsourcefile(lambda:0))
uiclass, baseclass = pg.Qt.loadUiType(os.path.dirname(path2thisFile) + "/Batch_processing_ui_02.ui")

class MainWindow(uiclass, baseclass):
    def __init__(self,parent):
        super().__init__()

        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon('icons/Main_icon.png')) # Application of the main icon
        self.parent = parent
        
        self.OpenData.clicked.connect(self.loaddata) # Load data
        self.Ref_bttn.clicked.connect(self.loadref) # Load image reference
        self.Run_bttn.clicked.connect(self.run_batch) # Run batch
        self.ChoiceBox_algo.currentTextChanged.connect(self.Filter_changed) # Change searching range
        
        self.flagref = 0
        
        self.Run_bttn.setEnabled(False) # Run button is disable until data opening.
        
        app = QApplication.instance()
        screen = app.screenAt(self.pos())
        geometry = screen.availableGeometry()
        
        # Position (self.move) and size (self.resize) of the main GUI on the screen
        self.move(int(geometry.width() * 0.1), int(geometry.height() * 0.1))
        self.resize(int(geometry.width() * 0.4), int(geometry.height() * 0.6))
        self.screen = screen

    def loaddata(self):
        StackLoc, StackDir = gf.getFilePathDialog("Series") # Image importation
        
        self.image = [] # Initialization of the variable self.image (which will be the full series folder)
        
        if len(StackLoc) > 1:
            self.progressBar.setValue(0) # Set the initial value of the Progress bar at 0
            self.progressBar.setRange(0, len(StackLoc)-1) 
            self.progressBar.setFormat("Loading series... %p%")

        for i in range(0,len(StackLoc)):
            Var = tf.TiffFile(StackLoc[i]).asarray() # Import the unit 2D array
            self.image.append(Var) # Append every 2D array in a list named self.image
            
            QApplication.processEvents()    
            self.ValSlice = i
            self.progression_bar()
            
        self.progressBar.setFormat("Series have been loaded!")
        
        self.Info_box.ensureCursorVisible()
        self.Info_box.insertPlainText("\n Image series have been loaded.")
        self.Info_box.insertPlainText("\n ----------")
        
        self.Run_bttn.setEnabled(True)

    def loadref(self):
        StackLoc, StackDir = gf.getFilePathDialog("image reference") # Image importation
        
        self.ref = [] # Initialization of the variable self.image (which will be the full series folder)
        
        for i in range(0,len(StackLoc)):
            Var = tf.TiffFile(StackLoc[i]).asarray() # Import the unit 2D array
            self.ref.append(Var) # Append every 2D array in a list named self.image
        
        self.Info_box.ensureCursorVisible()
        self.Info_box.insertPlainText("\n Reference images have been loaded.")
        
        for i in range(0,len(self.image)):
            ref2 = np.reshape(self.ref[i],(-1,len(self.ref[i]),len(self.ref[i][0])))
            self.image[i] = np.concatenate((ref2, self.image[i]))
        
        self.flagref = 1
        self.meanRef_val.setEnabled(False)

    def Filter_changed(self): # KAD filtering processes
        self.Filter_choice = self.ChoiceBox_algo.currentText()
        
        if self.Filter_choice == "NLMD":
            self.spinStart = self.spinStart_val.setValue(0)
            self.spinEnd = self.spinEnd_val.setValue(20)
            self.spinNbr = self.spinNbr_val.setValue(30)
            
        elif self.Filter_choice == "BM3D":
            self.spinStart = self.spinStart_val.setValue(0)
            self.spinEnd = self.spinEnd_val.setValue(20)
            self.spinNbr = self.spinNbr_val.setValue(15)
            
        elif self.Filter_choice == "VSNR":
            self.spinStart = self.spinStart_val.setValue(0)
            self.spinEnd = self.spinEnd_val.setValue(5)
            self.spinNbr = self.spinNbr_val.setValue(15)
            
        elif self.Filter_choice == "TV Chambolle":
            self.spinStart = self.spinStart_val.setValue(0)
            self.spinEnd = self.spinEnd_val.setValue(20)
            self.spinNbr = self.spinNbr_val.setValue(30)

    def run_batch(self):
        # Initialiser Tkinter
        root = tk.Tk()
        root.withdraw()  # Masquer la fenêtre principale Tkinter
        # Ouvrir une boîte de dialogue pour choisir un dossier de sauvegarde
        dossier = filedialog.askdirectory(title="Choose a saving folder for treated series")
        
        if len(self.image) > 1:
            self.progressBar.setValue(0) # Set the initial value of the Progress bar at 0
            self.progressBar.setRange(0, len(self.image)-1) 
            self.progressBar.setFormat("Treatment... %p%")
        
        # Variable extraction      
        self.radius_reg = self.Radius_box.value()
        self.threshold_reg = self.Threshold_box.value()
        self.blur_reg = int(self.Blur_box.currentText())
        self.sobel_reg = int(self.Sobel_box.currentText())
        self.reg1 = self.transfo_comboBox.currentText()
        self.reg2 = self.transfo_comboBox2.currentText()
        
        self.FFT = self.FFT_val.value()
        
        self.radius_remout = self.radius_remout_val.value()
        self.threshold_remout = self.threshold_remout_val.value()
        
        self.spinStart = self.spinStart_val.value()
        self.spinEnd = self.spinEnd_val.value()
        self.spinNbr = self.spinNbr_val.value()
        self.meanRef = self.meanRef_val.value()
        
        self.radius = self.Radius_box_2.value()
        self.threshold = self.Threshold_box_2.value()
        self.blur = int(self.Blur_box_2.currentText())
        self.sobel = int(self.Sobel_box_2.currentText())
        
        self.Denoising_algo = self.ChoiceBox_algo.currentText()
        self.Index_algo = self.Choice_Idx.currentText()
                
        for i in range(0,len(self.image)):
            
            self.Info_box.ensureCursorVisible()
            self.Info_box.insertPlainText(f"\n Current processed series: {i+1} out of {len(self.image)}")
            
            QApplication.processEvents()    
            self.ValSlice = i
            self.progression_bar()

            # Creation of the Main_TSG object that will store modifications
            self.Info_box.ensureCursorVisible()
            self.Info_box.insertPlainText("\n     Main object initialization.")
            
            Main_TSG = self.parent
            Main_TSG.Current_stack = self.image[i]
            
            # First registration step
            w = align.MainWindow(Main_TSG)
        
            w.radius_Val = self.radius_reg
            w.threshold_Val = self.threshold_reg
            w.blur_value = self.blur_reg
            w.sobel_value = self.sobel_reg
            w.choice_transfo = str(self.reg1)
        
            self.Info_box.ensureCursorVisible()
            self.Info_box.insertPlainText("\n     Registration step I in progress.")
            
            self.Info_box.ensureCursorVisible()
            self.Info_box.insertPlainText("\n     Microstructural features definition I.")
            w.Pre_treatment()

            self.Info_box.ensureCursorVisible()
            self.Info_box.insertPlainText("\n     Start of sequential registration I.")
            w.Seq_registration()
            
            self.Info_box.ensureCursorVisible()
            self.Info_box.insertPlainText("\n     Crop extra black border I.")
            w.Crop_data()
            
            Main_TSG.Current_stack = w.Cropped_stack
            
            # Second registration step
            if self.reg2 != "None":
                
                self.Info_box.ensureCursorVisible()
                self.Info_box.insertPlainText("\n     Registration step II in progress.")
                
                wbis = align.MainWindow(Main_TSG)
            
                wbis.radius_Val = self.radius_reg
                wbis.threshold_Val = self.threshold_reg
                wbis.blur_value = self.blur_reg
                wbis.sobel_value = self.sobel_reg
                wbis.choice_transfo = str(self.reg2)
                
                self.Info_box.ensureCursorVisible()
                self.Info_box.insertPlainText("\n     Microstructural features definition II.")
                wbis.Pre_treatment()
  
                self.Info_box.ensureCursorVisible()
                self.Info_box.insertPlainText("\n     Start of sequential registration II.")
                wbis.Seq_registration()
                
                self.Info_box.ensureCursorVisible()
                self.Info_box.insertPlainText("\n     Crop extra black border II.")
                wbis.Crop_data()
            
                Main_TSG.Current_stack = wbis.Cropped_stack
 
            # Background substraction
            w2 = RemFFT.MainWindow(Main_TSG)
        
            w2.fft = self.FFT
            
            self.Info_box.ensureCursorVisible()
            self.Info_box.insertPlainText("\n     Background substraction step in progress.")
            w2.FFTStack()
        
            Main_TSG.Current_stack = w2.filtered_Stack
        
            # Outlier filtering            
            w3 = rO.MainWindow(Main_TSG)
        
            w3.radius = self.radius_remout
            w3.threshold = self.threshold_remout
            
            self.Info_box.ensureCursorVisible()
            self.Info_box.insertPlainText("\n     Remove outliers step in progress.")
            w3.remOutStack()
        
            Main_TSG.Current_stack = w3.denoised_Stack
        
            # Auto-denoising
            self.Info_box.ensureCursorVisible()
            self.Info_box.insertPlainText("\n     Denoising step in progress.")
            
            Main_TSG.flag = False
            
            w4 = autoden.MainWindow(Main_TSG)
        
            w4.AVG_value = self.meanRef
            w4.Spin_first_value = self.spinStart
            w4.Spin_final_value = self.spinEnd
            w4.Spin_nbr_value = self.spinNbr
            w4.Step_choice = self.Denoising_algo
            w4.Idx = self.Index_algo
            
            if self.flagref == 0 :
                self.Info_box.ensureCursorVisible()
                self.Info_box.insertPlainText("\n     Proxy image of reference creation.")
                w4.AVG_slices()
                
            elif self.flagref == 1 :
                self.Info_box.ensureCursorVisible()
                self.Info_box.insertPlainText("\n     Search using the image reference.")
                
                w4.Reference = np.copy(w4.expStack[0,:,:]) # Extraction of the image of reference
                w4.Slice1 = np.copy(w4.expStack[1,:,:]) # Copy of the first slice of the stack
                w4.expStack = np.copy(w4.expStack[1:,:,:]) # Create of the Current stack without reference
                w4.Denoise_stack = np.copy(w4.expStack)
                
            self.Info_box.ensureCursorVisible()
            self.Info_box.insertPlainText("\n     Search for the perfect denoising parameter.")
            w4.AutoDenoisingStep()
            
            self.Info_box.ensureCursorVisible()
            self.Info_box.insertPlainText("\n     The SSIM denoising parameters is : " + str(w4.idx_SSIM_info))
            self.Info_box.ensureCursorVisible()
            self.Info_box.insertPlainText("\n     The MSE denoising parameters is : " + str(w4.idx_MSE_info))
            
            self.Info_box.ensureCursorVisible()
            self.Info_box.insertPlainText("\n     Stack denoising using the absolute and perfect denoising parameter.")
            w4.StackDenoising()
        
            Main_TSG.Current_stack = w4.Denoise_stack
        
            # KAD computation 
            self.Info_box.ensureCursorVisible()
            self.Info_box.insertPlainText("\n     KAD computation step in progress.")
            
            stack_norm = KADfunc.centeredEuclidianNorm(Main_TSG.Current_stack, 0) # Normalization of the image series
            KAD = KADfunc.Divided_KAD(stack_norm) # Compute the KAD map
            
            # Contour map computation
            self.Info_box.ensureCursorVisible()
            self.Info_box.insertPlainText("\n     Contour map in progress.")
            
            w5 = Contour.MainWindow(Main_TSG)
            
            w5.radius = self.radius
            w5.threshold = self.threshold
            w5.blur = self.blur
            w5.sobel = self.sobel
            
            w5.compute_map()
            Contour_map = w5.contour_map
            
            # ask for saving datas 
            chemin_complet = os.path.join(dossier, f"Treated_serie_{i+1}.tiff")
            # Sauvegarder le tableau 3D au format TIFF avec le type de données d'origine
            tf.imwrite(chemin_complet, Main_TSG.Current_stack)
            
            chemin_complet = os.path.join(dossier, f"KAD_{i+1}.tiff")
            # Sauvegarder le tableau 3D au format TIFF avec le type de données d'origine
            tf.imwrite(chemin_complet, KAD)
            
            chemin_complet = os.path.join(dossier, f"Contour_{i+1}.tiff")
            # Sauvegarder le tableau 3D au format TIFF avec le type de données d'origine
            tf.imwrite(chemin_complet, Contour_map)
            
            del(KAD, stack_norm, Main_TSG, Contour_map)
            
        self.Info_box.ensureCursorVisible()
        self.Info_box.insertPlainText("\n ----------")
        self.Info_box.insertPlainText("\n Treated series have been saved.")

    def progression_bar(self): # Function for the ProgressBar uses
        self.prgbar = self.ValSlice
        self.progressBar.setValue(self.prgbar)
