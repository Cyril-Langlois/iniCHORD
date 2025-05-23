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
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
import numpy as np
from pyqtgraph.Qt import QtGui

#------------------------------import for pypi lib use-------------------------
import inichord.General_Functions as gf

#------------------------------import for local dev use------------------------
# import General_Functions as gf


from skimage.restoration import denoise_tv_chambolle
from skimage import filters, exposure
from skimage.morphology import disk

path2thisFile = abspath(getsourcefile(lambda:0))
uiclass, baseclass = pg.Qt.loadUiType(os.path.dirname(path2thisFile) + "/Denoise_2Dmap.ui")

class MainWindow(uiclass, baseclass):
    def __init__(self,parent):
        super().__init__()

        self.setupUi(self)
        self.parent = parent
        
        self.OpenData.clicked.connect(self.loaddata) # Load data
        self.Push_validate.clicked.connect(self.extract_data)
        
        self.setWindowIcon(QtGui.QIcon('icons/filter_icon.png')) 
        self.defaultIV() # Default ImageView (when no image)
        
        try: # if data is imported from the main GUI
            self.expStack = parent.KAD
            self.denoised_map = np.copy(self.expStack)
            
            self.expStack = self.check_type(self.expStack) # Convert data to float32 if needed
            self.denoised_map = self.check_type(self.denoised_map) # Convert data to float32 if needed   
            
            self.CLAHE_map = np.copy(self.expStack)
            self.displayExpStack(self.CLAHE_map)
                 
        except:
            pass
        
        # CLAHE
        self.CLAHE_SpinBox.valueChanged.connect(self.CLAHE_changed) # Change initial filtering
        
        # NLMD
        self.patch_size = 5
        self.patch_distance = 6
        self.param_h = self.slider_h.value() / 10_0.0
        
        self.label_distance.setText("Patch Distance: " + str(self.patch_distance))
        self.label_size.setText("Patch Size: " + str(self.patch_size))
        self.label_h.setText("Parameter h: " + str(self.param_h))
        
        self.slider_size.valueChanged.connect(self.size_changed)
        self.slider_distance.valueChanged.connect(self.distance_changed)
        self.slider_h.valueChanged.connect(self.h_changed)
        
        # VSNR
        self.noise = self.slider_noiseVSNR.value() / 100.0
        self.label_noiseVSNR.setText("Noise level: " + str(self.noise))
        self.slider_noiseVSNR.valueChanged.connect(self.noiseVSNR_changed)
        
        #TV
        self.noise = self.slider_noiseTV.value() / 100.0
        self.label_noiseTV.setText("Noise level: " + str(self.noise))
        self.slider_noiseTV.valueChanged.connect(self.noiseTV_changed)

        # Gaussian blur and median         
        self.GB_box.currentTextChanged.connect(self.GB_changed)
        self.MED_box.currentTextChanged.connect(self.MED_changed)

        app = QApplication.instance()
        screen = app.screenAt(self.pos())
        geometry = screen.availableGeometry()
        
        # Position (self.move) and size (self.resize) of the main GUI on the screen
        self.move(int(geometry.width() * 0.1), int(geometry.height() * 0.1))
        self.resize(int(geometry.width() * 0.8), int(geometry.height() * 0.6))
        self.screen = screen

    def loaddata(self):
        StackLoc, StackDir = gf.getFilePathDialog("2D map") # Image importation
        
        self.expStack = [] # Initialization of the variable self.image (which will be the full series folder)
        self.expStack = tf.TiffFile(StackLoc[0]).asarray() # Import the unit 2D array

        self.expStack = np.flip(self.expStack, 0)
        self.expStack = np.rot90(self.expStack, k=1, axes=(1, 0))

        self.denoised_map = np.copy(self.expStack)
        
        self.expStack = self.check_type(self.expStack) # Convert data to float32 if needed
        self.denoised_map = self.check_type(self.denoised_map) # Convert data to float32 if needed   

        self.displayExpStack(self.expStack)

    def check_type(self,data): # Check if the data has type uint8 or uint16 and modify it to float32
        self.data = np.nan_to_num(data)
        self.maxInt = np.max(self.data)

        return self.data

    # CLAHE
    def CLAHE_changed(self):
        self.CLAHE_value = self.CLAHE_SpinBox.value()
        
        if self.CLAHE_value == 0:
            self.CLAHE_map = np.copy(self.expStack)
        else:
            self.CLAHE_map = exposure.equalize_adapthist(self.expStack, kernel_size=None, clip_limit=self.CLAHE_value, nbins=256) # CLAHE step

        self.displayExpStack(self.CLAHE_map) # Display the map after load

    # NLMD
    def denoiseNLMD(self):
        value = self.slider_h.value() 
        
        if self.maxInt < 2:
            self.param_h = value / 200_00.0  
        elif self.maxInt < 256:
            self.param_h = value / 10_0.0  
        else:
            self.param_h = value
        
        a = gf.NonLocalMeanDenoising(self.CLAHE_map, self.param_h, True, self.patch_size, self.patch_distance)
        
        self.denoised_map[:, :] = a

        self.displayExpStack(self.denoised_map)
    
    def size_changed(self):
        value = self.slider_size.value()
        self.patch_size = value
        self.label_size.setText("Patch Size: " + str(value))
        self.denoiseNLMD()
    
    def distance_changed(self):
        value = self.slider_distance.value()
        self.threshold = value
        self.label_distance.setText("Patch Distance: " + str(value))
        self.denoiseNLMD()
        
    def h_changed(self):
        value = self.slider_h.value() 
        
        if self.maxInt < 2:
            self.param_h = value / 100_00.0  
        elif self.maxInt < 256:
            self.param_h = value / 10_0.0  
        else:
            self.param_h = value
            
        self.label_h.setText("Parameter h: " + str(self.param_h))

        self.denoiseNLMD()  

    # VSNR
    def noiseVSNR_changed(self):
        self.noise = self.slider_noiseVSNR.value() / 100.0        
        self.label_noiseVSNR.setText("Noise level: " + str(self.noise))
        self.denoiseVSNR()
    
    def denoiseVSNR(self):
        filter_ = [{'name':'Dirac', 'noise_level':self.noise}]  

        self.denoised_map = gf.VSNR_funct(self.CLAHE_map, filter_)
        self.displayExpStack(self.denoised_map)
        
    # TV chambolle
    def noiseTV_changed(self):
        if self.maxInt < 2:
            self.noise = self.slider_noiseTV.value() / 100_00.0  
        elif self.maxInt < 256:
            self.noise = self.slider_noiseTV.value()
        else:
            self.noise = self.slider_noiseTV.value() * 100
        
        self.label_noiseTV.setText("Noise level: " + str(self.noise))
        self.denoiseTV()
    
    def denoiseTV(self):
        self.denoised_map = denoise_tv_chambolle(self.CLAHE_map, self.noise)
        self.displayExpStack(self.denoised_map)

    # GB and MED filters
    def GB_changed(self):
        self.GB_value = int(self.GB_box.currentText())
        self.denoised_map = filters.gaussian(self.CLAHE_map, self.GB_value)
        self.displayExpStack(self.denoised_map)
        
    def MED_changed(self):
        self.MED_value = int(self.MED_box.currentText())
        self.denoised_map = filters.median(self.CLAHE_map, disk(self.MED_value))
        self.displayExpStack(self.denoised_map)

    def extract_data(self): # Save data in a folder
        
        self.denoised_map = np.flip(self.denoised_map, 0)
        self.denoised_map = np.rot90(self.denoised_map, k=1, axes=(1, 0))
        
        gf.Saving_img_or_stack(self.denoised_map)
    
        self.close()

    def defaultIV(self):
        self.expSeries.clear()
        self.expSeries.ui.histogram.hide()
        self.expSeries.ui.roiBtn.hide()
        self.expSeries.ui.menuBtn.hide()
        
        view = self.expSeries.getView()
        view.setBackgroundColor(self.parent.color1)
        
        ROIplot = self.expSeries.getRoiPlot()
        ROIplot.setBackground(self.parent.color1)

    def displayExpStack(self, Series):
        self.expSeries.ui.histogram.hide()
        self.expSeries.ui.roiBtn.hide()
        self.expSeries.ui.menuBtn.hide()
        
        view = self.expSeries.getView()
        state = view.getState()        
        self.expSeries.setImage(Series) 
        view.setState(state)
        
        view.setBackgroundColor(self.parent.color1)
        ROIplot = self.expSeries.getRoiPlot()
        ROIplot.setBackground(self.parent.color1)
        
        font=QtGui.QFont('Noto Sans Cond', 9)
        ROIplot.getAxis("bottom").setTextPen('k') # Apply size of the ticks label
        ROIplot.getAxis("bottom").setTickFont(font)
        
        self.expSeries.timeLine.setPen(color=self.parent.color3, width=15)
        self.expSeries.frameTicks.setPen(color=self.parent.color1, width=5)
        self.expSeries.frameTicks.setYRange((0, 1))

        s = self.expSeries.ui.splitter
        s.handle(1).setEnabled(True)
        s.setStyleSheet("background: 5px white;")
        s.setHandleWidth(5) 