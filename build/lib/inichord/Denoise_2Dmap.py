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

from inichord import General_Functions as gf

path2thisFile = abspath(getsourcefile(lambda:0))
uiclass, baseclass = pg.Qt.loadUiType(os.path.dirname(path2thisFile) + "/Denoise_2Dmap.ui")

class MainWindow(uiclass, baseclass):
    def __init__(self,parent):
        super().__init__()

        self.setupUi(self)
        self.parent = parent
        
        self.OpenData.clicked.connect(self.loaddata) # Load data
        # self.preview.clicked.connect(self.run_Denoising) # Run batch
        self.Push_validate.clicked.connect(self.extract_data)
        
        self.setWindowIcon(QtGui.QIcon('icons/filter_icon.png')) 
        self.defaultIV() # Default ImageView (when no image)
        
        try: # if data is imported from the main GUI
            self.expStack = parent.KAD
            self.denoised_map = np.copy(self.expStack)
            
            self.expStack = self.check_type(self.expStack) # Convert data to float32 if needed
            self.denoised_map = self.check_type(self.denoised_map) # Convert data to float32 if needed   
            
            self.displayExpStack(self.expStack)
            
            self.run_Denoising()
            
        except:
            pass
        
        self.patch_size = 5
        self.patch_distance = 6
        self.param_h = self.slider_h.value() / 10_0.0
        
        self.label_distance.setText("Patch Distance: " + str(self.patch_distance))
        self.label_size.setText("Patch Size: " + str(self.patch_size))
        self.label_h.setText("Parameter h: " + str(self.param_h))
        
        self.slider_size.valueChanged.connect(self.size_changed)
        self.slider_distance.valueChanged.connect(self.distance_changed)
        self.slider_h.valueChanged.connect(self.h_changed)
        
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
        
        self.run_Denoising()

    def check_type(self,data): # Check if the data has type uint8 or uint16 and modify it to float32
        datatype = data.dtype
        print(datatype)

        if datatype == "float64":
            data = gf.convertToUint8(data)
            self.data = data.astype(np.float32)
        if datatype == "uint16":
            data = gf.convertToUint8(data)
            self.data = data.astype(np.float32)
        elif datatype == "uint8":
            self.data = data.astype(np.float32)
        elif datatype == "float32":
            data = gf.convertToUint8(data)
            self.data = data.astype(np.float32)
            
        return self.data

    def run_Denoising(self):
        a = gf.NonLocalMeanDenoising(self.expStack[:, :], self.param_h, True, self.patch_size, self.patch_distance)
        
        self.denoised_map[:, :] = a

        self.displayExpStack(self.denoised_map)
    
    def size_changed(self):
        value = self.slider_size.value()
        self.patch_size = value
        self.label_size.setText("Patch Size: " + str(value))
        self.run_Denoising()
    
    def distance_changed(self):
        value = self.slider_distance.value()
        self.threshold = value
        self.label_distance.setText("Patch Distance: " + str(value))
        self.run_Denoising()
        
    def h_changed(self):
        value = self.slider_h.value()
        self.param_h = value / 10_0.0        
        self.label_h.setText("Parameter h: " + str(self.param_h))
        self.run_Denoising()  
        
    def extract_data(self):
        self.parent.KAD = np.copy(self.denoised_map) # Copy in the main GUI
        self.parent.StackList.append(self.denoised_map) # Add the data in the stack list
        
        Combo_text = '\u2022 Denoised 2D map'
        Combo_data = self.denoised_map
        self.parent.choiceBox.addItem(Combo_text, Combo_data) # Add the data in the QComboBox

        self.parent.displayDataview(self.parent.KAD) # Display the labeled grain
        self.parent.choiceBox.setCurrentIndex(self.parent.choiceBox.count() - 1) # Show the last data in the choiceBox QComboBox

        self.parent.Info_box.ensureCursorVisible()
        self.parent.Info_box.insertPlainText("\n \u2022 Denoised 2D map.") 
        
        self.parent.Save_button.setEnabled(True)
        self.parent.Reload_button.setEnabled(True)
        self.parent.choiceBox.setEnabled(True)
        
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