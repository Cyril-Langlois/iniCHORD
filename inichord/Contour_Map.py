# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:18:21 2024

@author: glhote1
"""

#%% Imports

import os
from os.path import abspath
from inspect import getsourcefile

import cv2
import numpy as np

#------------------------------import for pypi lib use-------------------------
import inichord.General_Functions as gf

#------------------------------import for local dev use------------------------
# import General_Functions as gf

import tifffile as tf

import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtGui

path2thisFile = abspath(getsourcefile(lambda:0))
uiclass, baseclass = pg.Qt.loadUiType(os.path.dirname(path2thisFile) + "/Contour_Map.ui")

class MainWindow(uiclass, baseclass):
    def __init__(self,parent):
        super().__init__()

        self.setupUi(self)
        self.parent = parent
        
        self.OpenData.clicked.connect(self.loaddata) # Load data
        self.Compute_bttn.clicked.connect(self.compute_map) # Generate the contour map
        self.Push_validate.clicked.connect(self.extract_data)
        
        self.radius = self.Radius_box.value()
        self.threshold = self.Threshold_box.value()
        self.blur = int(self.Blur_box.currentText())
        self.sobel = int(self.Sobel_box.currentText())
        
        self.Radius_box.valueChanged.connect(self.get_modifications) # modification connexion (espacially to help batch processing GUI)
        self.Threshold_box.valueChanged.connect(self.get_modifications) # modification connexion (espacially to help batch processing GUI)
        self.Blur_box.activated.connect(self.get_modifications) # modification connexion (espacially to help batch processing GUI)
        self.Sobel_box.activated.connect(self.get_modifications) # modification connexion (espacially to help batch processing GUI)
        
        self.setWindowIcon(QtGui.QIcon('icons/filter_icon.png')) 
        self.defaultIV() # Default ImageView (when no image)
        
        try: # if data is imported from the main GUI
            self.expStack = parent.Current_stack
            self.displayExpStack(self.expStack)
        except:
            pass
        
        app = QApplication.instance()
        screen = app.screenAt(self.pos())
        geometry = screen.availableGeometry()
        
        # Position (self.move) and size (self.resize) of the main GUI on the screen
        self.move(int(geometry.width() * 0.1), int(geometry.height() * 0.1))
        self.resize(int(geometry.width() * 0.8), int(geometry.height() * 0.6))
        self.screen = screen

    def loaddata(self):
        StackLoc, StackDir = gf.getFilePathDialog("Import an image series") # Image importation
        
        self.expStack = [] # Initialization of the variable
        self.expStack = tf.TiffFile(StackLoc[0]).asarray() # Import the unit 2D array

        self.expStack = np.flip(self.expStack, 1)
        self.expStack = np.rot90(self.expStack, k=1, axes=(2, 1))

        self.displayExpStack(self.expStack)

    def get_modifications(self):
        self.radius = self.Radius_box.value()
        self.threshold = self.Threshold_box.value()
        self.blur = int(self.Blur_box.currentText())
        self.sobel = int(self.Sobel_box.currentText())

    def compute_map(self):
        self.working_stack = np.copy(self.expStack)
        
        self.progressBar.setValue(0) # Set the initial value of the Progress bar at 0
        self.progressBar.setRange(0, len(self.working_stack)-1) 
        
        # Application of the remove outlier step
        for i in range(0,len(self.expStack[:, 0, 0])): # Applique pour chaque slice les paramètres du remove outlier
        
            QApplication.processEvents()    
            self.ValSlice = i
            self.progression_bar()
        
            _, self.working_stack[i, :, :] =  gf.remove_outliers(self.expStack[i,:,:], self.radius, self.threshold)
            if self.blur != 0 :
                self.working_stack[i,:,:] =cv2.GaussianBlur(self.working_stack[i,:,:],(self.blur,self.blur),0)
            if self.sobel != 0 :
                self.grad_x = cv2.Sobel(self.working_stack[i,:,:],cv2.CV_32F,1,0,ksize=self.sobel)
                self.grad_y = cv2.Sobel(self.working_stack[i,:,:],cv2.CV_32F,0,1,ksize=self.sobel)
            
                self.working_stack[i,:,:] = cv2.addWeighted(np.absolute(self.grad_x), 0.5, np.absolute(self.grad_y), 0.5, 0)
        
        self.contour_map = np.sum(self.working_stack,0)
        self.contour_map = (self.contour_map - np.min(self.contour_map)) / (np.max(self.contour_map) - np.min(self.contour_map))

        self.displayres(self.contour_map)
        
    def extract_data(self):
        self.parent.contour_map = np.copy(self.contour_map) # Copy in the main GUI
        self.parent.StackList.append(self.contour_map) # Add the data in the stack list
        
        Combo_text = '\u2022 Contour map'
        Combo_data = self.contour_map
        self.parent.choiceBox.addItem(Combo_text, Combo_data) # Add the data in the QComboBox

        self.parent.displayDataview(self.parent.contour_map) # Display the labeled grain
        self.parent.choiceBox.setCurrentIndex(self.parent.choiceBox.count() - 1) # Show the last data in the choiceBox QComboBox

        self.parent.Info_box.ensureCursorVisible()
        self.parent.Info_box.insertPlainText("\n \u2022 Contour map.") 
        
        self.parent.Save_button.setEnabled(True)
        self.parent.Reload_button.setEnabled(True)
        self.parent.choiceBox.setEnabled(True)
        
        self.close()
        
    def progression_bar(self): # Function for the ProgressBar uses
        self.prgbar = self.ValSlice
        self.progressBar.setValue(self.prgbar)

    def defaultIV(self):
        # image series
        self.expSeries.clear()
        self.expSeries.ui.histogram.hide()
        self.expSeries.ui.roiBtn.hide()
        self.expSeries.ui.menuBtn.hide()
        
        view = self.expSeries.getView()
        view.setBackgroundColor(self.parent.color1)
        
        ROIplot = self.expSeries.getRoiPlot()
        ROIplot.setBackground(self.parent.color1)
        
        # Result map
        self.Resmap.clear()
        self.Resmap.ui.histogram.hide()
        self.Resmap.ui.roiBtn.hide()
        self.Resmap.ui.menuBtn.hide()
        
        view = self.Resmap.getView()
        view.setBackgroundColor(self.parent.color1)
        
        ROIplot = self.Resmap.getRoiPlot()
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
        
    def displayres(self, series):
        self.Resmap.ui.histogram.hide()
        self.Resmap.ui.roiBtn.hide()
        self.Resmap.ui.menuBtn.hide()
        
        view = self.Resmap.getView()
        state = view.getState()        
        self.Resmap.setImage(series) 
        view.setState(state)
        view.setBackgroundColor(self.parent.color1)
        
        self.Resmap.autoRange()
