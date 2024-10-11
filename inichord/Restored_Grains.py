# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 23:26:20 2023

@author: clanglois1
"""
import os

from inspect import getsourcefile
from os.path import abspath

import numpy as np
import pyqtgraph as pg
import tifffile as tf
import time

from PyQt5.QtWidgets import QApplication, QMessageBox, QLabel, QDialog, QVBoxLayout, QPushButton
from PyQt5 import QtCore, QtGui

from inichord import General_Functions as gf

from skimage import morphology, filters, exposure
from scipy import ndimage as ndi

path2thisFile = abspath(getsourcefile(lambda:0))
uiclass, baseclass = pg.Qt.loadUiType(os.path.dirname(path2thisFile) + "/Restored_Grains.ui")

class MainWindow(uiclass, baseclass):
    def __init__(self, parent):
        super().__init__()
        
        self.setupUi(self)
        self.parent = parent
        
        self.setWindowIcon(QtGui.QIcon('icons/Restored_Icons.png'))
        
        self.flag_restored_only = False # To display Otsu n°1 value or not
        self.flag_restored_struc = False # To display grain labels value or not
        self.flag_overlay = False # No metric at the beginning
        
        self.x = 0
        self.y = 0

        self.crosshair_v1= pg.InfiniteLine(angle=90, movable=False, pen=self.parent.color5)
        self.crosshair_h1 = pg.InfiniteLine(angle=0, movable=False, pen=self.parent.color5)
        
        self.crosshair_v2= pg.InfiniteLine(angle=90, movable=False, pen=self.parent.color5)
        self.crosshair_h2 = pg.InfiniteLine(angle=0, movable=False, pen=self.parent.color5)
        
        self.crosshair_v3= pg.InfiniteLine(angle=90, movable=False, pen=self.parent.color5)
        self.crosshair_h3 = pg.InfiniteLine(angle=0, movable=False, pen=self.parent.color5)
        
        self.crosshair_v4= pg.InfiniteLine(angle=90, movable=False, pen=self.parent.color5)
        self.crosshair_h4 = pg.InfiniteLine(angle=0, movable=False, pen=self.parent.color5)
        
        self.crosshair_v5= pg.InfiniteLine(angle=90, movable=False, pen=self.parent.color5)
        self.crosshair_h5 = pg.InfiniteLine(angle=0, movable=False, pen=self.parent.color5)
        
        self.crosshair_v6= pg.InfiniteLine(angle=90, movable=False, pen=self.parent.color5)
        self.crosshair_h6 = pg.InfiniteLine(angle=0, movable=False, pen=self.parent.color5)
        
        self.proxy1 = pg.SignalProxy(self.KADSeries.scene.sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.proxy2 = pg.SignalProxy(self.KADSeries.ui.graphicsView.scene().sigMouseClicked, rateLimit=60, slot=self.mouseClick)
        
        self.proxy3 = pg.SignalProxy(self.FiltKADSeries.scene.sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.proxy4 = pg.SignalProxy(self.FiltKADSeries.ui.graphicsView.scene().sigMouseClicked, rateLimit=60, slot=self.mouseClick)

        self.proxy5 = pg.SignalProxy(self.Otsu1Series.scene.sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.proxy6 = pg.SignalProxy(self.Otsu1Series.ui.graphicsView.scene().sigMouseClicked, rateLimit=60, slot=self.mouseClick)

        self.proxy7 = pg.SignalProxy(self.Binary1Series.scene.sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.proxy8 = pg.SignalProxy(self.Binary1Series.ui.graphicsView.scene().sigMouseClicked, rateLimit=60, slot=self.mouseClick)

        self.proxy9 = pg.SignalProxy(self.Otsu2Series.scene.sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.proxy10 = pg.SignalProxy(self.Otsu2Series.ui.graphicsView.scene().sigMouseClicked, rateLimit=60, slot=self.mouseClick)

        self.proxy11 = pg.SignalProxy(self.Binary2Series.scene.sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.proxy12 = pg.SignalProxy(self.Binary2Series.ui.graphicsView.scene().sigMouseClicked, rateLimit=60, slot=self.mouseClick)

        self.OpenData.clicked.connect(self.loaddata) # To load a map data
        self.ComputeClass_bttn.clicked.connect(self.Otsu1) # Computation of the Otsu 1 (classes creation)
        self.Threshold_bttn.clicked.connect(self.Binary_1) # Computation of the Otsu thresholding 1
        self.ComputeClass_bttn_2.clicked.connect(self.Otsu2) # Computation of the Otsu 2 (classes creation) 
        self.Threshold_bttn_2.clicked.connect(self.Binary_2) # Computation of the Otsu 2 thresholding
        self.Save_bttn.clicked.connect(self.Save_results) # To change the displayed maps
        self.Full_Run_bttn.clicked.connect(self.FullRun) # Run all parameters as defined
        
        self.PresetBox.currentTextChanged.connect(self.auto_set) # Run the defined autoset 
        self.spinBox_filter.valueChanged.connect(self.Filter_changed) # Apply filter modification 
        
        self.ChoiceBox.currentTextChanged.connect(self.ViewLabeling) # Change the displayed map
        
        self.defaultIV() # Hide the PlotWidget until a data has been loaded
        self.mouseLock.setVisible(False)
        
        try:
            if hasattr(parent, 'KAD') : # Choice of KAD if only available
                self.InitKAD_map = parent.KAD
                self.flag_KAD = True
                
            if hasattr(parent, 'contour_map') : # Choice of contour map if only avalaible
                self.InitKAD_map = parent.contour_map
                self.flag_KAD = False
                
            if hasattr(parent, 'KAD') and hasattr(parent, 'contour_map'):
                self.show_choice_message() # Choice of the map if the two are available
                
            self.StackDir = self.parent.StackDir
            self.run_init_computation()
        except:
            pass
        
        app = QApplication.instance()
        screen = app.screenAt(self.pos())
        geometry = screen.availableGeometry()
        
        # Control window position and dimensions
        self.move(int(geometry.width() * 0.05), int(geometry.height() * 0.05))
        self.resize(int(geometry.width() * 0.9), int(geometry.height() * 0.8))
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
        
    def show_choice_message(self): # Qmessage box for the try import at the initialization 
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Choice of the data")
        msg_box.setText("Data to use")
    
        btn_kad = msg_box.addButton("KAD map", QMessageBox.ActionRole)
        btn_contour = msg_box.addButton("Contour map", QMessageBox.ActionRole)

        msg_box.exec_()
    
        if msg_box.clickedButton() == btn_kad:
            self.InitKAD_map = self.parent.KAD
            self.flag_KAD = True
        elif msg_box.clickedButton() == btn_contour:
            self.InitKAD_map = self.parent.contour_map
            self.flag_KAD = False

    def data_choice(self): # Allow to apply other treatment depending if the data is a KAD one
        msg = QMessageBox.question(self, 'Restored grains', 'Is it a KAD data ?')
        if msg == QMessageBox.Yes:
            self.flag_KAD = True
        if msg == QMessageBox.No:
            self.flag_KAD = False

    def loaddata(self):
        self.defaultIV() # Hide the PlotWidget until a data has been loaded
        self.ChoiceBox.clear() # Clear the QComboBox

        self.StackLoc, self.StackDir = gf.getFilePathDialog("Open 2D map (*.tiff)") 
        
        checkimage = tf.TiffFile(self.StackLoc[0]).asarray() # Check for dimension. If 2 dimensions : 2D array. If 3 dimensions : stack of images
        if checkimage.ndim != 2: # Check if the data is a 2D array
            self.popup_message("Restored grain determination","Please import a 2D array",'icons/Restored_Icons.png')
            return
        
        self.data_choice()
        
        self.InitKAD_map = tf.TiffFile(self.StackLoc[0]).asarray()
        self.InitKAD_map = np.flip(self.InitKAD_map, 0)
        self.InitKAD_map = np.rot90(self.InitKAD_map, k=1, axes=(1, 0))
        
        self.InitKAD_map = np.nan_to_num(self.InitKAD_map) # Exclude NaN value if needed
        self.InitKAD_map = (self.InitKAD_map - np.min(self.InitKAD_map)) / (np.max(self.InitKAD_map) - np.min(self.InitKAD_map)) # Normalization step
        self.InitKAD_map = exposure.equalize_adapthist(self.InitKAD_map, kernel_size=None, clip_limit=0.01, nbins=256) # CLAHE step
        
        self.displayExpKAD(self.InitKAD_map) # Display of the map
        
        self.auto_set() # Allow different preset of parameters to be used.
        
    def run_init_computation(self):
        self.InitKAD_map = np.nan_to_num(self.InitKAD_map) # Exclude NaN value if needed
        self.InitKAD_map = (self.InitKAD_map - np.min(self.InitKAD_map)) / (np.max(self.InitKAD_map) - np.min(self.InitKAD_map)) # Normalization step
        self.InitKAD_map = exposure.equalize_adapthist(self.InitKAD_map, kernel_size=None, clip_limit=0.01, nbins=256) # CLAHE step
        
        self.displayExpKAD(self.InitKAD_map) # Display of the map
        
        self.auto_set() # Default set is "Default"

    def auto_set(self):
        self.Preset_choice = self.PresetBox.currentText()
        
        if self.Preset_choice == "Default":
            self.spinBox_filter.setValue(1) # Filter value
            self.ClassBox.setValue(4) # Number of classes for first Otsu
            self.ThresholdBox.setValue(1) # Threshold value for first Otsu
            self.ClassBox2.setValue(4) # Number of classes for second Otsu
            self.ThresholdBox_2.setValue(1) # Threshold value for last Otsu

            self.Filter_changed() # Compute the filtered map
            self.Otsu1() # Compute a first Otsu map
            self.Binary_1() # Compute a first binary map
            self.Otsu2() # Compute the second Otsu
            self.Binary_2() # Compute the final thresholding
        
        elif self.Preset_choice == "Undeformed sample":
            self.spinBox_filter.setValue(1) # Filter value
            self.ClassBox.setValue(5) # Number of classes for first Otsu
            self.ThresholdBox.setValue(1) # Threshold value for first Otsu
            self.ClassBox2.setValue(5) # Number of classes for second Otsu
            self.ThresholdBox_2.setValue(1) # Threshold value for last Otsu

            self.Filter_changed() # Compute the filtered map
            self.Otsu1() # Compute a first Otsu map
            self.Binary_1() # Compute a first binary map
            self.Otsu2() # Compute the second Otsu
            self.Binary_2() # Compute the final thresholding
            
        elif self.Preset_choice == "Slightly deformed sample":
            self.spinBox_filter.setValue(2) # Filter value
            self.ClassBox.setValue(5) # Number of classes for first Otsu
            self.ThresholdBox.setValue(1) # Threshold value for first Otsu
            self.ClassBox2.setValue(6) # Number of classes for second Otsu
            self.ThresholdBox_2.setValue(1) # Threshold value for last Otsu
            
            self.Filter_changed() # Compute the filtered map
            self.Otsu1() # Compute a first Otsu map
            self.Binary_1() # Compute a first binary map
            self.Otsu2() # Compute the second Otsu
            self.Binary_2() # Compute the final thresholding
            
        elif self.Preset_choice == "Degraded and slightly deformed sample":
            self.spinBox_filter.setValue(2) # Filter value
            self.ClassBox.setValue(5) # Number of classes for first Otsu
            self.ThresholdBox.setValue(3) # Threshold value for first Otsu
            self.ClassBox2.setValue(5) # Number of classes for second Otsu
            self.ThresholdBox_2.setValue(3) # Threshold value for last Otsu
            
            self.Filter_changed() # Compute the filtered map
            self.Otsu1() # Compute a first Otsu map
            self.Binary_1() # Compute a first binary map
            self.Otsu2() # Compute the second Otsu
            self.Binary_2() # Compute the final thresholding
            
        elif self.Preset_choice == "Deformed sample":
            self.spinBox_filter.setValue(1) # Filter value
            self.ClassBox.setValue(6) # Number of classes for first Otsu
            self.ThresholdBox.setValue(3) # Threshold value for first Otsu
            self.ClassBox2.setValue(6) # Number of classes for second Otsu
            self.ThresholdBox_2.setValue(3) # Threshold value for last Otsu
            
            self.Filter_changed() # Compute the filtered map
            self.Otsu1() # Compute a first Otsu map
            self.Binary_1() # Compute a first binary map
            self.Otsu2() # Compute the second Otsu
            self.Binary_2() # Compute the final thresholding

    def Filter_changed(self): # Allow the modification of the map with different filtering and values
        self.Filter_choice = self.FilterBox.currentText()
        
        if self.Filter_choice == "Mean filter":
            self.spinBox_filter.setRange(0,10)
            self.spinBox_filter.setSingleStep(1)
            
            self.FilteredKAD_map = np.copy(self.InitKAD_map)
            self.Filter_value = int(self.spinBox_filter.value())
            self.FilteredKAD_map = filters.gaussian(self.FilteredKAD_map, self.Filter_value)
        
        elif self.Filter_choice == "Butterworth (LP) filter":
            self.spinBox_filter.setRange(0.1,0.5)
            self.spinBox_filter.setSingleStep(0.1)
            
            self.FilteredKAD_map = np.copy(self.InitKAD_map)
            self.Filter_value = self.spinBox_filter.value()
            self.FilteredKAD_map = filters.butterworth(self.FilteredKAD_map,self.Filter_value,False,8)
            
        self.displayFilteredKAD(self.FilteredKAD_map) # Display the map after load

    def Otsu1(self): # Segment map into classes
        self.Otsu1_Value = self.ClassBox.value()

        # Segmentation of the intensities for a given number of classes
        thresholds = filters.threshold_multiotsu(self.FilteredKAD_map, classes = self.Otsu1_Value) # Definition of the threshold values
        self.regions = np.digitize(self.FilteredKAD_map, bins=thresholds) # Using the threshold values, we generate the regions.

        self.displayOtsu1(self.regions) # Display the Otsu map

    def Binary_1(self): # Binarization of the Otsu map for a given threshold level
        self.Binary1_Value = self.ThresholdBox.value()
        
        self.regions2 = np.copy(self.regions)

        var_up = np.where(self.regions >= self.Binary1_Value) # Search for every value higher or equal to threshold
        var_down = np.where(self.regions < self.Binary1_Value) # Search for every value below the threshold
        self.regions2[var_up] = 1 # Replace values by 1 ==> Binary image created
        self.regions2[var_down] = 0 # Replace values by 1 ==> Binary image created

        self.regions3 = ndi.binary_closing(self.regions2) # Closing step 
        self.binary_regions = 1-(ndi.binary_dilation(self.regions3, iterations = 1)) # Dilation to increase connectivity

        self.binary_regions = ndi.binary_opening(self.binary_regions, iterations = 2) # Opening for better result

        self.displayBinary1(self.binary_regions) # Display the thresholded map

    def Otsu2(self): # Second Otsu (on the computed distance map)
        self.Otsu2_Value = self.ClassBox2.value()
        self.binary_regions_distance = ndi.distance_transform_edt(1-self.binary_regions)
        
        # Segmentation of the intensities for a given number of classes
        thresholds = filters.threshold_multiotsu(self.binary_regions_distance, classes = self.Otsu2_Value) # Definition of the threshold values
        self.regions_distance = np.digitize(self.binary_regions_distance, bins=thresholds) # Using the threshold values, we generate the regions.

        self.displayOtsu2(self.regions_distance) # Display the Otsu map

    def Binary_2(self):
        self.Binary2_Value = self.ThresholdBox_2.value()
        
        self.binary_regions2 = np.copy(self.regions_distance)
        var = np.where(self.binary_regions2 > self.Binary2_Value) # Every pixel with value > Binary2_Value is considerered
        self.binary_regions2[var] = 5 # Replace thoses pixel by a contant value of 5
      
        self.restored_grains = np.zeros((len(self.binary_regions2),len(self.binary_regions2[0])))
        self.restored_grains[var] = 5
        self.restored_grains = ndi.binary_fill_holes(self.restored_grains).astype("bool") # Fill holes processing # Restored grain 

        var = np.where(self.restored_grains == True) # Every pixel with value > Binary2_Value is considerered
        self.binary_regions2[var] = 5
        
        # Creation of the overlay map: map and restored pixels location
        var = np.where(self.restored_grains == True)
        self.overlay_KAD_restored = np.copy(self.InitKAD_map)
        self.overlay_KAD_restored[var] = 1   
        
        # Creation of items in the QComboBox
        self.ChoiceBox.clear() 
        
        Combo_text = 'Restored grains and structure'
        Combo_data = self.binary_regions2
        self.ChoiceBox.addItem(Combo_text, Combo_data)
        
        Combo_text = 'Restored grains only'
        Combo_data = self.restored_grains
        self.ChoiceBox.addItem(Combo_text, Combo_data)

        Combo_text = 'Overlay map-restored grains'
        Combo_data = self.overlay_KAD_restored
        self.ChoiceBox.addItem(Combo_text, Combo_data)
        
        # Run the restored fraction computation
        self.Compute_restoredFrac()
        
    def Compute_restoredFrac(self):       
        Dilate = morphology.binary_dilation(self.restored_grains) # Apply a binary dilatation of the pixels
        Erode = morphology.binary_erosion(self.restored_grains) # Apply a binary dilatation of the pixels

        fraction_base = (np.sum(self.restored_grains == True)/self.restored_grains.size)*100 # Computation the restored fraction
        fraction_lowerbound = (np.sum(Erode == True)/Erode.size)*100
        fraction_upperbound = (np.sum(Dilate == True)/Dilate.size)*100

        var = [fraction_base,fraction_lowerbound,fraction_upperbound]
        self.fraction_mean = np.round(np.mean(var),2)
        self.fraction_std = np.round(np.std(var),2)
        
        self.Restored_infos.setText('Restored fraction: ' + str(self.fraction_mean) + ' \u00B1 ' + str(self.fraction_std) + ' % ')

    def ViewLabeling(self):
        self.view_choice = self.ChoiceBox.currentText()
        
        self.flag_restored_only = False # To display Otsu n°1 value or not
        self.flag_restored_struc = False # To display grain labels value or not
        self.flag_overlay = False # No metric at the beginning

        if self.view_choice == "Restored grains only":
            self.displayBinary2(self.restored_grains) # Display the KAD map after load
            self.flag_restored_only = True
            
        if self.view_choice == "Restored grains and structure":
            self.displayBinary2(self.binary_regions2) # Display the KAD map after load
            self.flag_restored_struc = True
            
        if self.view_choice == "Overlay map-restored grains":
            self.displayBinary2(self.overlay_KAD_restored) # Display the KAD map after load
            self.flag_overlay = True
            
    def FullRun(self):
        # We take all the actual parameters and we compute the data
        self.Otsu1() # Compute a first Otsu map
        self.Binary_1() # Compute a first binary map
        self.Otsu2() # Compute the second Otsu
        self.Binary_2() # Compute the final thresholding
        
    def Save_results(self):
        ti = time.strftime("%Y-%m-%d__%Hh-%Mm-%Ss")
        
        directory = "Restored_grain_determination_" + ti
        PathDir = os.path.join(self.StackDir, directory) 
        os.mkdir(PathDir)  
        
        tf.imwrite(PathDir + '/Map_CLAHE.tiff', np.rot90(np.flip(self.InitKAD_map, 0), k=1, axes=(1, 0)))
        tf.imwrite(PathDir + '/Filtered_map.tiff', np.rot90(np.flip(self.FilteredKAD_map, 0), k=1, axes=(1, 0)))  
        tf.imwrite(PathDir + '/Otsu.tiff', np.rot90(np.flip(self.regions, 0), k=1, axes=(1, 0)).astype('float32')) 
        tf.imwrite(PathDir + '/Binary_Otsu.tiff', np.rot90(np.flip(self.binary_regions, 0), k=1, axes=(1, 0)))
        tf.imwrite(PathDir + '/Otsu_2.tiff', np.rot90(np.flip(self.regions_distance, 0), k=1, axes=(1, 0)).astype('float32')) 
        tf.imwrite(PathDir + '/Threshold_2.tiff', np.rot90(np.flip(self.binary_regions2, 0), k=1, axes=(1, 0)).astype('float32'))  # Map with restored pixel = 5
        tf.imwrite(PathDir + '/Restored_grains_only.tiff', np.rot90(np.flip(self.restored_grains, 0), k=1, axes=(1, 0))) # Map with restored pixel = 5
        tf.imwrite(PathDir + '/Overlay_map_restored.tiff', np.rot90(np.flip(self.overlay_KAD_restored, 0), k=1, axes=(1, 0))) # Map with restored pixel = 5
        
        with open(PathDir + '\Restored grains determination.txt', 'w') as file:
            file.write("KAD data: " + str(self.flag_KAD))
            file.write("\nFiltering parameter: " + str(self.Filter_choice) + " - " + (str(self.Filter_value)))   
            file.write("\nOtsu n°1 class: "+ str(self.Otsu1_Value) + "\nThresholded n°1 classes (keep values equal or higher than): " + str(self.Binary1_Value))   
            file.write("\nOtsu n°2 class: "+ str(self.Otsu2_Value) + "\nThresholded n°2 classes (keep values higher than): " + str(self.Binary2_Value))     
            file.write("\nRestored fraction: "+ str(self.fraction_mean) + ' \u00B1 ' + str(self.fraction_std) + ' % ')

        self.popup_message("Restored grain determination","Saving process is over.",'icons/Restored_Icons.png')

    def displayExpKAD(self, series): # Display of initial KAD map
        self.KADSeries.addItem(self.crosshair_v1, ignoreBounds=True)
        self.KADSeries.addItem(self.crosshair_h1, ignoreBounds=True) 
        
        self.KADSeries.ui.histogram.hide()
        self.KADSeries.ui.roiBtn.hide()
        self.KADSeries.ui.menuBtn.hide()
        
        view = self.KADSeries.getView()
        state = view.getState()        
        self.KADSeries.setImage(series) 
        view.setState(state)
        view.setBackgroundColor(self.parent.color1)
        
        self.KADSeries.autoRange()
        
    def displayFilteredKAD(self, series): # Display the filtered KAD map
        self.FiltKADSeries.addItem(self.crosshair_v2, ignoreBounds=True)
        self.FiltKADSeries.addItem(self.crosshair_h2, ignoreBounds=True) 
        
        self.FiltKADSeries.ui.histogram.hide()
        self.FiltKADSeries.ui.roiBtn.hide()
        self.FiltKADSeries.ui.menuBtn.hide()
        
        view = self.FiltKADSeries.getView()
        state = view.getState()        
        self.FiltKADSeries.setImage(series) 
        view.setState(state)
        view.setBackgroundColor(self.parent.color1)
        
        # self.FiltKADSeries.autoRange()
        
    def displayOtsu1(self, series): # Display Otsu 1 regions
        self.Otsu1Series.addItem(self.crosshair_v3, ignoreBounds=True)
        self.Otsu1Series.addItem(self.crosshair_h3, ignoreBounds=True) 
        
        self.Otsu1Series.ui.histogram.show()
        self.Otsu1Series.ui.roiBtn.hide()
        self.Otsu1Series.ui.menuBtn.hide()
        
        view = self.Otsu1Series.getView()
        state = view.getState()        
        self.Otsu1Series.setImage(series) 
        view.setState(state)
        view.setBackgroundColor(self.parent.color1)
        
        histplot = self.Otsu1Series.getHistogramWidget()
        histplot.setBackground(self.parent.color1)
        
        histplot.region.setBrush(pg.mkBrush(self.parent.color5 + (120,)))
        histplot.region.setHoverBrush(pg.mkBrush(self.parent.color5 + (60,)))
        histplot.region.pen = pg.mkPen(self.parent.color5)
        histplot.region.lines[0].setPen(pg.mkPen(self.parent.color5, width=2))
        histplot.region.lines[1].setPen(pg.mkPen(self.parent.color5, width=2))
        histplot.fillHistogram(color = self.parent.color5)        
        histplot.autoHistogramRange()
        
        self.Otsu1Series.setColorMap(pg.colormap.get('viridis'))
        
    def displayOtsu2(self, series): # Display the distance Otsu regions after Binary1
        self.Otsu2Series.addItem(self.crosshair_v4, ignoreBounds=True)
        self.Otsu2Series.addItem(self.crosshair_h4, ignoreBounds=True) 
        
        self.Otsu2Series.ui.histogram.show()
        self.Otsu2Series.ui.roiBtn.hide()
        self.Otsu2Series.ui.menuBtn.hide()
        
        view = self.Otsu2Series.getView()
        state = view.getState()        
        self.Otsu2Series.setImage(series) 
        view.setState(state)
        view.setBackgroundColor(self.parent.color1)
        
        histplot = self.Otsu2Series.getHistogramWidget()
        histplot.setBackground(self.parent.color1)
        
        histplot.region.setBrush(pg.mkBrush(self.parent.color5 + (120,)))
        histplot.region.setHoverBrush(pg.mkBrush(self.parent.color5 + (60,)))
        histplot.region.pen = pg.mkPen(self.parent.color5)
        histplot.region.lines[0].setPen(pg.mkPen(self.parent.color5, width=2))
        histplot.region.lines[1].setPen(pg.mkPen(self.parent.color5, width=2))
        histplot.fillHistogram(color = self.parent.color5)        
        histplot.autoHistogramRange()
        
        self.Otsu2Series.setColorMap(pg.colormap.get('viridis'))
                
    def displayBinary1(self, series): # Display first thresholding response
        self.Binary1Series.addItem(self.crosshair_v5, ignoreBounds=True)
        self.Binary1Series.addItem(self.crosshair_h5, ignoreBounds=True) 
        
        self.Binary1Series.ui.histogram.hide()
        self.Binary1Series.ui.roiBtn.hide()
        self.Binary1Series.ui.menuBtn.hide()
        
        view = self.Binary1Series.getView()
        state = view.getState()        
        self.Binary1Series.setImage(series) 
        view.setState(state)
        view.setBackgroundColor(self.parent.color1)
        
    def displayBinary2(self, series): # Display final results
        self.Binary2Series.addItem(self.crosshair_v6, ignoreBounds=True)
        self.Binary2Series.addItem(self.crosshair_h6, ignoreBounds=True) 
        
        self.Binary2Series.ui.histogram.show()
        self.Binary2Series.ui.roiBtn.hide()
        self.Binary2Series.ui.menuBtn.hide()
        
        view = self.Binary2Series.getView()
        state = view.getState()        
        self.Binary2Series.setImage(series) 
        view.setState(state)
        view.setBackgroundColor(self.parent.color1)
        
        histplot = self.Binary2Series.getHistogramWidget()
        histplot.setBackground(self.parent.color1)
        
        histplot.region.setBrush(pg.mkBrush(self.parent.color5 + (120,)))
        histplot.region.setHoverBrush(pg.mkBrush(self.parent.color5 + (60,)))
        histplot.region.pen = pg.mkPen(self.parent.color5)
        histplot.region.lines[0].setPen(pg.mkPen(self.parent.color5, width=2))
        histplot.region.lines[1].setPen(pg.mkPen(self.parent.color5, width=2))
        histplot.fillHistogram(color = self.parent.color5)        
        histplot.autoHistogramRange()
        
        self.Binary2Series.setColorMap(pg.colormap.get('CET-D7'))

    def defaultIV(self):
        # KADSeries: Initial KAD
        self.KADSeries.ui.histogram.hide()
        self.KADSeries.ui.roiBtn.hide()
        self.KADSeries.ui.menuBtn.hide()
        
        view = self.KADSeries.getView()
        view.setBackgroundColor(self.parent.color1)
        
        # FiltKADSeries: KAD after filtering
        self.FiltKADSeries.ui.histogram.hide()
        self.FiltKADSeries.ui.roiBtn.hide()
        self.FiltKADSeries.ui.menuBtn.hide()
        
        view = self.FiltKADSeries.getView()
        view.setBackgroundColor(self.parent.color1)
        
        # Otsu1Series: Otsu n°1 classes definition
        self.Otsu1Series.ui.histogram.hide()
        self.Otsu1Series.ui.roiBtn.hide()
        self.Otsu1Series.ui.menuBtn.hide()
        
        view = self.Otsu1Series.getView()
        view.setBackgroundColor(self.parent.color1)
        
        # Binary1Series: Otsu n°1 after binarisation
        self.Binary1Series.ui.histogram.hide()
        self.Binary1Series.ui.roiBtn.hide()
        self.Binary1Series.ui.menuBtn.hide()
        
        view = self.Binary1Series.getView()
        view.setBackgroundColor(self.parent.color1)
        
        # Otsu2Series: Otsu n°2 classes definition
        self.Otsu2Series.ui.histogram.hide()
        self.Otsu2Series.ui.roiBtn.hide()
        self.Otsu2Series.ui.menuBtn.hide()
        
        view = self.Otsu2Series.getView()
        view.setBackgroundColor(self.parent.color1)
        
        # Binary2Series: Otsu n°2 after binarisation
        self.Binary2Series.ui.histogram.hide()
        self.Binary2Series.ui.roiBtn.hide()
        self.Binary2Series.ui.menuBtn.hide()
        
        view = self.Binary2Series.getView()
        view.setBackgroundColor(self.parent.color1)
        
    def mouseMoved(self, e):
        pos = e[0]
        sender = self.sender()
  
        if not self.mouseLock.isChecked():
            if self.KADSeries.view.sceneBoundingRect().contains(pos)\
                or self.FiltKADSeries.view.sceneBoundingRect().contains(pos)\
                or self.Otsu1Series.view.sceneBoundingRect().contains(pos)\
                or self.Binary1Series.view.sceneBoundingRect().contains(pos)\
                or self.Otsu2Series.view.sceneBoundingRect().contains(pos)\
                or self.Binary2Series.view.sceneBoundingRect().contains(pos):   
                
                if sender == self.proxy1:
                    item = self.KADSeries.view
                elif sender == self.proxy3:
                    item = self.FiltKADSeries.view
                elif sender == self.proxy5:
                    item = self.Otsu1Series.view
                elif sender == self.proxy7:
                    item = self.Binary1Series.view
                elif sender == self.proxy9:
                    item = self.Otsu2Series.view
                else:
                    item = self.Binary2Series.view
                
                mousePoint = item.mapSceneToView(pos) 
                     
                self.crosshair_v1.setPos(mousePoint.x())
                self.crosshair_h1.setPos(mousePoint.y())
                
                self.crosshair_v2.setPos(mousePoint.x())
                self.crosshair_h2.setPos(mousePoint.y())
                
                self.crosshair_v3.setPos(mousePoint.x())
                self.crosshair_h3.setPos(mousePoint.y())
                
                self.crosshair_v4.setPos(mousePoint.x())
                self.crosshair_h4.setPos(mousePoint.y())
                
                self.crosshair_v5.setPos(mousePoint.x())
                self.crosshair_h5.setPos(mousePoint.y())
                
                self.crosshair_v6.setPos(mousePoint.x())
                self.crosshair_h6.setPos(mousePoint.y())

            self.x = int(mousePoint.x())
            self.y = int(mousePoint.y())
            
            self.printClick(self.x, self.y, sender)
    
    def mouseClick(self, e):
        pos = e[0]
        
        self.mouseLock.toggle()
        
        fromPosX = pos.scenePos()[0]
        fromPosY = pos.scenePos()[1]
        
        posQpoint = QtCore.QPointF()
        posQpoint.setX(fromPosX)
        posQpoint.setY(fromPosY)

        if self.KADSeries.view.sceneBoundingRect().contains(posQpoint):
                
            item = self.KADSeries.view
            mousePoint = item.mapSceneToView(posQpoint) 

            self.crosshair_v1.setPos(mousePoint.x())
            self.crosshair_h1.setPos(mousePoint.y())
                 
            self.x = int(mousePoint.x())
            self.y = int(mousePoint.y())
            
    def printClick(self, x, y, sender):
        try: # Display values of Otsu1 and Otsu2
            self.Otsu1_label.setText("Otsu classes n°1: " + str(self.regions[x, y]))
            self.Otsu2_label.setText("Otsu classes n°2: " + str(self.regions_distance[x, y]))
        except:
            pass
            
        if self.flag_restored_only == True: # Display value of restored
            try:
                self.Threshold2_label.setText("Threshold n°2: " + str(self.restored_grains[x, y]))
            except:
                pass
            
        if self.flag_restored_struc == True: # Display value of restored (threshold)
            try:
                self.Threshold2_label.setText("Threshold n°2: " + str(self.binary_regions2[x, y]))        
            except:
                pass
        
        if self.flag_overlay == True: # Display value of KAD (with restored as 1)
            try:
                self.Threshold2_label.setText("Map overlay: " + str(np.round(self.overlay_KAD_restored[x, y],2)))        
            except:
                pass