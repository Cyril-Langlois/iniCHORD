# -*- coding: utf-8 -*-

import os

from inspect import getsourcefile
from os.path import abspath

import numpy as np
import pyqtgraph as pg
import tifffile as tf
import time

from PyQt5.QtWidgets import QApplication, QLabel, QDialog, QVBoxLayout, QPushButton
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QMessageBox

from inichord import General_Functions as gf

from skimage import morphology, filters, exposure
from skimage.measure import label, regionprops
from skimage.segmentation import expand_labels
from scipy import ndimage as ndi

path2thisFile = abspath(getsourcefile(lambda:0))
uiclass, baseclass = pg.Qt.loadUiType(os.path.dirname(path2thisFile) + "/Grain_Treatment.ui")

class MainWindow(uiclass, baseclass):
    def __init__(self, parent):
        super().__init__()
        
        self.setupUi(self)
        self.parent = parent
        
        self.setWindowIcon(QtGui.QIcon('icons/Grain_Icons.png'))
                
        self.flag_info = False # To display Otsu n°1 value or not
        self.flag_info_labels = False # To display grain labels value or not
        self.flag_info_labels_metric = False # No metric at the beginning
        self.flag_PixelSize = False # No consideration of pixel size at the opening
        self.flag_info_overlay = False # No consideration of the overlay map at the opening
        self.flag_info_filtered = False # No consideration of excluded pixels (pxls) at the opening
        self.flag_info_filtered2 = False # No consideration of excluded pixels (µm) at the opening
        self.flag_Border = False #No consideration of excluded borders at the opening
        
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
        
        self.proxy1 = pg.SignalProxy(self.KADSeries.scene.sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.proxy2 = pg.SignalProxy(self.KADSeries.ui.graphicsView.scene().sigMouseClicked, rateLimit=60, slot=self.mouseClick)
        
        self.proxy3 = pg.SignalProxy(self.FiltKADSeries.scene.sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.proxy4 = pg.SignalProxy(self.FiltKADSeries.ui.graphicsView.scene().sigMouseClicked, rateLimit=60, slot=self.mouseClick)

        self.proxy5 = pg.SignalProxy(self.Otsu1Series.scene.sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.proxy6 = pg.SignalProxy(self.Otsu1Series.ui.graphicsView.scene().sigMouseClicked, rateLimit=60, slot=self.mouseClick)

        self.proxy7 = pg.SignalProxy(self.Binary1Series.scene.sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.proxy8 = pg.SignalProxy(self.Binary1Series.ui.graphicsView.scene().sigMouseClicked, rateLimit=60, slot=self.mouseClick)

        self.proxy9 = pg.SignalProxy(self.LabelsSeries.scene.sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.proxy10 = pg.SignalProxy(self.LabelsSeries.ui.graphicsView.scene().sigMouseClicked, rateLimit=60, slot=self.mouseClick)

        self.OpenData.clicked.connect(self.loaddata) # Load a data
        self.ComputeClass_bttn.clicked.connect(self.Otsu1) # Computation of the Otsu (classes creation)
        self.Threshold_bttn.clicked.connect(self.Binary_1) # Computation of the Otsu thresholding
        self.ComputeLabels_bttn.clicked.connect(self.Grain_labeling) # Labeling of grains
        self.Save_bttn.clicked.connect(self.Save_results) # Saving process (processing steps, results, infos)
        self.Push_validate.clicked.connect(self.validate_data)
        self.Grainsize_bttn.clicked.connect(self.labels_computation)
        self.exclude_border_bttn.clicked.connect(self.exclude_border) # Exclude labels with borders
                
        self.PixelSize_edit.setText("Add a pixel size (µm).")
        self.PixelSize_edit.editingFinished.connect(self.changeText) # Take into account the pixel size
        
        self.PresetBox.currentTextChanged.connect(self.auto_set) # Allow different pre-set to be used for computation
        self.spinBox_filter.valueChanged.connect(self.Filter_changed) # Change initial filtering
        self.Den_spinbox.valueChanged.connect(self.denoise_labels) # Change denoising value of the labels
        self.Filter_labelBox.valueChanged.connect(self.Filter_labels) # To exclude small grains
        self.ChoiceBox.currentTextChanged.connect(self.ViewLabeling) # Change displayed map
        
        self.Grainsize_bttn.setEnabled(False)
        
        self.label_filterdiameter.setText("Exclude \u2300 < x(pxls)")
        
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

    def loaddata(self): # Opening of a 2D array (KAD array)
        self.defaultIV() # Hide the PlotWidget until a data has been loaded
        self.ChoiceBox.clear() 
        
        self.StackLoc, self.StackDir = gf.getFilePathDialog("Open map (*.tiff)") 
        
        checkimage = tf.TiffFile(self.StackLoc[0]).asarray() # Check for dimension. If 2 dimensions : 2D array. If 3 dimensions : stack of images
        if checkimage.ndim != 2: # Check if the data is a KAD map (2D array)
            self.popup_message("Grain segmentation","Please import a 2D array",'icons/Grain_Icons.png')
            return
        
        self.data_choice()
        
        self.InitKAD_map = tf.TiffFile(self.StackLoc[0]).asarray()
        self.InitKAD_map = np.flip(self.InitKAD_map, 0)
        self.InitKAD_map = np.rot90(self.InitKAD_map, k=1, axes=(1, 0))
        
        self.InitKAD_map = np.nan_to_num(self.InitKAD_map) # Exclude NaN value if needed
        self.InitKAD_map = (self.InitKAD_map - np.min(self.InitKAD_map)) / (np.max(self.InitKAD_map) - np.min(self.InitKAD_map)) # Normalization step
        self.InitKAD_map = exposure.equalize_adapthist(self.InitKAD_map, kernel_size=None, clip_limit=0.01, nbins=256) # CLAHE step
        
        self.displayExpKAD(self.InitKAD_map) # Display of the map
        
        # If data is too large, hence the labeling step is not performed to avoid long computation prior checking if the data is good to be labeled
        if (len(self.InitKAD_map) * len(self.InitKAD_map[0])) < 1_000_000 :
            self.auto_set()
        else:
            self.high_auto_set()

    def run_init_computation(self): # Run a first analysis automatically
        self.InitKAD_map = np.nan_to_num(self.InitKAD_map) # Exclude NaN value if needed
        self.InitKAD_map = (self.InitKAD_map - np.min(self.InitKAD_map)) / (np.max(self.InitKAD_map) - np.min(self.InitKAD_map)) # Normalization step
        self.InitKAD_map = exposure.equalize_adapthist(self.InitKAD_map, kernel_size=None, clip_limit=0.01, nbins=256) # CLAHE step
        
        self.displayExpKAD(self.InitKAD_map) # Display of the map
        
        # If data is too large, hence the labeling step is not performed to avoid long computation prior checking if the data is good to be labeled
        if (len(self.InitKAD_map) * len(self.InitKAD_map[0])) < 1_000_000 :
            self.auto_set()
        else:
            self.high_auto_set()

    def auto_set(self): # Allow different pre-set to be used
        self.Preset_choice = self.PresetBox.currentText()

        if self.Preset_choice == "Undeformed sample":
            self.spinBox_filter.setValue(0.005)
            self.ClassBox.setValue(4)
            self.ThresholdBox.setValue(1)

            self.Filter_changed() # Compute the filtered map
            self.Otsu1() # Compute a first Otsu map
            self.Binary_1() # Compute a first binary map
            self.Grain_labeling() # Compute a labeled map (undenoised)      
            
        elif self.Preset_choice == "Slighlty deformed sample":
            self.spinBox_filter.setValue(0.02)
            self.ClassBox.setValue(5)
            self.ThresholdBox.setValue(2)
            
            self.Filter_changed() # Compute the filtered map
            self.Otsu1() # Compute a first Otsu map
            self.Binary_1() # Compute a first binary map
            self.Grain_labeling() # Compute a labeled map (undenoised)
            
        elif self.Preset_choice == "Heavily deformed sample":
            self.spinBox_filter.setValue(0.025)
            self.ClassBox.setValue(6)
            self.ThresholdBox.setValue(3)
            
            self.Filter_changed() # Compute the filtered map
            self.Otsu1() # Compute a first Otsu map
            self.Binary_1() # Compute a first binary map
            self.Grain_labeling() # Compute a labeled map (undenoised)

    def high_auto_set(self):
            self.popup_message("Grain boundaries determination","Labeling step is not apply to limit calculation times (large data map).",'icons/Grain_Icons.png')
            
            self.spinBox_filter.setValue(0.005)
            self.ClassBox.setValue(3)
            self.ThresholdBox.setValue(1)

            self.Filter_changed() # Compute the filtered map
            self.Otsu1() # Compute a first Otsu map
            self.Binary_1() # Compute a first binary map
            
    def changeText(self): # Importation of pixel size value after writing in the GUI
        self.pixelSize = float(self.PixelSize_edit.text())
        self.flag_PixelSize = True # Consideration of pixel size

    def Filter_changed(self): # KAD filtering processes
        self.Filter_choice = self.FilterBox.currentText()
        
        if self.Filter_choice == "Butterworth (HP) filter":
            self.spinBox_filter.setRange(0.005,0.5)
            self.spinBox_filter.setSingleStep(0.005)
            
            self.FilteredKAD_map = np.copy(self.InitKAD_map)
            self.Filter_value = self.spinBox_filter.value()
            self.FilteredKAD_map = filters.butterworth(self.FilteredKAD_map,self.Filter_value,True,8)
            
        elif self.Filter_choice == "Mean filter":
            self.spinBox_filter.setRange(0,10)
            self.spinBox_filter.setSingleStep(1)
            
            self.FilteredKAD_map = np.copy(self.InitKAD_map)
            self.Filter_value = self.spinBox_filter.value()
            self.FilteredKAD_map = filters.gaussian(self.FilteredKAD_map, self.Filter_value)
            
        elif self.Filter_choice == "Top-hat":
            self.spinBox_filter.setRange(1,20)
            self.spinBox_filter.setSingleStep(1)
            self.FilteredKAD_map = np.copy(self.InitKAD_map)
            self.Filter_value = self.spinBox_filter.value()
            footprint = morphology.disk(self.Filter_value)
            self.FilteredKAD_map = morphology.white_tophat(self.FilteredKAD_map, footprint)

        self.displayFilteredKAD(self.FilteredKAD_map) # Display the map after load

    def Otsu1(self): # Segment map into classes
        self.Otsu1_Value = self.ClassBox.value()

        # Segmentation of the intensities for a given number of classes
        thresholds = filters.threshold_multiotsu(self.FilteredKAD_map, classes = self.Otsu1_Value) # Definition of the threshold values
        self.regions = np.digitize(self.FilteredKAD_map, bins=thresholds) # Using the threshold values, we generate the regions.

        self.flag_info = True 
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

        self.displayBinary1(self.binary_regions) # Display the thresholded map

    def Grain_labeling(self):
        self.label_img = label(self.binary_regions, connectivity=2) # Labeling of the thresholded map
        self.d = np.zeros(np.amax(self.label_img) + 1) # Array of 0 value to store the area of grains
    
        self.flag_DenLabels = False
        self.labels_computation()
        
        self.Grainsize_bttn.setEnabled(True)
        
    def denoise_labels(self): # Fill labels with holes inside
        # Expansion of labels ==> Get rid of grains boundaries and dust or holes
        distance_val = self.Den_spinbox.value()
        
        self.flag_DenLabels = True
        self.label_img_refined = expand_labels(self.label_img, distance=distance_val)
        
        self.displaylabels(self.label_img_refined) # Display the labeled image

    def labels_computation(self):
        # Computation of equivalent diameters
        if self.flag_DenLabels == False:
            self.Var_Labels = np.copy(self.label_img)
        elif self.flag_DenLabels == True:
            self.Var_Labels = np.copy(self.label_img_refined)
            
        self.img_diameter = np.zeros(self.label_img.shape) #Array of labeled grains with associated equivalent diameters
        
        self.img_area = np.zeros(self.label_img.shape) #Array of labeled grains with associated area (pxls)
        self.form_factor = np.zeros(np.amax(self.label_img) + 1) # Form factor
        self.img_formfactor = np.zeros(self.label_img.shape) # Form factor map
        
        for i,region in enumerate(regionprops(self.Var_Labels)):
            
            i=i+1
            self.d[i] = region.area_filled
            
            major_axis_length = region.major_axis_length
            minor_axis_length = region.minor_axis_length
            if minor_axis_length == 0:
                self.form_factor[i] = 0
            else:
                self.form_factor[i] = major_axis_length / minor_axis_length
                
            var = np.where(self.Var_Labels == i)
            self.img_diameter[var] = self.d[i]
            self.img_area[var] = self.d[i]
            self.img_formfactor[var] = self.form_factor[i]      
            
        var = np.where(self.img_diameter == 0)
        
        self.img_diameter = ((2*np.sqrt(self.img_diameter/np.pi)) + 2)
        self.img_diameter[var] = 0
        
        # Correction of labels and diameter values
        var = np.where(self.img_diameter == 0)

        x = 1
        y = 2

        self.Corrected_img_diameter = np.copy(self.img_diameter) # Map of the grains (diameter value map) with border corrected
        self.Corrected_label_img = np.copy(self.Var_Labels) # Map of the grains (labeled value map) with border corrected

        for i in range(len(var[0])):
            varx = var[0][i] # X position of the pixel
            vary = var[1][i] # X position of the pixel
                
            var_diameter = self.img_diameter[varx-x:varx+y,vary-x:vary+y]
            var_label = self.Var_Labels[varx-x:varx+y,vary-x:vary+y]
                
            if var_diameter.size != 0:
          
                value_diameter = np.max(var_diameter)
                value_label = np.max(var_label)
                
                self.Corrected_img_diameter[varx,vary] = value_diameter
                self.Corrected_label_img[varx,vary] = value_label

        # Creation of the overlay map (KAD and grain boundaries)
        self.overlay_KAD_GB = np.copy(self.InitKAD_map)
        self.overlay_KAD_GB[var] = 1   

        # Creation of items in the QComboBox
        self.ChoiceBox.clear() 

        Combo_text = 'Labeled grains'
        Combo_data = self.Corrected_label_img
        self.ChoiceBox.addItem(Combo_text, Combo_data)

        Combo_text = 'Grains diameter (pxls)'
        Combo_data = self.Corrected_img_diameter
        self.ChoiceBox.addItem(Combo_text, Combo_data)
        
        self.flag_info_labels = False
        self.displaylabels(self.Corrected_label_img) # Display the labeled image
        
        if self.flag_PixelSize == True:
            self.Corrected_img_diameter_metric = np.copy(self.Corrected_img_diameter)
            self.Corrected_img_diameter_metric = self.Corrected_img_diameter_metric * self.pixelSize
            
            Combo_text = 'Grains diameter (µm)'
            Combo_data = self.Corrected_img_diameter_metric
            self.ChoiceBox.addItem(Combo_text, Combo_data)
            
            self.img_area_metric = np.copy(self.img_area)
            self.img_area_metric = self.img_area_metric * self.pixelSize
            
        Combo_text = 'Overlay map-GB'
        Combo_data = self.overlay_KAD_GB
        self.ChoiceBox.addItem(Combo_text, Combo_data)
        
        self.Filter_labels() # After labels exclusion
        self.displaylabels(self.Corrected_label_img) # Display the labeled image
            
        # Finished message
        self.popup_message("Grain boundaries determination","Properties computed.",'icons/Grain_Icons.png')
        
    def extract_value_list(self): # Extract informations       
        # Equivalent diameter data 
        self.extract_diameter = np.copy(self.d) # Area list
        self.extract_diameter = (2*np.sqrt(self.extract_diameter/np.pi)) + 2 # Conversion in diameter

        self.filtered_diameter = np.copy(self.extract_diameter) # Copy
        var = np.where(self.filtered_diameter <= self.Filter_labelValue) # Search for filtering
        self.filtered_diameter[var] = 0 # Replace value by 0
        
        # Area data 
        self.area_list_pxls = np.copy(self.d)
        self.Filtered_area_list_pxls = np.copy(self.area_list_pxls)
        self.Filtered_area_list_pxls[var] = 0
                
        # Form factor data
        self.form_factor_list = np.copy(self.form_factor)
        self.Filtered_form_factor_list = np.copy(self.form_factor_list)
        self.Filtered_form_factor_list[var] = 0
                
        if self.flag_PixelSize == True:
            self.extract_diameter_metric = self.extract_diameter * self.pixelSize
            self.filtered_diameter_metric = self.filtered_diameter * self.pixelSize
            
            # Area data
            self.area_list_metric = np.copy(self.d) * self.pixelSize
            self.Filtered_area_list_metric = np.copy(self.area_list_metric)
            self.Filtered_area_list_metric[var] = 0
            
        if self.flag_Border == True:
            self.area_list_pxls = np.delete(self.area_list_pxls, self.label_border)
            self.Filtered_area_list_pxls = np.delete(self.Filtered_area_list_pxls, self.label_border)
            self.Filtered_area_list_pxls = self.Filtered_area_list_pxls[self.Filtered_area_list_pxls != 0]

            self.extract_diameter = np.delete(self.extract_diameter, self.label_border)
            self.filtered_diameter = np.delete(self.filtered_diameter, self.label_border)
            self.filtered_diameter = self.filtered_diameter[self.filtered_diameter != 0]
            
            self.form_factor_list = np.delete(self.form_factor_list, self.label_border)
            self.Filtered_form_factor_list = np.delete(self.Filtered_form_factor_list, self.label_border)
            self.Filtered_form_factor_list = self.Filtered_form_factor_list[self.Filtered_form_factor_list != 0]
            
            if self.flag_PixelSize == True:
                self.area_list_metric = np.delete(self.area_list_metric, self.label_border)
                self.Filtered_area_list_metric = np.delete(self.Filtered_area_list_metric, self.label_border)
                self.Filtered_area_list_metric = self.Filtered_area_list_metric[self.Filtered_area_list_metric != 0]

                self.extract_diameter_metric = np.delete(self.extract_diameter_metric, self.label_border)
                self.filtered_diameter_metric = np.delete(self.filtered_diameter_metric, self.label_border)
                self.filtered_diameter_metric = self.filtered_diameter_metric[self.filtered_diameter_metric != 0]

        else:
            self.Filtered_area_list_pxls = self.Filtered_area_list_pxls[self.Filtered_area_list_pxls != 0]
            self.filtered_diameter = self.filtered_diameter[self.filtered_diameter != 0]
            self.Filtered_form_factor_list = self.Filtered_form_factor_list[self.Filtered_form_factor_list != 0]
            
            if self.flag_PixelSize == True:
                self.Filtered_area_list_metric = self.Filtered_area_list_metric[self.Filtered_area_list_metric != 0]
                self.filtered_diameter_metric = self.filtered_diameter_metric[self.filtered_diameter_metric != 0]

    def ViewLabeling(self):
        self.view_choice = self.ChoiceBox.currentText()
        
        self.flag_info_labels = False # To display grain labels value or not
        self.flag_info_labels_metric = False # No metric at the beginning
        self.flag_info_overlay = False # No consideration of the overlay map at the opening
        self.flag_info_filtered = False # No consideration of pixel size at the opening
        self.flag_info_filtered2 = False # No consideration of the overlay map at the opening
        
        if self.view_choice == "Labeled grains":
            self.displaylabels(self.Corrected_label_img) # Display the map after load
            self.flag_info_labels = False
            
        if self.view_choice == "Grains diameter (pxls)":
            self.displaylabels(self.Corrected_img_diameter) # Display the map after load
            self.flag_info_labels = True
            
        if self.view_choice == "Grains diameter (µm)":
            self.displaylabels(self.Corrected_img_diameter_metric) # Display the map after load
            self.flag_info_labels_metric = True
            
        if self.view_choice == "Overlay map-GB":
            self.displaylabels(self.overlay_KAD_GB) # Display the map after load
            self.flag_info_overlay = True
            
        if self.view_choice == "Grain diameter pxls (excluded \u2300)":
            self.displaylabels(self.filter_labeldiameter) # Display the map after load
            self.flag_info_filtered = True

        if self.view_choice == "Grain diameter µm (excluded \u2300)":
            self.displaylabels(self.filter_labeldiameter_metric) # Display the map after load
            self.flag_info_filtered2 = True

    def Filter_labels(self): # Labels filtering (== 0 if value <= the given excluding value)
        self.Filter_labelValue = self.Filter_labelBox.value()
        
        var = np.where(self.Corrected_img_diameter <= self.Filter_labelValue)
        
        # Equivalent diameter after exclusion of small labels
        self.filter_labeldiameter = np.copy(self.Corrected_img_diameter)
        self.filter_labeldiameter[var] = 0
                
        try : # Try to find if the data already exist to rewrite it
            index = self.ChoiceBox.findText('Grain diameter pxls (excluded \u2300)')
            self.ChoiceBox.removeItem(index)
            
            Combo_text = 'Grain diameter pxls (excluded \u2300)'
            Combo_data = self.filter_labeldiameter
            self.ChoiceBox.addItem(Combo_text, Combo_data)
        except:
            pass
        
        if self.flag_PixelSize == True: 
            self.filter_labeldiameter_metric = np.copy(self.Corrected_img_diameter_metric)
            self.filter_labeldiameter_metric[var] = 0
            
            try : # Try to find if the data already exist to rewrite it
                index = self.ChoiceBox.findText('Grain diameter µm (excluded \u2300)')
                self.ChoiceBox.removeItem(index)
                
                Combo_text = 'Grain diameter µm (excluded \u2300)'
                Combo_data = self.filter_labeldiameter_metric
                self.ChoiceBox.addItem(Combo_text, Combo_data)
            except:
                pass
            
        self.displaylabels(self.filter_labeldiameter) # Display the map after load

        # Area after exclusion of small labels in pxls
        self.filter_area_pxls = np.copy(self.img_area)
        self.filter_area_pxls[var] = 0
        
        if self.flag_PixelSize == True: 
            # Area after exclusion of small labels in µm
            self.filter_area_metric = np.copy(self.img_area_metric)
            self.filter_area_metric[var] = 0
        
        # Form factor after exclusion of small labels
        self.filter_form_factor = np.copy(self.img_formfactor)
        self.filter_form_factor[var] = 0

    def exclude_border(self): 
        self.flag_Border = True
        # Get position (pxls) of borders
        border_positions = self.get_border_positions(self.Var_Labels) 
        # Get labels value
        border_intensities = self.get_border_intensities(self.Var_Labels, border_positions)
        self.label_border = np.unique(border_intensities)
        # Replace labels with 0
        self.Var_Labels_cleaned = self.zero_out_matching_intensities(self.Var_Labels, border_intensities) 
        # Get all the 0 value positions 
        zero_positions = self.get_zero_positions(self.Var_Labels_cleaned)

        # Filter application on the other main maps ==> Pixel maps
        # Area map (pixels)
        self.img_area_cleaned = self.zero_out_positions(self.img_area, zero_positions)
        # Equivalent diameter (pixels)
        self.Corrected_img_diameter_cleaned = self.zero_out_positions(self.Corrected_img_diameter, zero_positions)
        # Form factor
        self.img_formfactor_cleaned = self.zero_out_positions(self.img_formfactor, zero_positions)
        
        # Filter application on the other main maps ==> metric maps
        if self.flag_PixelSize == True: 
            # Area map (µm)
            self.img_area_metric_cleaned = self.zero_out_positions(self.img_area_metric, zero_positions)
            # Equivalent diameter (µm)
            self.Corrected_img_diameter_metric_cleaned = self.zero_out_positions(self.Corrected_img_diameter_metric, zero_positions)
            
        # Filter application on the other main maps ==> pixel maps
        # Filtered area map (pixels)
        self.filter_area_pxls_cleaned = self.zero_out_positions(self.filter_area_pxls, zero_positions)
        # Filtered Equivalent diameter (pixels)
        self.filter_labeldiameter_cleaned = self.zero_out_positions(self.filter_labeldiameter, zero_positions)
        # Filtered form factor
        self.filter_form_factor_cleaned = self.zero_out_positions(self.filter_form_factor, zero_positions)
        
        # Filter application on the other main maps ==> metric maps
        if self.flag_PixelSize == True: 
            # Filtered area map (µm)
            self.filter_area_metric_cleaned = self.zero_out_positions(self.filter_area_metric, zero_positions)
            # Filtered Equivalent diameter (µm)
            self.filter_labeldiameter_metric_cleaned = self.zero_out_positions(self.filter_labeldiameter_metric, zero_positions)

        # Finished message
        self.popup_message("Grain boundaries determination","Border excluded.",'icons/Grain_Icons.png')

    def get_border_positions(self,image):# Fonction pour obtenir les positions des pixels des bordures
        border_positions = []
        rows, cols = image.shape
        for i in range(rows):
            for j in range(cols):
                if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
                    border_positions.append((i, j))
        return border_positions

    def get_border_intensities(self,image, border_positions):# Fonction pour obtenir les intensités des pixels des bordures
        border_intensities = [image[i, j] for i, j in border_positions]
        return border_intensities
    
    def zero_out_matching_intensities(self,image, border_intensities):# Fonction pour mettre à zéro les pixels ayant la même intensité que les bordures
        image_cleaned = np.copy(image)
        for intensity in border_intensities:
            image_cleaned[image == intensity] = 0
        return image_cleaned

    def get_zero_positions(self,image):
        zero_positions = np.argwhere(image == 0)# Fonction pour obtenir les positions des pixels mis à zéro
        return zero_positions

    def zero_out_positions(self,image, positions):# Fonction pour mettre à zéro les pixels aux positions spécifiées
        image2 = np.copy(image)
        for i, j in positions:
            image2[i, j] = 0
            
        return image2

    def Compute_clustered_profiles(self):
        # Try to open the current_stack and check if the dim are the same than the labeled img
        # Else, ask to open the image series
        try : 
            serie = self.parent.Current_stack
        except:
            StackLoc, StackDir = gf.getFilePathDialog("Stack of images (*.tiff)")  # Ask to open the stack of images
            serie = tf.TiffFile(StackLoc[0]).asarray() # Import the array
            serie = np.flip(serie, 1)
            serie = np.rot90(serie, k=1, axes=(2, 1))
            
        # Convert labeled image as integer
        Labels_int = np.zeros((len(self.Corrected_label_img),len(self.Corrected_label_img[0])), dtype = int)

        for i in range(0,len(self.Corrected_label_img)):
            for j in range(0,len(self.Corrected_label_img[0])):
                Labels_int[i,j] = int(self.Corrected_label_img[i,j])

        # Expansion of labels ==> Get rid of grains boundaries !
        Labels_int = expand_labels(Labels_int, distance=10)

        # Definition of the mean profiles for each label
        moyen_profil=np.zeros((len(regionprops(Labels_int)),len(serie[:,0,0])))
        
        try :
            for i in range (len(serie[:,0,0])) :
                regions = regionprops(Labels_int, intensity_image=serie[i,:,:])
                for j in range (len(regions)) :
                    moyen_profil[j][i]=regions[j].mean_intensity 
        except:
            self.popup_message("Grain boundaries determination","Computation of clustered profiles failed. Check for data.",'icons/Grain_Icons.png')
            return
        
        # Creation of the clustered profiles list
        liste_clusters = np.copy(moyen_profil)

        self.liste = []

        for i in range (0,np.max(Labels_int)):
            var = liste_clusters[i,:]
            self.liste.append(var)

        self.liste = np.dstack(self.liste)
        self.liste = np.swapaxes(self.liste, 0, 1)
        
        self.Labels_int = Labels_int

    def Save_results(self):
        ti = time.strftime("%Y-%m-%d__%Hh-%Mm-%Ss") # Absolute time 
        
        directory = "Grain_segmentation_" + ti # Name of the main folder
        PathDir = os.path.join(self.StackDir, directory)  # where to create the main folder
        os.mkdir(PathDir)  # Create main folder
        
        # Information (.TXT) step
        with open(PathDir + '\Grain boundaries determination.txt', 'w') as file:
            file.write("KAD data: " + str(self.flag_KAD))
            file.write("\nFiltering parameter: " + str(self.Filter_choice) + " - " + (str(self.Filter_value)))   
            file.write("\nOtsu class: "+ str(self.Otsu1_Value) + "\nThresholded classes (keep values equal or higher than): " + str(self.Binary1_Value))   
            file.write("\nLabel denoising step: "+ str(self.flag_DenLabels))
            file.write("\nLabel denoising value: "+ str(self.Den_spinbox.value()))
            file.write("\nGrain diameter excluded (below): "+ str(self.Filter_labelValue))
            
            if self.flag_PixelSize == True:
                file.write("\nPixel size (µm): " + str(self.pixelSize))
        
        # Creation of the processing folder + save
        processing_folder = "Processing_step" # Name of the sub-folder
        SubPathDir = os.path.join(PathDir, processing_folder) # Sub-folder for processing step
        os.mkdir(SubPathDir)  # Create sub-folder
        
        tf.imwrite(SubPathDir + '/Map_CLAHE.tiff', np.rot90(np.flip(self.InitKAD_map, 0), k=1, axes=(1, 0)))
        tf.imwrite(SubPathDir + '/Filtered_map.tiff', np.rot90(np.flip(self.FilteredKAD_map, 0), k=1, axes=(1, 0)))  
        tf.imwrite(SubPathDir + '/Otsu.tiff', np.rot90(np.flip(self.regions, 0), k=1, axes=(1, 0)).astype('float32')) 
        tf.imwrite(SubPathDir + '/Binary_Otsu.tiff', np.rot90(np.flip(self.binary_regions, 0), k=1, axes=(1, 0)))

        if self.flag_Border == True:
            self.Corrected_label_img = self.Var_Labels_cleaned
            self.img_area = self.img_area_cleaned
            self.Corrected_img_diameter = self.Corrected_img_diameter_cleaned
            self.img_formfactor = self.img_formfactor_cleaned
            
            self.filter_area_pxls = self.filter_area_pxls_cleaned
            self.filter_labeldiameter = self.filter_labeldiameter_cleaned
            self.filter_form_factor = self.filter_form_factor_cleaned
            
            if self.flag_PixelSize == True:
                self.img_area_metric = self.img_area_metric_cleaned
                self.Corrected_img_diameter_metric = self.Corrected_img_diameter_metric_cleaned
                self.filter_area_metric = self.filter_area_metric_cleaned
                self.filter_labeldiameter_metric = self.filter_labeldiameter_metric_cleaned

        # Creation of the pixel data folder  + save
        Pxls_folder = "Pixel data" # Name of the sub-folder for filtered data 
        Pxls_SubPathDir = os.path.join(PathDir, Pxls_folder) # Sub-folder for filtered data
        os.mkdir(Pxls_SubPathDir)  # Create sub-folder
        
        tf.imwrite(Pxls_SubPathDir + '/Labeled_img.tiff', np.rot90(np.flip(self.Corrected_label_img, 0), k=1, axes=(1, 0)).astype('float32')) 
        tf.imwrite(Pxls_SubPathDir + '/Area_pxls.tiff', np.rot90(np.flip(self.img_area, 0), k=1, axes=(1, 0))) # Area map in pxls
        tf.imwrite(Pxls_SubPathDir + '/Equivalent_diameter_pxls.tiff', np.rot90(np.flip(self.Corrected_img_diameter, 0), k=1, axes=(1, 0)))
        tf.imwrite(Pxls_SubPathDir + '/form_factor_map.tiff', np.rot90(np.flip(self.img_formfactor, 0), k=1, axes=(1, 0))) # Area map in pxls

        tf.imwrite(Pxls_SubPathDir + '/Filtered_area_pxls.tiff', np.rot90(np.flip(self.filter_area_pxls, 0), k=1, axes=(1, 0))) # Area map in pxls
        tf.imwrite(Pxls_SubPathDir + '/Filtered_equivalent_diameter_pxls.tiff', np.rot90(np.flip(self.filter_labeldiameter, 0), k=1, axes=(1, 0)))
        tf.imwrite(Pxls_SubPathDir + '/Filtered_form_factor_map.tiff', np.rot90(np.flip(self.filter_form_factor, 0), k=1, axes=(1, 0))) # Area map in pxls

        # Creation of the metric data folder 
        if self.flag_PixelSize == True:
            Metric_folder = "Metric data" # Name of the sub-folder for metric data 
            Metric_SubPathDir = os.path.join(PathDir, Metric_folder) # Sub-folder for metric data
            os.mkdir(Metric_SubPathDir)  # Create sub-folder
        
            tf.imwrite(Metric_SubPathDir + '/Area_µm.tiff', np.rot90(np.flip(self.img_area_metric, 0), k=1, axes=(1, 0))) # Area map in µm
            tf.imwrite(Metric_SubPathDir + '/Equivalent_diameter_µm.tiff', np.rot90(np.flip(self.Corrected_img_diameter_metric, 0), k=1, axes=(1, 0)))
            
            tf.imwrite(Metric_SubPathDir + '/Filtered_area_µm.tiff', np.rot90(np.flip(self.filter_area_metric, 0), k=1, axes=(1, 0))) # Area map in µm
            tf.imwrite(Metric_SubPathDir + '/Filtered_equivalent_diameter_µm.tiff', np.rot90(np.flip(self.filter_labeldiameter_metric, 0), k=1, axes=(1, 0)))

        # Saving cluster if asked
        if self.Save_cluster.isChecked(): # If QCheckBox 'Save clustered profiles' is True, then the function is run
            self.Compute_clustered_profiles() 
            tf.imwrite(PathDir + '/Clustered_profiles.tiff', self.liste)
            tf.imwrite(PathDir + '/Labeled_img_NoGB.tiff', np.rot90(np.flip(self.Labels_int, 0), k=1, axes=(1, 0)).astype('float32')) 

        # CSV save step
        self.extract_value_list()
        
        np.savetxt(Pxls_SubPathDir + "/Grain_area_list_pxls.csv", self.area_list_pxls, delimiter = ",")
        np.savetxt(Pxls_SubPathDir + "/Grain_size_list_pxls.csv", self.extract_diameter, delimiter = ",")
        
        np.savetxt(Pxls_SubPathDir + "/Filtered_grain_area_list_pxls.csv", self.Filtered_area_list_pxls, delimiter = ",")
        np.savetxt(Pxls_SubPathDir + "/Filtered_grain_size_list_pxls.csv", self.filtered_diameter, delimiter = ",")                           

        np.savetxt(Pxls_SubPathDir + "/Form_factor_list.csv", self.form_factor_list, delimiter = ",")
        np.savetxt(Pxls_SubPathDir + "/Filtered_Form_factor_list.csv", self.Filtered_form_factor_list, delimiter = ",")
        
        if self.flag_PixelSize == True:
            np.savetxt(Metric_SubPathDir + "/Grain_area_list_µm.csv", self.area_list_metric, delimiter = ",")
            np.savetxt(Metric_SubPathDir + "/Grain_size_list_µm.csv", self.extract_diameter_metric, delimiter = ",")
            
            np.savetxt(Metric_SubPathDir + "/Filtered_grain_area_list_µm.csv", self.Filtered_area_list_metric, delimiter = ",")
            np.savetxt(Metric_SubPathDir + "/Filtered_grain_size_list_µm.csv", self.filtered_diameter_metric, delimiter = ",")

        # Finished message
        self.popup_message("Grain boundaries determination","Saving process is over.",'icons/Grain_Icons.png')

    def validate_data(self): # Push labeled image in the main GUI
        self.parent.Label_image = np.copy(self.Corrected_label_img) # Copy in the main GUI
        self.parent.StackList.append(self.Corrected_label_img) # Add the data in the stack list
        
        Combo_text = '\u2022 Grain labeling'
        Combo_data = self.Corrected_label_img
        self.parent.choiceBox.addItem(Combo_text, Combo_data) # Add the data in the QComboBox

        self.parent.displayDataview(self.parent.Label_image) # Display the labeled grain
        self.parent.choiceBox.setCurrentIndex(self.parent.choiceBox.count() - 1) # Show the last data in the choiceBox QComboBox

        self.parent.Info_box.ensureCursorVisible()
        self.parent.Info_box.insertPlainText("\n \u2022 Grain labeled.") 
        
        self.parent.flag_labeling = True
        
        self.parent.Save_button.setEnabled(True)
        self.parent.Reload_button.setEnabled(True)
        self.parent.choiceBox.setEnabled(True)
        
        # Finished message
        self.popup_message("Grain boundaries determination","Labeled image has been exported to the main GUI.",'icons/Grain_Icons.png')

    def displayExpKAD(self, series): # Display of initial map
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
        
    def displayFilteredKAD(self, series): # Display of filtered map
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
        
    def displayOtsu1(self, series): # Display of Otsu1 map
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
                
    def displayBinary1(self, series): # Display of binary1 map
        self.Binary1Series.addItem(self.crosshair_v4, ignoreBounds=True)
        self.Binary1Series.addItem(self.crosshair_h4, ignoreBounds=True) 
        
        self.Binary1Series.ui.histogram.hide()
        self.Binary1Series.ui.roiBtn.hide()
        self.Binary1Series.ui.menuBtn.hide()
        
        view = self.Binary1Series.getView()
        state = view.getState()        
        self.Binary1Series.setImage(series) 
        view.setState(state)
        view.setBackgroundColor(self.parent.color1)
        
    def displaylabels(self, series): # Display of label map
        self.LabelsSeries.addItem(self.crosshair_v5, ignoreBounds=True)
        self.LabelsSeries.addItem(self.crosshair_h5, ignoreBounds=True) 
        
        self.LabelsSeries.ui.histogram.show()
        self.LabelsSeries.ui.roiBtn.hide()
        self.LabelsSeries.ui.menuBtn.hide()
        
        view = self.LabelsSeries.getView()
        state = view.getState()        
        self.LabelsSeries.setImage(series) 
        view.setState(state)
        view.setBackgroundColor(self.parent.color1)
        
        histplot = self.LabelsSeries.getHistogramWidget()
        histplot.setBackground(self.parent.color1)
        
        histplot.region.setBrush(pg.mkBrush(self.parent.color5 + (120,)))
        histplot.region.setHoverBrush(pg.mkBrush(self.parent.color5 + (60,)))
        histplot.region.pen = pg.mkPen(self.parent.color5)
        histplot.region.lines[0].setPen(pg.mkPen(self.parent.color5, width=2))
        histplot.region.lines[1].setPen(pg.mkPen(self.parent.color5, width=2))
        histplot.fillHistogram(color = self.parent.color5)        
        histplot.autoHistogramRange()   
        
        self.LabelsSeries.setColorMap(pg.colormap.get('viridis'))

    def defaultIV(self):
        # KADSeries: Initial map
        self.KADSeries.ui.histogram.hide()
        self.KADSeries.ui.roiBtn.hide()
        self.KADSeries.ui.menuBtn.hide()
        
        view = self.KADSeries.getView()
        view.setBackgroundColor(self.parent.color1)
        
        # FiltKADSeries: map after filtering
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
        
        # LabelsSeries: grain labeling
        self.LabelsSeries.ui.histogram.hide()
        self.LabelsSeries.ui.roiBtn.hide()
        self.LabelsSeries.ui.menuBtn.hide()
        
        view = self.LabelsSeries.getView()
        view.setBackgroundColor(self.parent.color1)
        
    def mouseMoved(self, e):
        pos = e[0]
        sender = self.sender()
  
        if not self.mouseLock.isChecked():
            if self.KADSeries.view.sceneBoundingRect().contains(pos)\
                or self.FiltKADSeries.view.sceneBoundingRect().contains(pos)\
                or self.Otsu1Series.view.sceneBoundingRect().contains(pos)\
                or self.Binary1Series.view.sceneBoundingRect().contains(pos)\
                or self.LabelsSeries.view.sceneBoundingRect().contains(pos):    
                
                if sender == self.proxy1:
                    item = self.KADSeries.view
                elif sender == self.proxy3:
                    item = self.FiltKADSeries.view
                elif sender == self.proxy5:
                    item = self.Otsu1Series.view
                elif sender == self.proxy7:
                    item = self.Binary1Series.view
                else:
                    item = self.LabelsSeries.view
                
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

            try:
                self.x = int(mousePoint.x())
                self.y = int(mousePoint.y())
                
                self.printClick(self.x, self.y, sender)
            except:
                pass
    
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
        if self.flag_info == True:
            try:
                self.Otsu1_label.setText("Otsu classes: " + str(self.regions[x, y]))
            except:
                pass
            
        if self.flag_info_labels == False:
            try:
                self.GrainQLabel.setText("Label n°: " + str(self.Corrected_label_img[x, y]))        
            except:
                pass
        
        if self.flag_info_labels == True:
            try:
                self.GrainQLabel.setText("Equivalent \u2300 (pxls): " + str(np.round(self.Corrected_img_diameter[x, y],2)))        
            except:
                pass
            
        if self.flag_info_labels_metric == True:
            try:
                self.GrainQLabel.setText("Equivalent \u2300 (µm): " + str(np.round(self.Corrected_img_diameter_metric[x, y],2)))        
            except:
                pass
            
        if self.flag_info_overlay == True:
            try:
                self.GrainQLabel.setText("Map: " + str(np.round(self.overlay_KAD_GB[x, y],2)))        
            except:
                pass
            
        if self.flag_info_filtered == True:
            try:
                self.GrainQLabel.setText("Equivalent \u2300 (pxls): " + str(np.round(self.filter_labeldiameter[x, y],2)))        
            except:
                pass
            
        if self.flag_info_filtered2 == True:
            try:
                self.GrainQLabel.setText("Equivalent \u2300 (µm): " + str(np.round(self.filter_labeldiameter_metric[x, y],2)))        
            except:
                pass