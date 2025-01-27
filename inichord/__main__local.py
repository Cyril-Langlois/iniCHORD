# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 23:26:20 2023

@author: clanglois1
"""
import os
from os.path import abspath

import sys

from inspect import getsourcefile
import numpy as np
from skimage.measure import regionprops
import tifffile as tf

import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMessageBox, QLabel, QDialog, QVBoxLayout, QPushButton
from PyQt5 import QtCore, QtGui

from inichord import General_Functions as gf
from inichord import Profile_Modification as fct
from inichord import Edit_Tools as sm
# from inichord import Registration as align
import Registration_local as align
from inichord import Remove_FFT as RemFFT
from inichord import Remove_Outliers as rO
# from inichord import NLMD as nl
import NLMD_local as nl
# from inichord import BM3D as bm
import BM3D_local as bm
from inichord import VSNR as vs
import TV_local as tv
# from inichord import Auto_Denoising as autoden
import Auto_Denoising_local as autoden
from inichord import KAD_Function as KADfunc
from inichord import Contour_Map as Contour
# from inichord import Denoise_2Dmap as Denmap
import Denoise_2Dmap_local as Denmap
# from inichord import Grain_Treatment as GB
import Grain_Treatment_local2 as GB
# from inichord import Restored_Grains as Restored
import Restored_Grains_local as Restored
# from inichord import Extract_Mosaic as 
import Extract_Mosaic_local as Extract_mosaic
# from inichord import Batch_Processing as Batch
import Batch_Processing_local as Batch
from inichord import TwoD_Stitching as Img_Stitch
from inichord import ThreeD_Stitching as Series_Stitch


path2thisFile = abspath(getsourcefile(lambda:0))
uiclass, baseclass = pg.Qt.loadUiType(os.path.dirname(path2thisFile) + "/__main__local.ui") 

if "cupy" in sys.modules:
    # from indexGPU import Indexation_GUI as Indexation_TSG
    sys.path.append("F:\eCHORD\Programmation\indexGPU\indexGPU")
    import Indexation_GUI as Indexation_TSG

class MainWindow(uiclass, baseclass):
    def __init__(self):
        super().__init__()
        
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon('icons/Main_icon.png')) # Application of the main icon
             
        # Colors will be applied to all the sub-gui
        self.color1 = (255, 255, 255) # Background color of imageView
        self.color2 = (255, 255, 255) # Background color of PlotWidget
        self.color3 = (243, 98, 64)   # PushButton color (for Qsplitter in ImageView)
        self.color4 = (243, 98, 64, 150) # Color of the line plot number 1
        self.color5 = (243, 98, 64)   # Color of the line plot number 2 
        self.color6 = (193, 167, 181,50) # Brush Color for legend in plot

        self.Open_data.clicked.connect(self.loaddata) # Load image series or 2D array (for KAD map)
        self.Eight_bits_button.clicked.connect(self.convert_to_8bits)
        self.Edit_tools_button.clicked.connect(self.stackmodifier) # Modification of image series (cropping, binning, slicing)
        self.Registration_button.clicked.connect(self.Stackregistration) # Stack registration
        self.Background_remover_button.clicked.connect(self.FFTFiltering) # FFT background substraction
        self.Remove_outliers_button.clicked.connect(self.RemoveOutlier) # Outlier filtering
        self.Manual_denoising_button.clicked.connect(self.ManualDenoising) # Denoising with manual parameter definition
        self.Auto_denoising_button.clicked.connect(self.AutoDenoisingStep) # Denoising with semi-automatic value determination
        self.Reload_button.clicked.connect(self.ReloadStack) # Reload previous data
        self.Tool_button.clicked.connect(self.toolbox) # Run the selected toolbox  
        self.Save_button.clicked.connect(self.ExtractionStack) # Extract data from the GUI
        
        self.crosshair_v1= pg.InfiniteLine(angle=90, movable=False, pen=self.color5)
        self.crosshair_h1 = pg.InfiniteLine(angle=0, movable=False, pen=self.color5)
        
        self.crosshair_v2= pg.InfiniteLine(angle=90, movable=False, pen=self.color5)
        self.crosshair_h2 = pg.InfiniteLine(angle=0, movable=False, pen=self.color5)
        
        self.proxy1 = pg.SignalProxy(self.expSeries.scene.sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.proxy4 = pg.SignalProxy(self.expSeries.ui.graphicsView.scene().sigMouseClicked, rateLimit=60, slot=self.mouseClick)
        
        self.proxy2 = pg.SignalProxy(self.dataview.scene.sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.proxy5 = pg.SignalProxy(self.dataview.ui.graphicsView.scene().sigMouseClicked, rateLimit=60, slot=self.mouseClick)

        self.plotIt = self.profiles.getPlotItem() # Get the PlotItem as a variable
        self.plotIt.addLine(x = self.expSeries.currentIndex) # Vertical line on the profile plot widget

        self.expSeries.timeLine.sigPositionChanged.connect(self.drawCHORDprofiles) # Update displayed profile
        self.defaultdrawCHORDprofiles() # Default plot display (when no drawing)
        
        self.Info_box.setFontPointSize(10) # Font size of the QTextEdit widget
        self.StackList = [] # Initialization of the data storage list
        
        self.defaultIV() # Default ImageView (when no image)

        # Buttons are not enables except [Open data ; Tool_choice ; Run button ; Indexation]
        self.Edit_tools_button.setEnabled(False)
        self.Registration_button.setEnabled(False)
        self.Eight_bits_button.setEnabled(False)
        self.Background_remover_button.setEnabled(False)
        self.Remove_outliers_button.setEnabled(False)
        self.Manual_denoising_button.setEnabled(False)
        self.Auto_denoising_button.setEnabled(False)
        self.Save_button.setEnabled(False)
        self.Reload_button.setEnabled(False)
        self.Choice_denoiser.setEnabled(False)
        self.choiceBox.setEnabled(False)
        self.progressBar.setVisible(False) # The progress bar is hidden for clarity (used only for GRDD-GDS)
        self.mouseLock.setVisible(False)
        
        for i in range(0,5): # Disable [Bleach correction ; STD map ; KAD map] at the beginning
            self.Tool_choice.model().item(i).setEnabled(False)
            
        self.Tool_choice.setCurrentIndex(5) # Default selection at the "Contour map" toolbox
        self.flag_image = False # Linked to the GRDD-GDS computation (image serie)
        self.flag_labeling = False # Linked to the GRDD-GDS computation (labeled image)
        self.flag_stitchKAD = False # In order to specify which data has been used for 2D stitching
                
        app = QApplication.instance()
        screen = app.screenAt(self.pos())
        geometry = screen.availableGeometry()
        
        # Position (self.move) and size (self.resize) of the main GUI on the screen
        self.move(int(geometry.width() * 0.05), int(geometry.height() * 0.05))
        self.resize(int(geometry.width() * 0.8), int(geometry.height() * 0.7))
        self.screen = screen
        
        self.Indexation_button.setVisible(False)
        
        if "cupy" in sys.modules:
            self.Indexation_button.setVisible(True)
            self.Indexation_button.clicked.connect(self.Indexation_orientation) # Orientation indexation
        
#%% Functions
    try:
        def Indexation_orientation(self): # Run the indexing sub-gui
            self.w = Indexation_TSG.MainWindow(self)
            self.w.show()
    except:
        pass
    
    def convert_to_8bits(self):
        if not (isinstance(self.Current_stack.flat[0], np.int8) or isinstance(self.Current_stack.flat[0], np.uint8)): #if not 8bits
            self.eight_bits_img = gf.convertToUint8(self.Current_stack)
            self.StackList.append(self.eight_bits_img)
            Combo_text = '\u2022 8 bits data'
            Combo_data = self.eight_bits_img
            self.choiceBox.addItem(Combo_text, Combo_data)
            self.Current_stack = self.eight_bits_img
            self.Info_box.insertPlainText("\n \u2022 Data has been converted to 8 bits.")
            self.Eight_bits_button.setEnabled(False)
        else:
            self.Info_box.insertPlainText("\n \u2022 Data was already 8 bits type.")
        
    def closeEvent(self, event):
        msgBox = QMessageBox(self)
        msgBox.setWindowTitle('Quit')
        msgBox.setText("Do you really want to quit ?")
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        
        # Changer la police
        font = msgBox.font()
        font.setPointSize(10)  # Ajuste la taille ici
        msgBox.setFont(font)
        
        reply = msgBox.exec()
    
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
            
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
        
    def loaddata(self): # Allow to load image serie (3D stack or image sequence) and 2D map (KAD data)
        self.StackLoc, self.StackDir = gf.getFilePathDialog("Image stack (*.tiff)")  # Ask to open stack of images
        
        if len(self.StackLoc) == 1 : # Check if several data has been imported (stack or image sequence)
            checkimage = tf.TiffFile(self.StackLoc[0]).asarray() # Check for dimension. If 2 dimensions : 2D array. If 3 dimensions : stack of images
            
            # 2D data importation management
            if checkimage.ndim != 3: # Check if the data is a 2D array               
                self.TwoDarray = tf.TiffFile(self.StackLoc[0]).asarray() # Import the array
                self.TwoDarray = np.flip(self.TwoDarray, 0) # Flip the array
                self.TwoDarray = np.rot90(self.TwoDarray, k=1, axes=(1, 0)) # Rotate the array
                
                if len(self.StackList) > 0: 
                    self.StackList = [] # Clean the StackList of arrays
                    self.choiceBox.clear() # Clean the choiceBox
                    
                try: # Delete of image series and current stack is any
                    del(self.image, self.Current_stack)
                except:
                    pass

                self.defaultIV() # Default ImageView 

                msg = QMessageBox.question(self, '2D image imported', 'Is it a KAD data ?')
                if msg == QMessageBox.Yes:
                    Combo_text = '\u2022 KAD map'
                    Combo_data = self.TwoDarray
                    self.choiceBox.addItem(Combo_text, Combo_data) # Add the KAD data to the choice_box
                    self.label_Treatment.setText("KAD map") # Change the title accordingly to the displayed data
                    
                    self.KAD = np.copy(self.TwoDarray)
                    self.displayDataview(self.KAD) # Display of the 2D map
                    
                    self.StackList.append(self.KAD) # Add the data to the StackList

                if msg == QMessageBox.No:
                    Combo_text = '\u2022 Contour map'
                    Combo_data = self.TwoDarray
                    self.choiceBox.addItem(Combo_text, Combo_data) # Add the contour map data to the choice_box
                    self.label_Treatment.setText("Contour map") # Change the title accordingly to the displayed data
                    
                    self.contour_map = np.copy(self.TwoDarray)
                    self.displayDataview(self.contour_map) # Display of the 2D map
                    
                    self.StackList.append(self.contour_map) # Add the data to the StackList

                self.Info_box.clear() # Clean the information box
                self.Info_box.insertPlainText("\n \u2022 Data has been loaded.")

                for i in range(0,3): # Disable [Bleach correction ; STD map ; KAD map]
                    self.Tool_choice.model().item(i).setEnabled(False)
                
                self.Tool_choice.setCurrentIndex(3) # Default selection at the "Grain boundaries" toolbox

                # Enable [Save ; Reload ; ChoiceBox] Widgets
                self.Save_button.setEnabled(True)
                self.Reload_button.setEnabled(True)
                self.choiceBox.setEnabled(True)

            # Stack of image importation management
            else: # If the imported data has 3 dimensions, then it is a stack of images
                self.image = tf.TiffFile(self.StackLoc[0]).asarray() # Import the array
                self.image = np.flip(self.image, 1) # Flip the array
                self.image = np.rot90(self.image, k=1, axes=(2, 1)) # Rotate the array
                
                for i in range(0,5): # Enable [Bleach correction ; STD map ; KAD map]
                    self.Tool_choice.model().item(i).setEnabled(True)
                        
                self.flag_image = True # Certifies that the stack of image has been imported (for GRDD-GDS computation)
        
        # Image sequence importation management
        else : # If the data is a sequence of images
            self.image = [] # Initialization of the variable self.image (which will be the stack)
    
            for i in range(0,len(self.StackLoc)):
                Var = tf.TiffFile(self.StackLoc[i]).asarray() # Import the unit 2D array
                self.image.append(Var) # Append every 2D array in a list named self.image
                
            self.image = np.dstack(self.image) # Convert list of 3D array
            self.image = np.swapaxes(self.image, 0, 2) # Rearrangement of the axis
            self.image = np.swapaxes(self.image, 1, 2) # Rearrangement of the axis
            self.image = np.flip(self.image, 1) # Flip the array
            self.image = np.rot90(self.image, k=1, axes=(2, 1)) # Rotate the array
            
            self.flag_image = True # Certifies that the stack of image has been imported (for GRDD-GDS computation)
            
            for i in range(0,5): # Enable [Bleach correction ; STD map ; KAD map]
                self.Tool_choice.model().item(i).setEnabled(True)
            
            del (Var) # Delete Var which become useless
            
        try: # After importation, try to display data, clean information box / StackList and enables buttons activation   
            self.displayExpStack(self.image) # Display the 3D array
                          
            self.Current_stack = np.copy(self.image) # Create the self.Current_stack which will be used for computation
            
            if len(self.StackList) > 0:
                self.StackList = [] # Clean the StackList of arrays
                self.choiceBox.clear() # Clean the choiceBox 
    
            self.StackList.append(self.image) # Add the 3D stack in the StackList
            
            Combo_text = '\u2022 Initial stack'
            Combo_data = self.image
            self.choiceBox.addItem(Combo_text, Combo_data)  # Add the 3D stack to choice_box
            
            self.flag = False # Certifies that no image reference has been imported
            
            if self.Import_reference.isChecked(): # If QCheckBox 'Import image reference' is True, then the function is run
                self.ImportReference() 
            
            self.Info_box.clear() # Clean the information box
            self.Info_box.insertPlainText("\n \u2022 Data has been loaded.")
            
            self.dataview.clear() # Clean the data view (no more data)
            self.dataview.ui.histogram.hide()
            self.dataview.ui.roiBtn.hide()
            self.dataview.ui.menuBtn.hide()
            
            view = self.dataview.getView()
            view.setBackgroundColor(self.color1)
            self.label_Treatment.setText("Treatment")
            
            # Activation of the widgets that were disable
            self.Edit_tools_button.setEnabled(True)
            
            self.Registration_button.setEnabled(True)
            self.Background_remover_button.setEnabled(True)
            self.Remove_outliers_button.setEnabled(True)
            self.Manual_denoising_button.setEnabled(True)
            self.Auto_denoising_button.setEnabled(True)
            self.Save_button.setEnabled(True)
            self.Reload_button.setEnabled(True)
            self.Choice_denoiser.setEnabled(True)
            self.choiceBox.setEnabled(True)
            
            if not (isinstance(self.Current_stack.flat[0], np.int8) or isinstance(self.Current_stack.flat[0], np.uint8)) :
                self.Eight_bits_button.setEnabled(True)
            
            self.Tool_choice.setCurrentIndex(0) 
        except: # If the try is not possible, then nothing happens
            pass

    def ImportReference(self): # Allow the importation of an image of reference
        # The reference (2D image) will be reshaped in 3D (with 0 dim as the first dim) to be included in the stack
        self.StackLoc, self.StackDir = gf.getFilePathDialog("Image reference (*.tiff)") 
        self.Reference = tf.TiffFile(self.StackLoc[0]).asarray()
        self.Reference = np.reshape(self.Reference,(-1,len(self.Reference),len(self.Reference[0])))
        self.Reference = np.flip(self.Reference, 1)
        self.Reference = np.rot90(self.Reference, k=1, axes=(2, 1))
        
        # Adding the reference image to the unaligned stack
        self.Current_stack = np.concatenate((self.Reference, self.Current_stack))
        self.displayExpStack(self.Current_stack)
        self.displayDataview(self.Reference)
        
        self.flag = True # Certifies that image reference has been imported
        
        self.Info_box.ensureCursorVisible()
        self.Info_box.insertPlainText("\n \u2022 Image reference has been loaded.")
        
    def ExtractReference(self): # Extract the image reference from the stack of images
        self.Reference = np.copy(self.Current_stack[0,:,:]) # Extraction of the image of reference
        self.Slice1 = np.copy(self.Current_stack[1,:,:]) # Copy of the first slice of the stack
        self.Current_stack = np.copy(self.Current_stack[1:,:,:]) # Create of the Current stack without reference
        self.displayExpStack(self.Current_stack) 

    def ReloadStack(self): # Allow to reload a data from the StackList QComboBox
        self.choice = self.choiceBox.currentText() # Define the data that must be considered
        self.index = self.choiceBox.currentIndex() # Define the index of the considered data
        self.max_index = self.choiceBox.count() # Count the number of data in the StackList
        
        if self.choice == "\u2022 STD map": # If the data is the STD map
            self.label_Treatment.setText("STD map")
            self.displayDataview(self.std_image) # Display STD map on the Treatment ImageView (dataview)
        elif self.choice == "\u2022 KAD map": # If the data is the KAD map
            self.label_Treatment.setText("KAD map")
            self.displayDataview(self.KAD) # Display KAD map on the Treatment ImageView (dataview)
        elif self.choice == "\u2022 AVG map": # If the data is the AVG map
            self.label_Treatment.setText("AVG map")
            self.displayDataview(self.avg_image) # Display AVG map on the Treatment ImageView (dataview)
        elif self.choice == "\u2022 MED map": # If the data is the MED map
            self.label_Treatment.setText("MED map")
            self.displayDataview(self.med_image) # Display MED map on the Treatment ImageView (dataview)
        elif self.choice == "\u2022 Contour map": # If the data is the KAD map
            self.label_Treatment.setText("Contour map")
            self.displayDataview(self.contour_map) # Display KAD map on the Treatment ImageView (dataview)
        elif self.choice == "\u2022 Grain labeling": # If the data is the grains labeling map
            self.label_Treatment.setText("Grain labels")
            self.displayDataview(self.Label_image) # Display the labeled image map on the Treatment ImageView (dataview)
        elif self.choice == "\u2022 GRDD map": # If the data is the GRDD map
            self.label_Treatment.setText("GRDD map")
            self.displayDataview(self.dot) # Display the GRDD map on the Treatment ImageView (dataview)
        elif self.choice == "\u2022 GDS map": # If the data is the GDS map
            self.label_Treatment.setText("GDS map")
            self.displayDataview(self.dot_mean) # Display the GDS map on the Treatment ImageView (dataview)
        elif self.choice == "\u2022 Stitched map": # If the data is the stitched map
            if self.flag_stitchKAD == True:
                self.label_Treatment.setText("Stitched map")
                self.displayDataview(self.KAD) # Display the stitched map on the Treatment ImageView (dataview)
            else:
                self.label_Treatment.setText("Stitched map")
                self.displayDataview(self.contour_map) # Display the stitched map on the Treatment ImageView (dataview)
                    
        else: # We assume the other data are 3D stacks (from stitching for example, or previous treatment)
        # In this case, all data after the selected one will be removed to avoid overload in the StackList
            self.Current_stack = np.copy(self.StackList[int(self.index)])
            self.displayExpStack(self.Current_stack)
            self.StackList = self.StackList[0:int(self.index)+1] # Delete item from list
            
            for i in range(self.max_index,(self.index),-1):  # Delete item for the choiceBox accordingly to StackList
                self.choiceBox.removeItem(i)
                
            self.dataview.clear()
            self.dataview.ui.histogram.hide()
            self.dataview.ui.roiBtn.hide()
            self.dataview.ui.menuBtn.hide()
            
            view = self.dataview.getView()
            
            view.setBackgroundColor(self.color1)
            self.label_Treatment.setText("Treatment")
        
    def ExtractionStack(self): # Function to save the selected data in the wanted folder
        self.choice = self.choiceBox.currentText() # Define the data that must be considered
        self.index = self.choiceBox.currentIndex() # Define the index of the considered data
        
        self.SavedStack = np.copy(self.StackList[int(self.index)]) # Save stack is the copy of the wanted data
    
        if self.SavedStack.ndim != 3: # If it is a 2D array
            self.SavedStack = np.flip(self.SavedStack, 0)
            self.SavedStack = np.rot90(self.SavedStack, k=1, axes=(1, 0))
        else:
            self.SavedStack = np.flip(self.SavedStack, 1)
            self.SavedStack = np.rot90(self.SavedStack, k=1, axes=(2, 1))
            
        gf.Saving_img_or_stack(self.SavedStack)
        self.Info_box.insertPlainText("\n \u2022 Data saved : " + str(self.choice))
                        
    def stackmodifier(self): # Run the stack modification sub-gui
        self.w = sm.MainWindow(self)
        self.w.show()

    def Stackregistration(self): # Run the stack registration sub-gui
        self.w = align.MainWindow(self)
        self.w.show()

    def FFTFiltering(self): # Run the FFT filtering sub-GUI
        self.w = RemFFT.MainWindow(self)
        self.w.show()

    def RemoveOutlier(self): # Run the remove outlier filtering sub-gui
        self.w = rO.MainWindow(self)
        self.w.show()
       
    def ManualDenoising(self): # Run the manual denoising sub-gui
        self.Manual_denoiser_choice = self.Choice_denoiser.currentText()
        
        if self.Manual_denoiser_choice == 'Manual NLMD':
            self.w = nl.MainWindow(self)
            self.w.show()
        elif self.Manual_denoiser_choice == 'Manual BM3D':
            self.w = bm.MainWindow(self)
            self.w.show()
        elif self.Manual_denoiser_choice == 'Manual VSNR':
            self.w = vs.MainWindow(self)
            self.w.show()     
        elif self.Manual_denoiser_choice == 'Manual TV':
            self.w = tv.MainWindow(self)
            self.w.show()      

    def AutoDenoisingStep(self): # Run the auto-denoising sub-gui
        if self.flag == True: # If an image reference was imported
            self.ExtractReference() # Extraction of the reference from the stack of images
            self.w = autoden.MainWindow(self)
            self.w.show()   
            
            self.flag = False
        
        elif self.flag == False: # If no image reference was imported
            self.w = autoden.MainWindow(self)
            self.w.show()   
                            
    def toolbox(self): # Allow to use different programs or sub-gui
        self.Toolchoice = self.Tool_choice.currentText() # Extract the current name of the toolbox selection
        
        if self.Toolchoice == 'STD map':
            self.std_image = np.nanstd(self.Current_stack,0) # Creation of the STD map
            self.displayDataview(self.std_image)
            
            self.StackList.append(self.std_image)
            
            Combo_text = '\u2022 STD map'
            Combo_data = self.std_image
            self.choiceBox.addItem(Combo_text, Combo_data)
            
            self.Info_box.ensureCursorVisible()
            self.Info_box.insertPlainText("\n \u2022 STD map has been computed.")
            self.choiceBox.setCurrentIndex(self.choiceBox.count() - 1) # Show the last data in the choiceBox QComboBox
            
        elif self.Toolchoice == 'AVG map':
            self.avg_image = np.nanmean(self.Current_stack,0) # Creation of the AVG map
            self.displayDataview(self.avg_image)
            
            self.StackList.append(self.avg_image)
            
            Combo_text = '\u2022 AVG map'
            Combo_data = self.avg_image
            self.choiceBox.addItem(Combo_text, Combo_data)
            
            self.Info_box.ensureCursorVisible()
            self.Info_box.insertPlainText("\n \u2022 AVG map has been computed.")
            self.choiceBox.setCurrentIndex(self.choiceBox.count() - 1) # Show the last data in the choiceBox QComboBox
            
        elif self.Toolchoice == 'MED map':
            self.med_image = np.median(self.Current_stack,0) # Creation of the MED map
            self.displayDataview(self.med_image)
            
            self.StackList.append(self.med_image)
            
            Combo_text = '\u2022 MED map'
            Combo_data = self.med_image
            self.choiceBox.addItem(Combo_text, Combo_data)
            
            self.Info_box.ensureCursorVisible()
            self.Info_box.insertPlainText("\n \u2022 MED map has been computed.")
            self.choiceBox.setCurrentIndex(self.choiceBox.count() - 1) # Show the last data in the choiceBox QComboBox
          
        elif self.Toolchoice == 'Bleach correction': # Application of bleach correction if surface contamination
            self.Current_stack = gf.bleach_ratio(self.Current_stack)
            self.Bleach_stack = gf.bleach_ratio(self.Current_stack) # An np.copy(self.current_stack) should also works
            self.displayExpStack(self.Current_stack)
            
            self.StackList.append(self.Bleach_stack)
            
            Combo_text = '\u2022 Stack after bleaching'
            Combo_data = self.Bleach_stack
            self.choiceBox.addItem(Combo_text, Combo_data)
            
            self.Info_box.ensureCursorVisible()
            self.Info_box.insertPlainText("\n \u2022 Bleach correction has been computed.")
            
        elif self.Toolchoice == "KAD map":
            self.Info_box.ensureCursorVisible()
            self.Info_box.insertPlainText("\n \u2022 KAD map construction in progress....")
            QApplication.processEvents()
            
            self.stack_norm = fct.centeredEuclidianNorm(self.Current_stack, 0) # Normalization of the image series
            self.KAD = KADfunc.Divided_KAD(self.stack_norm) # Compute the KAD map
            self.displayDataview(self.KAD)
            
            self.StackList.append(self.KAD)
            
            Combo_text = '\u2022 KAD map'
            Combo_data = self.KAD
            self.choiceBox.addItem(Combo_text, Combo_data)
            
            self.Info_box.ensureCursorVisible()
            self.Info_box.insertPlainText("\n \u2022 KAD map has been computed.")
            self.choiceBox.setCurrentIndex(self.choiceBox.count() - 1) # Show the last data in the choiceBox QComboBox
            
        elif self.Toolchoice == 'Contour map': # Run the contour computation (using sobel filter sub-gui)
            self.w = Contour.MainWindow(self)
            self.w.show()  
            
        elif self.Toolchoice == "Denoise 2D map": # Run the grain boundaries determination sub-gui
            self.w = Denmap.MainWindow(self)
            self.w.show()   
            
        elif self.Toolchoice == "Grain boundaries": # Run the grain boundaries determination sub-gui
            self.w = GB.MainWindow(self)
            self.w.show()   
            
        elif self.Toolchoice == "Restored grains": # Run the restored grains sub-gui
            self.w = Restored.MainWindow(self)
            self.w.show()   
            
        elif self.Toolchoice == "GRDD-GDS": # Run the GRDD-GDS computation step
            self.GRDD_GDS_computation() 
            
        elif self.Toolchoice == "Extract mosaic": # Run the procedure to extract series from folder sub-gui
            self.w = Extract_mosaic.MainWindow(self)
            self.w.show()
            
        elif self.Toolchoice == "Batch processing": # Run the procedure to extract series from folder sub-gui
            self.w = Batch.MainWindow(self)
            self.w.show()
            
        elif self.Toolchoice == "2D stitching": # Run the 2D stitching sub-gui
            self.w = Img_Stitch.MainWindow(self)
            self.w.show()   
            
        elif self.Toolchoice == "3D stitching": # Run the 3D stitching (image serie stitching) sub-gui
            self.w = Series_Stitch.MainWindow(self)
            self.w.show()
        
    def GRDD_GDS_computation(self): # Define which data are missing before GRDD-GDS computation
        # Case where no serie and no labeled image are available
        if self.flag_image == False and self.flag_labeling == False : 
            self.popup_message("GRDD-GDS","Import stack of images and the labeled image",'icons/Main_icon.png')

            try: # Import the image serie
                self.StackLoc, self.StackDir = gf.getFilePathDialog("Image series (*.tiff)") 
                self.image = tf.TiffFile(self.StackLoc[0]).asarray()
                self.image = np.flip(self.image, 1)
                self.image = np.rot90(self.image, k=1, axes=(2, 1))
            except:
                self.popup_message("GRDD-GDS","Error: please import a stack (3D)",'icons/Main_icon.png')
                return
            
            # Import the labeled image
            self.StackLoc, self.StackDir = gf.getFilePathDialog("labeled image (*.tiff)") 
                
            checkimage = tf.TiffFile(self.StackLoc[0]).asarray() # Check for dimensions
            if checkimage.ndim != 2: # Check if the data is a 2D array or not
                self.popup_message("GRDD-GDS","Error: please import a labeled image (2D)",'icons/Main_icon.png')
                return
            else:
                self.Label_image = tf.TiffFile(self.StackLoc[0]).asarray()
                self.Label_image = np.flip(self.Label_image, 0)
                self.Label_image = np.rot90(self.Label_image, k=1, axes=(1, 0))
                
            try:
                self.GRDD_GDS() # Run the GRDD-GDS computation
            except:
                self.popup_message("GRDD-GDS","GRDD-GDS fail. Check the exactness of the imports",'icons/Main_icon.png')
                return
            
        # Case where no serie is imported but labeled image has been computed
        elif self.flag_image == False and self.flag_labeling == True : 
            self.popup_message("GRDD-GDS","Please import the image series",'icons/Main_icon.png')
            
            self.StackLoc, self.StackDir = gf.getFilePathDialog("Image series (*.tiff)") # Import the image serie
            self.image = tf.TiffFile(self.StackLoc[0]).asarray()
            self.image = np.flip(self.image, 1)
            
            try:
                self.image = np.rot90(self.image, k=1, axes=(2, 1)) # If a 2D array is imported, an error occurs
            except:
                self.popup_message("GRDD-GDS","Please import 3D stack (not a 2D image)",'icons/Main_icon.png')
                return

            try:
                self.GRDD_GDS() # Run the GRDD-GDS computation
            except:
                self.popup_message("GRDD-GDS","GRDD-GDS fail. Check the exactness of the imports",'icons/Main_icon.png')
                return
            
        # Case where image serie has been imported but labeled image is missing
        elif self.flag_image == True and self.flag_labeling == False: 
            self.popup_message("GRDD-GDS","Please import the labeled image",'icons/Main_icon.png')

            self.StackLoc, self.StackDir = gf.getFilePathDialog("Labeled image (*.tiff)") # Import the labeled image
            
            checkimage = tf.TiffFile(self.StackLoc[0]).asarray() # Check for dimensions
            if checkimage.ndim != 2: # Check if the data is a 2D array or not
                self.popup_message("GRDD-GDS","Error: please import a labeled image (2D)",'icons/Main_icon.png')
                return
            else:
                self.Label_image = tf.TiffFile(self.StackLoc[0]).asarray()
                self.Label_image = np.flip(self.Label_image, 0)
                self.Label_image = np.rot90(self.Label_image, k=1, axes=(1, 0))

            try:
                self.GRDD_GDS() # Run the GRDD-GDS computation
            except:
                self.popup_message("GRDD-GDS","GRDD-GDS fail. Check the exactness of the imports",'icons/Main_icon.png')
                return
            
        # Case where serie and labeled image are presents
        elif self.flag_image == True and self.flag_labeling == True:
            self.image = np.copy(self.Current_stack)
            self.GRDD_GDS() # Run the GRDD-GDS computation
            
    def GRDD_GDS(self): # Compute the GRDD and the GDS of the data imported
        self.progressBar.setVisible(True) # Show the progress bar in the main gui
        self.progressBar.setValue(0) # Set the initial value of the Progress bar at 0
        label_max = int(np.max(self.Label_image)) # Define the total number of labels
        self.progressBar.setRange(0, label_max-1) # Set the range according to the number of labels
        
        self.image = fct.centeredEuclidianNorm(self.image, ax = 0) # Normalization of the image serie
    
        # Convert labeled image as integer
        self.Labels_int = np.zeros((len(self.Label_image),len(self.Label_image[0])), dtype = int)
        
        for i in range(0,len(self.Label_image)):
            for j in range(0,len(self.Label_image[0])):
                self.Labels_int[i,j] = int(self.Label_image[i,j])
    
        # Definition of the mean profiles for each label
        moyen_profil=np.zeros((len(regionprops(self.Labels_int)),len(self.image[:,0,0])))
        self.dot=np.zeros((len(self.image[0,:,0]),len(self.image[0,0,:])))
    
        for i in range (len(self.image[:,0,0])) :
            regions = regionprops(self.Labels_int, intensity_image=self.image[i,:,:])
            for j in range (len(regions)) :
                moyen_profil[j][i]=regions[j].mean_intensity 
                
        # Normalization of the mean profiles 
        moyen_profil = fct.centeredEuclidianNorm(moyen_profil, ax = 1)
        
        # Distance between the mean profile of each label and their associated profiles
        self.dot_mean=np.zeros((len(self.image[0,:,0]),len(self.image[0,0,:])))
        compteur=0
        sum_dot=0
        compteur_interne=0
    
        for k in range (len(regions)) :
            coord=regions[k].coords
            for (y,x) in coord : #coords renvoie (y,x)
                compteur=compteur+1 
                compteur_interne=compteur_interne+1
                
                QApplication.processEvents()    
                self.ValSlice = k
                self.progression_bar()
                    
                self.dot[y,x]=1.0-np.dot(self.image[:,y,x],moyen_profil[k,:]) # Dot product computation for each pixel with the mean profiles of each labels ==> GRDD           
                sum_dot=sum_dot+self.dot[y,x]  # Compute the sum of distances in the given grain in order to compute the mean distance of each grain
                
            if compteur_interne!=0:
                dot_moyen=sum_dot/compteur_interne # Mean distance of each grain
            sum_dot=0
            compteur_interne=0
            
            for (y,x) in coord :
                self.dot_mean[y,x]=dot_moyen # GDS
            dot_moyen=0
            
        self.displayDataview(self.dot_mean) # Display the GRDD map
        
        Combo_text = '\u2022 GRDD map'
        Combo_data = self.dot
        self.choiceBox.addItem(Combo_text, Combo_data)
        
        self.StackList.append(self.dot) # Append GRDD in the stack list
        
        Combo_text = '\u2022 GDS map'
        Combo_data = self.dot_mean
        self.choiceBox.addItem(Combo_text, Combo_data)
        
        self.StackList.append(self.dot_mean) # Append GDS in the StackList
    
        self.progressBar.setVisible(False) # Hide the Progress bar
        self.Save_button.setEnabled(True)
        self.Reload_button.setEnabled(True)
        self.choiceBox.setEnabled(True)
        
        self.choiceBox.setCurrentIndex(self.choiceBox.count() - 1) # Show the last data in the choiceBox QComboBox
            
    def progression_bar(self): # Function for the ProgressBar uses
        self.prgbar = self.ValSlice
        self.progressBar.setValue(self.prgbar)
        
    def defaultdrawCHORDprofiles(self): # Default display of CHORDprofiles
        self.profiles.clear()
        self.profiles.setBackground(self.color2)
        
        self.profiles.getPlotItem().hideAxis('bottom')
        self.profiles.getPlotItem().hideAxis('left')
        
    def drawCHORDprofiles(self): # Display of CHORDprofiles
        try:
            self.profiles.clear()           
            line = self.plotIt.addLine(x = self.expSeries.currentIndex) # Line associated to the current slice
            line.setPen({'color': (42, 42, 42, 100), 'width': 2}) # Style of the line
                      
            pen = pg.mkPen(color=self.color4, width=5) # Color and line width of the profile
            self.profiles.plot(self.Current_stack[:, self.x, self.y], pen=pen) # Plot of the profile
            
            styles = {"color": "black", "font-size": "15px", "font-family": "Noto Sans Cond"} # Style for labels
            
            self.profiles.setLabel("left", "GrayScale value", **styles) # Import style for Y label
            self.profiles.setLabel("bottom", "Slice", **styles) # Import style for X label
            
            font=QtGui.QFont('Noto Sans Cond', 9) # Font definition of the plot
            
            self.profiles.getAxis("left").setTickFont(font) # Apply size of the ticks label
            self.profiles.getAxis("left").setStyle(tickTextOffset = 10) # Apply a slight offset

            self.profiles.getAxis("bottom").setTickFont(font) # Apply size of the ticks label
            self.profiles.getAxis("bottom").setStyle(tickTextOffset = 10) # Apply a slight offset
            
            self.profiles.getAxis('left').setTextPen('k') # Set the axis in black
            self.profiles.getAxis('bottom').setTextPen('k') # Set the axis in black
            
            self.profiles.setBackground(self.color2)
            self.profiles.showGrid(x=True, y=True)
            
        except:
            pass

    def mouseMoved(self, e): # Define action during mouse displacement
        pos = e[0]
        sender = self.sender()
        
        if not self.mouseLock.isChecked():
            if self.expSeries.view.sceneBoundingRect().contains(pos)\
                or self.dataview.view.sceneBoundingRect().contains(pos):
    
                if sender == self.proxy1: # If the mouse is in the Currend_stack ImageView
                    item = self.expSeries.view
                elif sender == self.proxy2: # If the mouse is in the treatment ImageView
                    item = self.dataview.view                
    
                mousePoint = item.mapSceneToView(pos) 
                     
                self.crosshair_v1.setPos(mousePoint.x())
                self.crosshair_h1.setPos(mousePoint.y())
                
                self.crosshair_v2.setPos(mousePoint.x())
                self.crosshair_h2.setPos(mousePoint.y())
    
            try:
                self.x = int(mousePoint.x())
                self.y = int(mousePoint.y())
                
                self.printClick(self.x, self.y, sender) # Give information of the current mouse position
            
                if self.x >= 0 and self.y >= 0 and self.x < len(self.Current_stack[0, :, 0]) and self.y < len(self.Current_stack[0, 0, :]):
                    self.drawCHORDprofiles()
            except:
                pass
    
    def mouseClick(self, e): # Define action during mouse click
        pos = e[0]
        sender = self.sender()
        
        self.mouseLock.toggle()
        
        fromPosX = pos.scenePos()[0]
        fromPosY = pos.scenePos()[1]
        
        posQpoint = QtCore.QPointF()
        posQpoint.setX(fromPosX)
        posQpoint.setY(fromPosY)

        if self.expSeries.view.sceneBoundingRect().contains(posQpoint)\
            or self.dataview.view.sceneBoundingRect().contains(pos):
                
            if sender == self.proxy4: # If the mouse is in the Currend_stack ImageView
                item = self.expSeries.view
            elif sender == self.proxy5: # If the mouse is in the treatment ImageView
                item = self.dataview.view    
            
            mousePoint = item.mapSceneToView(posQpoint) 

            self.crosshair_v1.setPos(mousePoint.x())
            self.crosshair_h1.setPos(mousePoint.y())
            
            self.crosshair_v2.setPos(mousePoint.x())
            self.crosshair_h2.setPos(mousePoint.y())
                 
            self.x = int(mousePoint.x())
            self.y = int(mousePoint.y())
            
            try:
                if self.x >= 0 and self.y >= 0 and self.x < len(self.Current_stack[0, :, 0])and self.y < len(self.Current_stack[0, 0, :]):
                    self.drawCHORDprofiles()
            except:
                pass
            
    def printClick(self, x, y, sender):
        self.choice = self.choiceBox.currentText() # Define the data that must be considered
        
        if self.choice == "\u2022 STD map":
            self.try_display("STD map: ",self.std_image,x,y)  
        if self.choice == "\u2022 KAD map":
            self.try_display("KAD map: ",self.KAD,x,y)
        if self.choice == "\u2022 Contour map":
            self.try_display("Contour map: ",self.contour_map,x,y)
        if self.choice == "\u2022 Grain labeling":
            self.try_display("Grain labels: ",self.Label_image,x,y)
        if self.choice == "\u2022 GRDD map":
            self.try_display("GRDD map: ",self.dot,x,y)
        if self.choice == "\u2022 GDS map":
            self.try_display("GDS map: ",self.dot_mean,x,y)
        if self.choice == "\u2022 Stitched map":
            if self.flag_stitchKAD == True:
                self.try_display("Stitched map: ",self.KAD,x,y)
            else:
                self.try_display("Stitched map: ",self.contour_map,x,y)
            
    def try_display(self,text,Current_data,x,y): # Default function to display results in the label
            try:
                self.label_Treatment.setText(text + str(np.round(Current_data[x, y],2)))
            except:
                pass
        
    def displayExpStack(self, series):
        self.expSeries.addItem(self.crosshair_v1, ignoreBounds=True) # Add the vertical crosshair
        self.expSeries.addItem(self.crosshair_h1, ignoreBounds=True) # Add the horizontal crosshair
        
        self.expSeries.ui.histogram.hide()
        self.expSeries.ui.roiBtn.hide()
        self.expSeries.ui.menuBtn.hide()
        
        view = self.expSeries.getView() # Extract view
        state = view.getState() # Extract state
        self.expSeries.setImage(series) # Add the wanted data inside the ImageView
        view.setState(state)
        
        view.setBackgroundColor(self.color1) # Define the background color
        ROIplot = self.expSeries.getRoiPlot() # Extract the ROI
        ROIplot.setBackground(self.color1) # Define the ROI background color
        
        font=QtGui.QFont('Noto Sans Cond', 8) # Set fontsize
        ROIplot.getAxis("bottom").setTextPen('k') # Apply color of the ticks label
        ROIplot.getAxis("bottom").setTickFont(font) # Apply size of the ticks label
        
        self.expSeries.timeLine.setPen(color=self.color3, width=15) # Define timeLine line color and width
        self.expSeries.frameTicks.setPen(color=self.color1, width=5) # Define ticks color and width
        self.expSeries.frameTicks.setYRange((0, 1))

        s = self.expSeries.ui.splitter
        s.handle(1).setEnabled(True) # Allow to change splitter height
        s.setStyleSheet("background: 5px white;") # Define splitter background color 
        s.setHandleWidth(5) # Define splitter width

    def displayDataview(self, series):
        self.dataview.addItem(self.crosshair_v2, ignoreBounds=True) # Add the vertical crosshair
        self.dataview.addItem(self.crosshair_h2, ignoreBounds=True) # Add the horizontal crosshair
        self.dataview.setImage(series) 
        self.dataview.autoRange()
    
        self.dataview.clear()
        self.dataview.ui.menuBtn.hide()
        self.dataview.ui.roiBtn.hide()
        self.dataview.ui.histogram.show()
        
        view = self.dataview.getView() # Extract view
        state = view.getState() # Extract state
        self.dataview.setImage(series) # Add the wanted data inside the ImageView
        view.setState(state)
        
        view.setBackgroundColor(self.color1) # Define the background color
        
        histplot = self.dataview.getHistogramWidget() # Extract histogram widget to apply modification
        histplot.setBackground(self.color1)
        
        histplot.region.setBrush(pg.mkBrush(self.color5 + (120,)))
        histplot.region.setHoverBrush(pg.mkBrush(self.color5 + (60,)))
        histplot.region.pen = pg.mkPen(self.color5)
        histplot.region.lines[0].setPen(pg.mkPen(self.color5, width=2))
        histplot.region.lines[1].setPen(pg.mkPen(self.color5, width=2))
        histplot.fillHistogram(color = self.color5)        
        histplot.autoHistogramRange()
        
        view.setBackgroundColor(self.color1) # Define the background color
        ROIplot = self.dataview.getRoiPlot() # Extract the ROI
        ROIplot.setBackground(self.color1) # Define the ROI background color
        
        font=QtGui.QFont('Noto Sans Cond', 8) # Set fontsize
        ROIplot.getAxis("bottom").setTextPen('k') # Apply color of the ticks label
        ROIplot.getAxis("bottom").setTickFont(font) # Apply size of the ticks label
        
        self.dataview.timeLine.setPen(color=self.color3, width=15) # Define timeLine line color and width
        self.dataview.frameTicks.setPen(color=self.color1, width=5) # Define ticks color and width
        self.dataview.frameTicks.setYRange((0, 1))

        s = self.dataview.ui.splitter
        s.handle(1).setEnabled(True) # Allow to change splitter height
        s.setStyleSheet("background: 5px white;") # Define splitter background color 
        s.setHandleWidth(5) # Define splitter width
        
    def defaultIV(self):
        self.expSeries.clear()
        self.expSeries.ui.histogram.hide()
        self.expSeries.ui.roiBtn.hide()
        self.expSeries.ui.menuBtn.hide()
        
        view = self.expSeries.getView()
        view.setBackgroundColor(self.color1)
        
        ROIplot = self.expSeries.getRoiPlot()
        ROIplot.setBackground(self.color1)
        
        self.dataview.clear()
        self.dataview.ui.histogram.hide()
        self.dataview.ui.roiBtn.hide()
        self.dataview.ui.menuBtn.hide()
        
        view = self.dataview.getView()
        view.setBackgroundColor(self.color1)
        
        self.label_Treatment.setText("Treatment")

def main():

	app = QApplication(sys.argv)
	w = MainWindow()
	w.show()
	app.setQuitOnLastWindowClosed(True)
	app.exec_() 
		
#%% Opening of the initial data    
if __name__ == '__main__':
	main() 