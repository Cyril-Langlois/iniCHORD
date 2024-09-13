

import os
from os.path import abspath

from inspect import getsourcefile
import numpy as np
import tifffile as tf
import cv2
import largestinteriorrectangle as lir
import scipy.ndimage as sci
from pystackreg import StackReg

from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMessageBox

import general_functions as gf

path2thisFile = abspath(getsourcefile(lambda:0))
uiclass, baseclass = pg.Qt.loadUiType(os.path.dirname(path2thisFile) + "/Aligned_pg_ui_12_TSG.ui")

class MainWindow(uiclass, baseclass):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setupUi(self)
        
        self.setWindowIcon(QtGui.QIcon('icons/alignment_icon.png'))
        
        self.expStack = parent.Current_stack
        self.Pre_treatment_stack = np.copy(parent.Current_stack) # Stack that will be used to process the pretreatment 
        
        self.type = self.expStack.dtype

        if self.type == "uint16" or self.type == "uint32":
            self.expStack = gf.convertToUint8(self.expStack) # Necessary for computation
            self.Pre_treatment_stack = gf.convertToUint8(self.Pre_treatment_stack)

        self.Pre_treatment_stack_Remout = np.copy(self.Pre_treatment_stack)
        self.Pre_treatment_stack_output = np.copy(self.Pre_treatment_stack)
        self.Pre_treatment_stack_output2 = np.copy(self.Pre_treatment_stack)

        self.Pre_treatment_doppelstack = np.copy(self.Pre_treatment_stack)
        self.Pre_treatment_doppelstack_Remout = np.copy(self.Pre_treatment_stack)
        self.Pre_treatment_doppelstack_output = np.copy(self.Pre_treatment_stack)
        self.Pre_treatment_doppelstack_output2 = np.copy(self.Pre_treatment_stack)
        
        # Stack that will store the final result
        self.Aligned_stack = np.copy(parent.Current_stack)

        self.blur_value = int(self.Blur_box.currentText())
        self.sobel_value = int(self.Sobel_box.currentText())

        self.im_reference = 0 # n° of the reference image (for sequential approach)
        self.iter_value = 2000 # Number of iteration to converged
        self.thres_value = 0.000001 # Convergence value 
        
        self.img_number = len(self.expStack)
        self.stack_height = len(self.expStack[0])
        self.stack_width = len(self.expStack[0][0])
        
        self.radius_Val = self.Radius_slider.value() # Input filtre Remout
        self.threshold_Val = self.Threshold_slider.value()  # Input filtre Remout
        
        self.Label_radius.setText("Radius: " + str(self.radius_Val)) # Input filtre Remout
        self.Label_threshold.setText("Threshold: " + str(self.threshold_Val)) # Input filtre Remout
        
        self.Radius_slider.valueChanged.connect(self.radius_changed) # Input filtre Remout
        self.Threshold_slider.valueChanged.connect(self.threshold_changed) # Input filtre Remout
        
        self.Blur_box.activated.connect(self.get_blur_changed) # Gaussian blur changes connexion
        self.Sobel_box.activated.connect(self.get_sobel_changed) # Sobel changes connexion
        
        self.push_compute.clicked.connect(self.Choice_method)
        self.Push_valid.clicked.connect(self.extract_align_stack) # Extract data to the main gui
        self.Push_valid.setEnabled(False) # Validation Unauthorized at the beginning 
        
        self.progressBar.setValue(0) # Progress bar initial value is set to 0
        self.progressBar.setRange(0, self.img_number-1)
        
        self.checkBox.clicked.connect(self.checking) # Check if the registration must be done on the raw stack or on the pretreated one
        self.checkBox_Uncropped.clicked.connect(self.checking_Uncrop) # Check if the final result is cropped or no
        self.Keep_param.clicked.connect(self.Pre_treatment) # apply pretreatment parameter to all slices
        
        if self.checkBox_Uncropped.isChecked() == False: 
            self.flagCrop = "No"
        
        self.Pre_treatment_slice() # Run a pretreatment on 1/10 images
        
        self.transfo_comboBox.setCurrentIndex(4) 
        self.wrapmode  = cv2.MOTION_HOMOGRAPHY
        self.textEdit.insertPlainText("\n Homographic transformation used to estim correlation.")
        self.flagTransformation = "Homography"
         
        self.CC_info.setText('Estimated correlation coefficient.') # Information about the correlation coefficient between 2 images (pretreated)
        self.textEdit.setFontPointSize(10)
        
        self.Coefficient_estim() # Estim corrcoeff between the 2 first images (Pretreated)
        
        self.defaultdrawCC()
        self.displayExpStack(self.expStack)
        self.display_Aligne_treatment(self.Aligned_stack)
        
        app = QApplication.instance()
        screen = app.screenAt(self.pos())
        geometry = screen.availableGeometry()
        
        # Position (self.move) and size (self.resize) of the main GUI on the screen
        self.move(int(geometry.width() * 0.1), int(geometry.height() * 0.15))
        self.resize(int(geometry.width() * 0.7), int(geometry.height() * 0.6))
        self.screen = screen

#%% Functions : initialization
    def checking(self): # Allow to define if registration must be applied to the raw stack or to the pretreated one
        if self.checkBox.isChecked() == True:
            self.checkBox.clicked.connect(self.Pre_notreatment) # Registration on the raw stack
            self.textEdit.insertPlainText("\n Registration performs on raw images.")
        elif self.checkBox.isChecked() == False:
            self.Blur_box.activated.connect(self.radius_changed) 
            self.Sobel_box.activated.connect(self.threshold_changed)
            self.Blur_box.activated.connect(self.get_blur_changed) 
            self.Sobel_box.activated.connect(self.get_sobel_changed)
            self.textEdit.insertPlainText("\n Registration performs using the treated stack.")

    def checking_Uncrop(self): # Define if the final stack must be cropped
        if self.checkBox_Uncropped.isChecked() == True:
            self.textEdit.insertPlainText("\n Uncropped stack will be saved.")
            self.flagCrop = "Yes" 
        elif self.checkBox_Uncropped.isChecked() == False:
            self.textEdit.insertPlainText("\n Cropped stack will be saved.")
            self.flagCrop = "No"

    def warp_mode_selection(self): # Define the wrapmode dimensions depending if an affine or an homographic trasnformation is used
        if self.choice_transfo == "Affine":
            self.wrapmode  = cv2.MOTION_AFFINE
            self.flagTransformation = "Affine"
        elif self.choice_transfo == "Homography":
            self.wrapmode  = cv2.MOTION_HOMOGRAPHY
            self.flagTransformation = "Homography"
        # else :
        #     self.wrapmode  = cv2.MOTION_HOMOGRAPHY
        #     self.textEdit.insertPlainText("\n CoReg°: homography will be used.")
        #     self.flagTransformation = "Homography"
        
        return self.wrapmode
               
    def radius_changed(self): # apply remove outliers radius modification
        value = self.Radius_slider.value()
        self.radius_Val = value
        self.Label_radius.setText("Radius: " + str(self.radius_Val)) 
        self.Coefficient_estim()
        self.Pre_treatment_slice()

    def threshold_changed(self):  # Apply remove outliers threshold modification
        value = self.Threshold_slider.value()
        self.threshold_Val = value
        self.Label_threshold.setText("Threshold: " + str(self.threshold_Val))
        self.Coefficient_estim()
        self.Pre_treatment_slice()  
        
    def get_blur_changed(self):  # Apply Gaussian blur modification
        value = self.Blur_box.currentText()
        self.blur_value = int(value)
        self.Coefficient_estim()
        self.Pre_treatment_slice()

    def get_sobel_changed(self): # Apply Sobel factor modification
        value = self.Sobel_box.currentText()
        self.sobel_value = int(value)
        self.Coefficient_estim()
        self.Pre_treatment_slice()

    def defaultdrawCC(self): # Default display of the coefficient correlation
        self.RecapCC.clear()
        self.RecapCC.setBackground(self.parent.color2)
        
        self.RecapCC.getPlotItem().hideAxis('bottom')
        self.RecapCC.getPlotItem().hideAxis('left')

    def drawRecapCC(self,recap): # Plot Corrcoeff during registration (affine and homography only)
        self.RecapCC.clear()
                  
        pen = pg.mkPen(color=self.parent.color4, width=5) # Color and line width of the profile
        self.RecapCC.plot(recap[:, 0], pen=pen) # Plot of the profile
        
        styles = {"color": "black", "font-size": "40px", "font-family": "Noto Sans Cond"} # Style for labels
        self.RecapCC.setLabel("left", "Coeff°", **styles) # Import style for Y label
        self.RecapCC.setLabel("bottom", "Slice", **styles) # Import style for X label
        
        font=QtGui.QFont('Noto Sans Cond', 9)
        
        self.RecapCC.getAxis("left").setTickFont(font) # Apply size of the ticks label
        self.RecapCC.getAxis("left").setStyle(tickTextOffset = 20) # Apply a slight offset
        self.RecapCC.getAxis("bottom").setTickFont(font) # Apply size of the ticks label
        self.RecapCC.getAxis("bottom").setStyle(tickTextOffset = 20) # Apply a slight offset
        
        self.RecapCC.getAxis('left').setTextPen('k') # Set the axis in black
        self.RecapCC.getAxis('bottom').setTextPen('k') # Set the axis in black
        
        self.RecapCC.setBackground(self.parent.color2)
        self.RecapCC.showGrid(x=True, y=True)

    def Pre_notreatment(self): # Steps prior registration is no pretreatment
        self.Pre_treatment_stack_output = np.copy(self.Pre_treatment_stack)
        self.Pre_treatment_stack_output2 = self.Pre_treatment_stack_output
        self.display_Pre_treatment(self.Pre_treatment_stack_output2)
        self.Pre_treatment_stack_output2 = self.Pre_treatment_stack_output2
    
    def Pre_treatment_slice(self): # Apply pretreatment on 1/10 slices at the opening of the sub-gui
        self.Pre_treatment_stack_output = np.copy(self.Pre_treatment_stack)
        self.Pre_treatment_stack_Remout = np.copy(self.Pre_treatment_stack)
        
        if len(self.Pre_treatment_stack_Remout) < 10:
            self.parent.popup_message("Registration","A minimum of 10 images is needed for registration. Please, considerer more images",'icons/Main_icon.png')
            return
        
        else: # Here, remouve outliers, Gaussian blur and sobel factor are applied
            for i in range(0,int(len(self.Pre_treatment_stack)),int(len(self.Pre_treatment_stack)/10)): # Applique pour chaque slice les paramètres du remove outlier
                _, self.Pre_treatment_stack_Remout[i, :, :] =  gf.remove_outliers(self.Pre_treatment_stack[i,:,:], self.radius_Val, self.threshold_Val)
            
            if self.blur_value != 0 :
                for i in range(0,int(len(self.Pre_treatment_stack)),int(len(self.Pre_treatment_stack)/10)):
                    self.Pre_treatment_stack_output[i,:,:] =cv2.GaussianBlur(self.Pre_treatment_stack_Remout[i,:,:],(self.blur_value,self.blur_value),0)
            else :
                self.Pre_treatment_stack_output = self.Pre_treatment_stack_Remout
            
            if self.sobel_value != 0 :
                for i in range(0,int(len(self.Pre_treatment_stack)),int(len(self.Pre_treatment_stack)/10)):
                    self.grad_x = cv2.Sobel(self.Pre_treatment_stack_output[i,:,:],cv2.CV_32F,1,0,ksize=self.sobel_value)
                    self.grad_y = cv2.Sobel(self.Pre_treatment_stack_output[i,:,:],cv2.CV_32F,0,1,ksize=self.sobel_value)
                
                    self.Pre_treatment_stack_output2[i,:,:] = cv2.addWeighted(np.absolute(self.grad_x), 0.5, np.absolute(self.grad_y), 0.5, 0)
            else :   
                self.Pre_treatment_stack_output2 = self.Pre_treatment_stack_output
         
            self.Pre_treatment_stack_slice = self.Pre_treatment_stack_output2[0:-1:int(len(self.Pre_treatment_stack)/10),:,:]
            self.display_Pre_treatment(self.Pre_treatment_stack_slice)
        
    def Pre_treatment(self): # Apply pretreatment on the entire stack
        self.Pre_treatment_stack_output = np.copy(self.Pre_treatment_stack)
        self.Pre_treatment_stack_Remout = np.copy(self.Pre_treatment_stack)
        
        for i in range(0,len(self.Pre_treatment_stack[:, 0, 0])): # Applique pour chaque slice les paramètres du remove outlier
            _, self.Pre_treatment_stack_Remout[i, :, :] =  gf.remove_outliers(self.Pre_treatment_stack[i,:,:], self.radius_Val, self.threshold_Val)
        
        if self.blur_value != 0 :
            for i in range(0,len(self.Pre_treatment_stack)):
                self.Pre_treatment_stack_output[i,:,:] =cv2.GaussianBlur(self.Pre_treatment_stack_Remout[i,:,:],(self.blur_value,self.blur_value),0)
        else :
            self.Pre_treatment_stack_output = self.Pre_treatment_stack_Remout
            
        if self.sobel_value != 0 :
            for i in range(0,len(self.Pre_treatment_stack)):
                self.grad_x = cv2.Sobel(self.Pre_treatment_stack_output[i,:,:],cv2.CV_32F,1,0,ksize=self.sobel_value)
                self.grad_y = cv2.Sobel(self.Pre_treatment_stack_output[i,:,:],cv2.CV_32F,0,1,ksize=self.sobel_value)
            
                self.Pre_treatment_stack_output2[i,:,:] = cv2.addWeighted(np.absolute(self.grad_x), 0.5, np.absolute(self.grad_y), 0.5, 0)
        else :   
            self.Pre_treatment_stack_output2 = self.Pre_treatment_stack_output
     
        self.display_Pre_treatment(self.Pre_treatment_stack_output2)
        self.textEdit.insertPlainText("\n The stack is ready for registration.")

    def Pre_treatment_doppel(self): # Apply pretreatment on the doppelganger stack (for co-registration)
        self.doppelganger_stack = np.flip(self.doppelganger_stack, 1)
        self.doppelganger_stack = np.rot90(self.doppelganger_stack, k=1, axes=(2, 1))
        
        self.type = self.doppelganger_stack.dtype

        if self.type == "uint16" or self.type == "uint32":
            self.doppelganger_stack = gf.convertToUint8(self.doppelganger_stack)
               
        self.Pre_treatment_doppelstack = np.copy(self.doppelganger_stack)
        self.Pre_treatment_doppelstack_Remout = np.copy(self.doppelganger_stack)
        
        for i in range(0,len(self.Pre_treatment_doppelstack[:, 0, 0])): # Applique pour chaque slice les paramètres du remove outlier
            _, self.Pre_treatment_doppelstack_Remout[i, :, :] =  gf.remove_outliers(self.Pre_treatment_doppelstack[i,:,:], self.radius_Val, self.threshold_Val)
        
        if self.blur_value != 0 :
            for i in range(0,len(self.Pre_treatment_doppelstack)):
                self.Pre_treatment_doppelstack_output[i,:,:] =cv2.GaussianBlur(self.Pre_treatment_doppelstack_Remout[i,:,:],(self.blur_value,self.blur_value),0)
        else :
            self.Pre_treatment_doppelstack_output = self.Pre_treatment_doppelstack_Remout
        
        if self.sobel_value != 0 :
            for i in range(0,len(self.Pre_treatment_doppelstack)):
                self.grad_x = cv2.Sobel(self.Pre_treatment_doppelstack_output[i,:,:],cv2.CV_32F,1,0,ksize=self.sobel_value)
                self.grad_y = cv2.Sobel(self.Pre_treatment_doppelstack_output[i,:,:],cv2.CV_32F,0,1,ksize=self.sobel_value)
            
                self.Pre_treatment_doppelstack_output2[i,:,:] = cv2.addWeighted(np.absolute(self.grad_x), 0.5, np.absolute(self.grad_y), 0.5, 0)
     
        else :   
            self.Pre_treatment_doppelstack_output2 = self.Pre_treatment_doppelstack

    def Coefficient_estim(self): # Estim corrcoeff between the two first image (using pretreated slices and homography)
        self.Pre_treatment_stack_output = np.copy(self.Pre_treatment_stack)
        self.Pre_treatment_stack_Remout = np.copy(self.Pre_treatment_stack)
        
        for i in range(0,2): # Applique pour chaque slice les paramètres du remove outlier
            _, self.Pre_treatment_stack_Remout[i, :, :] =  gf.remove_outliers(self.Pre_treatment_stack[i,:,:], self.radius_Val, self.threshold_Val)
        
        if self.blur_value != 0 :
            for i in range(0,2):
                self.Pre_treatment_stack_output[i,:,:] =cv2.GaussianBlur(self.Pre_treatment_stack_Remout[i,:,:],(self.blur_value,self.blur_value),0)
        else :
            self.Pre_treatment_stack_output = self.Pre_treatment_stack_Remout
        
        if self.sobel_value != 0 :
            for i in range(0,2):
                self.grad_x = cv2.Sobel(self.Pre_treatment_stack_output[i,:,:],cv2.CV_32F,1,0,ksize=self.sobel_value)
                self.grad_y = cv2.Sobel(self.Pre_treatment_stack_output[i,:,:],cv2.CV_32F,0,1,ksize=self.sobel_value)
            
                self.Pre_treatment_stack_output2[i,:,:] = cv2.addWeighted(np.absolute(self.grad_x), 0.5, np.absolute(self.grad_y), 0.5, 0)
        else :   
            self.Pre_treatment_stack_output2 = self.Pre_treatment_stack_output
             
        self.warp_mode = self.wrapmode
        if self.warp_mode  == cv2.MOTION_HOMOGRAPHY:
            self.warp =  np.eye(3, 3, dtype=np.float32) # Matrice pour le stockage des transformations homographiques
        else :
            self.warp =  np.eye(2, 3, dtype=np.float32) # Matrice pour le stockage des transformations affines

        self.im1 = self.Pre_treatment_stack_output2[self.im_reference,:,:] # Définition de la slice de référence
        self.im2 = self.Pre_treatment_stack_output2[self.im_reference +1,:,:]
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.iter_value, self.thres_value) # Stockage des critères pour l'alignement
        
        self.Recap_cc = []

        try :   
            self.cc, self.warp = cv2.findTransformECC(self.im1, self.im2, self.warp, self.warp_mode, self.criteria)
            self.cc = '%.3f'%(self.cc)
        except :
            QApplication.processEvents()
            self.textEdit.insertPlainText("\n Impossible estimation. Try other parameters.")

        self.CC_info.setText('Correlation coefficient: ' + self.cc)

    def Choice_method(self): # Define the program that must be use to perform registration (between sequential, incremental and coregistration)
        self.choice = self.choiceBox.currentText()
        
        if self.choice == 'Sequential':
            self.Registration_seq_step() 
        elif self.choice == 'Incremental':
            self.Registration_incre_step()
        elif self.choice == 'Coregistration':
            self.Registration_coreg_step()

#%% Functions : registration
    def Registration_seq_step(self): # Step to perform sequential registration
        self.choice_transfo = self.transfo_comboBox.currentText()
        if self.choice_transfo == "Translation" or self.choice_transfo == "Scaled rotation" or self.choice_transfo == "Rigid body" :
            self.Aligned_stack = self.Seq_registration()
            self.AlignedStack.ui.roiBtn.hide()
            self.AlignedStack.ui.histogram.hide()
            self.display_Aligne_treatment(self.Aligned_stack)
            self.textEdit.insertPlainText("\n Sequential registration is complete.")
            self.Push_valid.setEnabled(True) # Validation button is enables 
        else:
            self.Aligned_stack, self.Recap_cc, self.Recap_warp = self.Seq_registration()
            self.AlignedStack.ui.roiBtn.hide()
            self.AlignedStack.ui.histogram.hide()
            self.display_Aligne_treatment(self.Aligned_stack)
            self.drawRecapCC(self.Recap_cc)
            self.textEdit.insertPlainText("\n Sequential registration is complete.")
            self.Push_valid.setEnabled(True) # Validation button is enables 
        
    def Registration_incre_step(self):  # Step to perform incremental registration
        self.choice_transfo = self.transfo_comboBox.currentText()
        if self.choice_transfo == "Translation" or self.choice_transfo == "Scaled rotation" or self.choice_transfo == "Rigid body" :
            self.Aligned_stack = self.Incre_registration()
            self.AlignedStack.ui.roiBtn.hide()
            self.AlignedStack.ui.histogram.hide()
            self.display_Aligne_treatment(self.Aligned_stack)
            self.textEdit.insertPlainText("\n Incremental registration is complete.")
            self.Push_valid.setEnabled(True) # Validation button is enables 
        else:
            self.Aligned_stack, self.Recap_cc, self.Recap_warp = self.Incre_registration()
            self.AlignedStack.ui.roiBtn.hide()
            self.AlignedStack.ui.histogram.hide()
            self.display_Aligne_treatment(self.Aligned_stack)
            self.drawRecapCC(self.Recap_cc)
            self.textEdit.insertPlainText("\n Incremental registration is complete.")
            self.Push_valid.setEnabled(True) # Validation button is enables 

    def Registration_coreg_step(self):  # Step to perform Co-registration
        self.parent.popup_message("Registration","Only homographic transformation will be used for the Co-registration approach",'icons/Main_icon.png')

        StackLoc_doppel, StackDir_doppel  = gf.getFilePathDialog("Choose the already aligned stack that will be used.' (*.tiff)")
        self.doppelganger_stack = tf.TiffFile(StackLoc_doppel[0]).asarray()       
        self.Pre_treatment_doppel()
        
        self.Aligned_stack, self.Recap_cc, self.Recap_warp = self.coreg_registration()
        self.AlignedStack.ui.roiBtn.hide()
        self.AlignedStack.ui.histogram.hide()
        self.display_Aligne_treatment(self.Aligned_stack)
        self.drawRecapCC(self.Recap_cc)
        self.textEdit.insertPlainText("\n Coregistration is complete.")
        self.Push_valid.setEnabled(True) # Validation button is enables 

    def Seq_registration(self): # Sequential registration
        QApplication.processEvents()
        self.textEdit.insertPlainText("\n ------------------------------------")
        self.textEdit.insertPlainText("\n Sequential registration is running.")
        
        self.Aligned_stack = np.copy(self.expStack)
        
        self.choice_transfo = self.transfo_comboBox.currentText()
        self.flagTransformation = self.choice_transfo
        
        QApplication.processEvents()
        self.textEdit.insertPlainText("\n Transformation: " + str(self.flagTransformation))
        
        # Cases of Translation - Scaled rotation - Rigid body
        if self.choice_transfo == "Translation":
            sr = StackReg(StackReg.TRANSLATION)
            tmat = sr.register_stack(self.Pre_treatment_stack_output2, axis=0, reference='first',  progress_callback=self.show_progress)
            self.Aligned_stack = sr.transform_stack(self.Aligned_stack)
            return self.Aligned_stack
        elif self.choice_transfo == "Scaled rotation":
            sr = StackReg(StackReg.SCALED_ROTATION)
            tmat = sr.register_stack(self.Pre_treatment_stack_output2, axis=0, reference='first',  progress_callback=self.show_progress)
            self.Aligned_stack = sr.transform_stack(self.Aligned_stack)
            return self.Aligned_stack
        elif self.choice_transfo == "Rigid body":
            sr = StackReg(StackReg.RIGID_BODY)
            tmat = sr.register_stack(self.Pre_treatment_stack_output2, axis=0, reference='first',  progress_callback=self.show_progress)
            self.Aligned_stack = sr.transform_stack(self.Aligned_stack)
            return self.Aligned_stack
        
        # Case of affine and homography
        else:
            self.warp_mode_selection()
            
            self.Aligned_stack = np.zeros((len(self.Pre_treatment_stack_output2), len(self.Pre_treatment_stack_output2[0]), len(self.Pre_treatment_stack_output2[0][0])), dtype = np.float32)

            self.im1 = self.Pre_treatment_stack_output2[self.im_reference,:,:] # Définition de la slice de référence
    
            self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.iter_value, self.thres_value) # Stockage des critères pour l'alignement
            
            self.Recap_cc = []
            
            self.warp_mode = self.wrapmode
            if self.warp_mode  == cv2.MOTION_HOMOGRAPHY:
                self.warp =  np.eye(3, 3, dtype=np.float32) # Matrice pour le stockage des transformations homographiques
                self.Recap_warp = np.zeros((len(self.expStack),len(self.warp),len(self.warp[0])))
            else :
                self.warp =  np.eye(2, 3, dtype=np.float32) # Matrice pour le stockage des transformations affines
                self.Recap_warp = np.zeros((len(self.expStack),len(self.warp),len(self.warp[0])))
    
            try :
                for i in np.arange(0,self.img_number, 1):
    
                    if i == self.im_reference:
                        self.Aligned_stack[i,:,:] = self.expStack[self.im_reference,:,:]
                    else :
                        self.im2 = self.Pre_treatment_stack_output2[i,:,:]
                        # Recherche des transformations à appliquer pour passer de im1 (ref) et im2 (slice)
                        self.cc, self.warp = cv2.findTransformECC(self.im1, self.im2, self.warp, self.warp_mode, self.criteria)
                        
                        # Application des transformations déterminées pour obtenir une slice alignée par rapport à im1
                        self.Recap_warp[i,:,:] = self.warp
                        self.Recap_cc.append(self.cc) 
                        self.cc = '%.3f'%(self.cc)
                        
                        QApplication.processEvents()                    
                        self.ValSlice = i
                        self.progression_bar()
                        
                        self.Recap_cc_IRT = np.vstack(self.Recap_cc)
                        self.drawRecapCC(self.Recap_cc_IRT)
                            
                        if self.warp_mode  == cv2.MOTION_HOMOGRAPHY:
                            self.Aligned_stack[i,:,:] = cv2.warpPerspective (self.expStack[i,:,:], self.warp, ( self.stack_width,self.stack_height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)          
                            self.Pre_treatment_stack_output2[i,:,:] = cv2.warpPerspective (self.Pre_treatment_stack_output2[i,:,:], self.warp, ( self.stack_width,self.stack_height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                        else :
                            self.Aligned_stack[i,:,:] = cv2.warpAffine (self.expStack[i,:,:], self.warp, ( self.stack_width,self.stack_height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                            self.Pre_treatment_stack_output2[i,:,:] = cv2.warpAffine (self.Pre_treatment_stack_output2[i,:,:], self.warp, ( self.stack_width,self.stack_height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
            except :
                QApplication.processEvents()
                self.textEdit.insertPlainText("\n Sequential registration failed. Try other parameters.")
            
            self.Recap_cc = np.vstack(self.Recap_cc)
    
            return self.Aligned_stack, self.Recap_cc, self.Recap_warp

    def Incre_registration(self): # Incremental registration
        self.textEdit.insertPlainText("\n ------------------------------------")
        self.textEdit.insertPlainText("\n Incremental registration is running.")
        
        QApplication.processEvents()
        self.textEdit.insertPlainText("\n Transformation: " + str(self.flagTransformation))
        
        self.Aligned_stack = np.copy(self.expStack)
        
        self.choice_transfo = self.transfo_comboBox.currentText()
        self.flagTransformation = self.choice_transfo
        
        # Cases of Translation - Scaled rotation - Rigid body
        if self.choice_transfo == "Translation":
            sr = StackReg(StackReg.TRANSLATION)
            tmat = sr.register_stack(self.Pre_treatment_stack_output2, axis=0, reference='previous',  progress_callback=self.show_progress)
            self.Aligned_stack = sr.transform_stack(self.Aligned_stack)
            return self.Aligned_stack
        elif self.choice_transfo == "Scaled rotation":
            sr = StackReg(StackReg.SCALED_ROTATION)
            tmat = sr.register_stack(self.Pre_treatment_stack_output2, axis=0, reference='previous',  progress_callback=self.show_progress)
            self.Aligned_stack = sr.transform_stack(self.Aligned_stack)
            return self.Aligned_stack
        elif self.choice_transfo == "Rigid body":
            sr = StackReg(StackReg.RIGID_BODY)
            tmat = sr.register_stack(self.Pre_treatment_stack_output2, axis=0, reference='previous',  progress_callback=self.show_progress)
            self.Aligned_stack = sr.transform_stack(self.Aligned_stack)
            return self.Aligned_stack
        
        # Case of affine and homography
        else:
            self.warp_mode_selection()
            
            self.Aligned_stack = np.zeros((len(self.Pre_treatment_stack_output2), len(self.Pre_treatment_stack_output2[0]), len(self.Pre_treatment_stack_output2[0][0])), dtype = np.float32)
    
            self.warp_mode = self.wrapmode
            if self.warp_mode  == cv2.MOTION_HOMOGRAPHY:
                self.warp =  np.eye(3, 3, dtype=np.float32) # Matrice pour le stockage des transformations homographiques
                self.Recap_warp = np.zeros((len(self.expStack),len(self.warp),len(self.warp[0])))
            else :
                self.warp =  np.eye(2, 3, dtype=np.float32) # Matrice pour le stockage des transformations affines
                self.Recap_warp = np.zeros((len(self.expStack),len(self.warp),len(self.warp[0])))
    
            self.im1 = self.Pre_treatment_stack_output2[self.im_reference,:,:] # Définition de la slice de référence
    
            self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.iter_value, self.thres_value) # Stockage des critères pour l'alignement
            
            self.Recap_cc = []
    
            try :
                for i in np.arange(0,self.img_number, 1):
    
                    if i == self.im_reference:
                        self.Aligned_stack[i,:,:] = self.expStack[self.im_reference,:,:]
                    else :
                        self.im1bis = self.Pre_treatment_stack_output2[i-1,:,:] # Label de la slice de référence (glissante)
                        self.im2 = self.Pre_treatment_stack_output2[i,:,:]
                        # Recherche des transformations à appliquer pour passer de im1 (ref) et im2 (slice)
                        self.cc, self.warp = cv2.findTransformECC(self.im1bis, self.im2, self.warp, self.warp_mode, self.criteria)
                        
                        # Application des transformations déterminées pour obtenir une slice alignée par rapport à im1
                        self.Recap_warp[i,:,:] = self.warp
                        self.Recap_cc.append(self.cc) 
                        self.cc = '%.3f'%(self.cc)
    
                        QApplication.processEvents()
                        self.ValSlice = i
                        self.progression_bar()
                        
                        self.Recap_cc_IRT = np.vstack(self.Recap_cc)
                        self.drawRecapCC(self.Recap_cc_IRT)
                            
                        if self.warp_mode  == cv2.MOTION_HOMOGRAPHY:
                            self.Aligned_stack[i,:,:] = cv2.warpPerspective (self.expStack[i,:,:], self.warp, ( self.stack_width,self.stack_height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                            self.Pre_treatment_stack_output2[i,:,:] = cv2.warpPerspective (self.Pre_treatment_stack_output2[i,:,:], self.warp, ( self.stack_width,self.stack_height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                        else :
                            self.Aligned_stack[i,:,:] = cv2.warpAffine (self.expStack[i,:,:], self.warp, ( self.stack_width,self.stack_height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                            self.Pre_treatment_stack_output2[i,:,:] = cv2.warpAffine (self.Pre_treatment_stack_output2[i,:,:], self.warp, ( self.stack_width,self.stack_height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    
            except :
                QApplication.processEvents()
                self.textEdit.insertPlainText("\n Incremental registration failed. Try other parameters.")
            
            self.Recap_cc = np.vstack(self.Recap_cc)
    
            return self.Aligned_stack, self.Recap_cc, self.Recap_warp

    def coreg_registration(self): # Co-registration (homographic transformation only)
        self.textEdit.insertPlainText("\n ------------------------------------")
        self.textEdit.insertPlainText("\n Coregistration is running.")
        
        self.Aligned_stack = np.zeros((len(self.Pre_treatment_stack_output2), len(self.Pre_treatment_stack_output2[0]), len(self.Pre_treatment_stack_output2[0][0])), dtype = np.float32)
        
        self.wrapmode  = cv2.MOTION_HOMOGRAPHY
        self.textEdit.insertPlainText("\n CoReg°: homography will be used.")
        self.flagTransformation = "Homography"
        
        if self.warp_mode  == cv2.MOTION_HOMOGRAPHY:
            self.warp =  np.eye(3, 3, dtype=np.float32) # Matrice pour le stockage des transformations homographiques
            self.Recap_warp = np.zeros((len(self.expStack),len(self.warp),len(self.warp[0])))
        else :
            self.warp =  np.eye(2, 3, dtype=np.float32) # Matrice pour le stockage des transformations affines
            self.Recap_warp = np.zeros((len(self.expStack),len(self.warp),len(self.warp[0])))

        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.iter_value, self.thres_value) # Stockage des critères pour l'alignement
        
        self.Recap_cc = []
        
        try :
            for i in np.arange(0,self.img_number, 1):

                if i == self.im_reference:
                    self.Aligned_stack[i,:,:] = self.expStack[self.im_reference,:,:]
                else :
                    self.im_stack = self.Pre_treatment_stack_output2[i,:,:] # Label de la slice à aligner (glissante)
                    self.im_doppel = self.Pre_treatment_doppelstack_output2[i,:,:] # Label de la slice de référence (glissante)

                    # Recherche des transformations à appliquer pour passer de im1 (ref) et im2 (slice)
                    self.cc, self.warp = cv2.findTransformECC(self.im_doppel, self.im_stack, self.warp, self.warp_mode, self.criteria)
                        
                    # Application des transformations déterminées pour obtenir une slice alignée par rapport à im1
                    self.Recap_warp[i,:,:] = self.warp
                    self.Recap_cc.append(self.cc) 
                    self.cc = '%.3f'%(self.cc)

                    QApplication.processEvents()    
                    self.ValSlice = i
                    self.progression_bar()
                    
                    self.Recap_cc_IRT = np.vstack(self.Recap_cc)
                    self.drawRecapCC(self.Recap_cc_IRT)
                            
                    if self.warp_mode  == cv2.MOTION_HOMOGRAPHY:
                        self.Aligned_stack[i,:,:] = cv2.warpPerspective (self.expStack[i,:,:], self.warp, ( self.stack_width,self.stack_height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                        self.Pre_treatment_stack_output2[i,:,:] = cv2.warpPerspective (self.Pre_treatment_stack_output2[i,:,:], self.warp, ( self.stack_width,self.stack_height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    else :
                        self.Aligned_stack[i,:,:] = cv2.warpAffine (self.expStack[i,:,:], self.warp, ( self.stack_width,self.stack_height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                        self.Pre_treatment_stack_output2[i,:,:] = cv2.warpAffine (self.Pre_treatment_stack_output2[i,:,:], self.warp, ( self.stack_width,self.stack_height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    
        except :
            self.textEdit.insertPlainText("\n Coregistration failed. Try other parameters.")
        
        self.Recap_cc = np.vstack(self.Recap_cc)

        return self.Aligned_stack, self.Recap_cc, self.Recap_warp

#%% Functions : final treatment
    def Crop_data(self): # Step to crop stack
        self.Cropped_min_stack(self.Aligned_stack)
        
        self.Cropped_stack = self.Aligned_stack[:,self.Selection[1]:self.Selection[3],self.Selection[0]:self.Selection[2]]   # Application sur la stack alignée pour rogner
        self.Cropped_pre_treatment_stack = self.Pre_treatment_stack_output2[:,self.Selection[1]:self.Selection[3],self.Selection[0]:self.Selection[2]]   # Application sur la stack alignée pour rogner

        self.textEdit.insertPlainText("\n Data has been cropped.")
        
        self.AlignedStack.ui.roiBtn.hide()
        self.AlignedStack.ui.histogram.hide()
        self.display_Aligne_treatment(self.Cropped_stack)
        
    def Cropped_min_stack(self, stack): # Step to define the position of the largest rectangle to be cropped
        self.Min_stack = np.min(stack,0) # Define min map
        self.Min_stack = self.Mask_min(self.Min_stack,0) # If value is not 0, else it became 1
        self.Min_stack = self.Min_stack.astype("bool") # Convert to Boolean
        self.Min_stack = sci.binary_fill_holes(self.Min_stack).astype("bool") # Fill holes to optimize the cropping step
    
        self.Selection = lir.lir(self.Min_stack) # Define the dimension of the largest rectangle 
        
    def Mask_min(self,stack,threshold): # If value is not 0, else it became 1
        self.Mask = np.zeros((len(stack),len(stack[0])))
        self.Mask[stack > threshold] = 1

        return self.Mask
        
    def extract_align_stack(self): # Extract registered stack to the main gui
    
        if self.flagCrop == "No" :
            self.Crop_data()
            
            self.parent.Current_stack = np.copy(self.Cropped_stack)
            self.parent.StackList.append(self.Cropped_stack)
            
            Combo_text = '\u2022 Registered stack'
            # Combo_text = '\u2022 Registered stack. Transformation : ' + self.flagTransformation + ' ; ' + self.choice
            Combo_data = self.Cropped_stack
            self.parent.choiceBox.addItem(Combo_text, Combo_data)
    
            self.parent.displayExpStack(self.parent.Current_stack)
            
            self.parent.Info_box.ensureCursorVisible()
            self.parent.Info_box.insertPlainText("\n \u2022 Registered stack has been exported.")
            
            self.close()
            
        elif self.flagCrop == "Yes" :
            self.parent.Current_stack = np.copy(self.Aligned_stack)
            self.parent.StackList.append(self.Aligned_stack)
            
            Combo_text = '\u2022 Registered stack. Transformation : ' + self.flagTransformation + ' ; ' + self.choice
            Combo_data = self.Aligned_stack
            self.parent.choiceBox.addItem(Combo_text, Combo_data)
    
            self.parent.displayExpStack(self.parent.Current_stack)
            
            self.parent.Info_box.ensureCursorVisible()
            self.parent.Info_box.insertPlainText("\n \u2022 Registered stack has been exported.")
            
            self.close()
        
#%% Functions : data visualization
    def displayExpStack(self, series):
        self.expSeries.ui.histogram.hide()
        self.expSeries.ui.roiBtn.hide()
        self.expSeries.ui.menuBtn.hide()

        view = self.expSeries.getView()
        state = view.getState()        
        self.expSeries.setImage(series) 
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
        
    def display_Aligne_treatment(self, Series): # Affichage de la stack alignée
        self.AlignedStack.ui.histogram.hide()
        self.AlignedStack.ui.roiBtn.hide()
        self.AlignedStack.ui.menuBtn.hide()
        
        view = self.AlignedStack.getView()
        state = view.getState()        
        self.AlignedStack.setImage(Series) 
        view.setState(state)
        
        view.setBackgroundColor(self.parent.color1)
        ROIplot = self.AlignedStack.getRoiPlot()
        ROIplot.setBackground(self.parent.color1)
        
        font=QtGui.QFont('Noto Sans Cond', 9)
        ROIplot.getAxis("bottom").setTextPen('k') # Apply size of the ticks label
        ROIplot.getAxis("bottom").setTickFont(font)
        
        self.AlignedStack.timeLine.setPen(color=self.parent.color3, width=15)
        self.AlignedStack.frameTicks.setPen(color=self.parent.color1, width=5)
        self.AlignedStack.frameTicks.setYRange((0, 1))

        s = self.AlignedStack.ui.splitter
        s.handle(1).setEnabled(True)
        s.setStyleSheet("background: 5px white;")
        s.setHandleWidth(5) 
       
    def display_Pre_treatment(self, Series):  # Affichage de la stack de pré-traitement
    
        self.pretreatment.setImage(Series) 
    
        self.pretreatment.clear()
        self.pretreatment.ui.menuBtn.hide()
        self.pretreatment.ui.roiBtn.hide()
        
        view = self.pretreatment.getView()
        state = view.getState()        
        self.pretreatment.setImage(Series) 
        view.setState(state)
        
        view.setBackgroundColor(self.parent.color1)
        ROIplot = self.pretreatment.getRoiPlot()
        ROIplot.setBackground(self.parent.color1)
        
        font=QtGui.QFont('Noto Sans Cond', 9)
        ROIplot.getAxis("bottom").setTextPen('k') # Apply size of the ticks label
        ROIplot.getAxis("bottom").setTickFont(font)
        
        self.pretreatment.timeLine.setPen(color=self.parent.color3, width=15)
        self.pretreatment.frameTicks.setPen(color=self.parent.color1, width=5)
        self.pretreatment.frameTicks.setYRange((0, 1))

        s = self.pretreatment.ui.splitter
        s.handle(1).setEnabled(True)
        s.setStyleSheet("background: 5px white;")
        s.setHandleWidth(5) 
        
        histplot = self.pretreatment.getHistogramWidget()
        histplot.setBackground(self.parent.color1)
        
        histplot.region.setBrush(pg.mkBrush(self.parent.color5 + (120,)))
        histplot.region.setHoverBrush(pg.mkBrush(self.parent.color5 + (60,)))
        histplot.region.pen = pg.mkPen(self.parent.color5)
        histplot.region.lines[0].setPen(pg.mkPen(self.parent.color5, width=2))
        histplot.region.lines[1].setPen(pg.mkPen(self.parent.color5, width=2))
        histplot.fillHistogram(color = self.parent.color5)        
        histplot.autoHistogramRange()
        
    def progression_bar(self): # Use to display and manage Progress bar
        self.prgbar = self.ValSlice
        self.progressBar.setValue(self.prgbar)
        
    def show_progress(self,current_iteration, end_iteration): # For Translation - Scaled rotation - Rigid body
        QApplication.processEvents()                    
        self.ValSlice = current_iteration
        self.progression_bar()
