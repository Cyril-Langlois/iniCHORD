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
from PyQt5.QtWidgets import QApplication, QLabel, QDialog, QVBoxLayout, QPushButton
from PyQt5 import QtCore, QtGui
from scipy import ndimage

from inichord import General_Functions as gf
from scipy.fft import fft2, ifft2, fftshift

path2thisFile = abspath(getsourcefile(lambda:0))
uiclass, baseclass = pg.Qt.loadUiType(os.path.dirname(path2thisFile) + "/Edit_Tools.ui")

class MainWindow(uiclass, baseclass):
    def __init__(self, parent):
        super().__init__()
        self.setupUi(self)
        self.parent = parent # Link sub-gui with main gui
        self.expStack = parent.Current_stack
        
        self.nSlices = self.expStack.shape[0]
        self.indice = self.expStack.shape[0]
        self.stackInfoValues_label.setText(str(self.expStack.shape)) # Shape of the image serie
        
        self.roi = pg.ROI((0, 0), (int(self.expStack.shape[1] * 1), int(self.expStack.shape[2] * 1)))
        self.roi.addScaleHandle((1, 1), (0, 0)) 
        self.roi.addScaleHandle((0, 1), (1, 0)) 
        self.roi.addScaleHandle((1, 0), (0, 1)) 
        self.roi.addScaleHandle((0, 0), (1, 1))
        self.roi.sigRegionChanged.connect(self.updateROIcoord)
        
        self.ROIinfoValues_label.setText(f"W: {int(self.roi.size()[0])} ; H: {int(self.roi.size()[1])} ; Origin: {int(self.roi.pos()[0]), int(self.roi.pos()[1])}")

        self.ROIwidth = int(self.roi.size()[0])
        self.ROIheight = int(self.roi.size()[1])
        
        self.ROIoriX = int(self.roi.pos()[0])
        self.ROIoriY = int(self.roi.pos()[1])
        
        self.expSeries.roiClicked()
        self.expSeries.addItem(self.roi)
        
        self.roi.setPen(color=self.parent.color5)
        
        self.initiateSliceKeeper()
        
        self.x = 0
        self.y = 0
           
        self.crosshair_v1= pg.InfiniteLine(angle=90, movable=False, pen=self.parent.color5)
        self.crosshair_h1 = pg.InfiniteLine(angle=0, movable=False, pen=self.parent.color5)
        
        self.proxy1 = pg.SignalProxy(self.expSeries.scene.sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.proxy4 = pg.SignalProxy(self.expSeries.ui.graphicsView.scene().sigMouseClicked, rateLimit=60, slot=self.mouseClick)
    
        self.start_spinBox.valueChanged.connect(self.modifySliceKeeper)
        self.step_spinBox.valueChanged.connect(self.modifySliceKeeper)
        self.validate_bttn.clicked.connect(self.validate_modified_Stack)
        self.Del_bttn.clicked.connect(self.delete_slice)
        self.Replace_bttn.clicked.connect(self.replace_slice)
        self.Erroneous_bttn.clicked.connect(self.image_correction)
        
        self.setWindowIcon(QtGui.QIcon('icons/crop_icon.png'))
        
        self.displayExpStack(self.expStack)
        
        app = QApplication.instance()
        screen = app.screenAt(self.pos())
        geometry = screen.availableGeometry()
        
        # Position (self.move) and size (self.resize) of the main GUI on the screen
        self.move(int(geometry.width() * 0.05), int(geometry.height() * 0.05))
        self.resize(int(geometry.width() * 0.7), int(geometry.height() * 0.6))
        self.screen = screen

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

    def initiateSliceKeeper(self):
        self.start_spinBox.setRange(0, self.nSlices)
        self.step_spinBox.setRange(1, self.nSlices)
        self.nbSlice_spinBox.setRange(1, self.nSlices)
        self.nbSlice_spinBox.setValue(self.nSlices)
    
    def modifySliceKeeper(self):
        sender = self.sender()
        
        if sender == self.start_spinBox or sender == self.step_spinBox:
            
            self.indice = 0
            n = 0
            
            while self.indice < (self.nSlices - self.step_spinBox.value() - self.start_spinBox.value()):
                self.indice += self.step_spinBox.value()
                n +=1
                self.nbSlice_spinBox.setRange(1, n + 1)
            self.nbSlice_spinBox.setValue(n + 1)
            
        if sender == self.start_spinBox:
            self.expSeries.setCurrentIndex(self.start_spinBox.value())                

    def updateROIcoord(self):
        self.ROIwidth = int(self.roi.size()[0])
        self.ROIheight = int(self.roi.size()[1])
        
        self.ROIoriX = int(self.roi.pos()[0])
        self.ROIoriY = int(self.roi.pos()[1])
        self.ROIinfoValues_label.setText(f"W: {int(self.roi.size()[0])} ; H: {int(self.roi.size()[1])} ; Origin: {int(self.roi.pos()[0]), int(self.roi.pos()[1])}")

    def delete_slice(self):
        indexes = [self.expSeries.currentIndex] # Index of the slice to delete (current view)
        self.expStack = np.delete(self.expStack, indexes, axis=0) # Remove the slice 
        
        self.stackInfoValues_label.setText(str(self.expStack.shape)) # Extract new stack shape
        self.nSlices = self.expStack.shape[0] # Extract the new number of slices
        self.nbSlice_spinBox.setRange(1, self.nSlices) # Modificiation of the dispatcher
        
        self.displayExpStack(self.expStack)
        
    def replace_slice(self): 
        indexes = self.expSeries.currentIndex # Index of the slice to delete (current view)

        if indexes > 0: # Remplacer le tableau 2D à l'indice 'i' par le tableau 2D de l'indice 'i-1'
            self.expStack[indexes] = self.expStack[indexes - 1]
        elif indexes == 0:
            [indexes] = self.expStack[indexes + 1]
            
        self.displayExpStack(self.expStack)
        
    def image_correction(self):
        self.Threshold_value = self.Threshold_2.value()

        distance = []
    
        # Comparer chaque image avec l'image de référence
        for j in range(0, self.expStack.shape[0]):
            if j == 0:
                image_reference = self.expStack[j]
            else:
                image_reference = self.expStack[j-1]
                
            decalage = self.Compute_shift(image_reference, self.expStack[j])
            distance_decalage = np.linalg.norm(decalage)
            
            distance.append(distance_decalage)
    
        index = np.hstack(distance)
        index = index < self.Threshold_value
        
        # Parcourir l'array pour trouver les paires de 'False'
        for k in range(len(index) - 1):
            # Vérifier si deux éléments consécutifs sont 'False'
            if index[k] == False and index[k + 1] == False:       
                self.expStack[k] = self.expStack[k - 1]
    
        self.displayExpStack(self.expStack)
        
    def Compute_shift(self,image1, image2):
        # Calculer la transformation de Fourier des deux images
        image1 = gf.convertToUint8(image1)
        image2 = gf.convertToUint8(image2)
        
        f_image1 = fft2(image1)
        f_image2 = fft2(image2)

        # Calculer la corrélation croisée en utilisant le produit conjugué
        produit_conjugue = f_image1 * np.conj(f_image2)
        cross_correlation = fftshift(ifft2(produit_conjugue))

        # Trouver le pic de la corrélation croisée pour déterminer le décalage
        decalage = np.unravel_index(np.argmax(np.abs(cross_correlation)), cross_correlation.shape)

        # Calculer le décalage relatif à partir du centre
        centre = np.array(image1.shape) // 2
        decalage = np.array(decalage) - centre

        return decalage

    def validate_modified_Stack(self):
        indStartX = self.ROIoriX
        indStartY = self.ROIoriY
        
        indEndX = indStartX + self.ROIwidth
        indEndY = indStartY + self.ROIheight
        
        toBin_Stack = self.expStack[:, indStartX:indEndX, indStartY:indEndY]
        
        start = self.start_spinBox.value()
        step = self.step_spinBox.value()
        indEnd = start + self.nbSlice_spinBox.value() * step
 
        toBin_Stack = toBin_Stack[start:indEnd:step,:,:]      
        
        reduction = 1 / self.XYbin_spinBox.value()
        splineOrder = self.spline_spinBox.value()
        
        self.binStack = ndimage.zoom(toBin_Stack, (1, reduction, reduction), order = splineOrder)
        
        if len(self.binStack) == 1:
            self.popup_message("Edit tools","Only one slice. Please change the number of images",'icons/crop_icon.png')
            
        else:
            self.parent.Current_stack = self.binStack
            self.parent.StackList.append(self.binStack)

            Combo_text = '\u2022 Edited stack'
    
            Combo_data = self.binStack
            self.parent.choiceBox.addItem(Combo_text, Combo_data)
            
            self.parent.Info_box.ensureCursorVisible()
            self.parent.Info_box.insertPlainText("\n \u2022 Edited stack : achieved.")   
            
            self.parent.displayExpStack(self.parent.Current_stack)
            
            self.close()
        
    def displayExpStack(self, series):
        self.expSeries.ui.histogram.hide()
        self.expSeries.ui.roiBtn.hide()
        self.expSeries.ui.menuBtn.hide()
        
        self.expSeries.addItem(self.crosshair_v1, ignoreBounds=True)
        self.expSeries.addItem(self.crosshair_h1, ignoreBounds=True) 
        
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
    
    def mouseMoved(self, e):
        pos = e[0]

        if self.expSeries.view.sceneBoundingRect().contains(pos):

            item = self.expSeries.view
            mousePoint = item.mapSceneToView(pos) 
                 
            self.crosshair_v1.setPos(mousePoint.x())
            self.crosshair_h1.setPos(mousePoint.y())

        try:
            self.x = int(mousePoint.x())
            self.y = int(mousePoint.y())
            
            if self.x >= 0 and self.y >= 0 and self.x < len(self.expStack[0, :, 0])and self.y < len(self.expStack[0, 0, :]):
                self.cursor_label.setText(str(self.x) + " ; " + str(self.y))
        except:
            pass

    def mouseClick(self, e):
        pos = e[0]
        
        fromPosX = pos.scenePos()[0]
        fromPosY = pos.scenePos()[1]
        
        posQpoint = QtCore.QPointF()
        posQpoint.setX(fromPosX)
        posQpoint.setY(fromPosY)

        if self.expSeries.view.sceneBoundingRect().contains(posQpoint):
                
            item = self.expSeries.view
            mousePoint = item.mapSceneToView(posQpoint) 

            self.crosshair_v1.setPos(mousePoint.x())
            self.crosshair_h1.setPos(mousePoint.y())
                 
            self.x = int(mousePoint.x())
            self.y = int(mousePoint.y())