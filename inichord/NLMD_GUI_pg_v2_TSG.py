# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:11:27 2023

@author: clanglois1
"""

import os
from os.path import abspath

from inspect import getsourcefile

import numpy as np

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from PyQt5 import QtWidgets

from PyQt5.QtWidgets import QApplication

from inichord import general_functions as gf

if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    # enable highdpi scaling

if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    # use highdpi icons

path2thisFile = abspath(getsourcefile(lambda:0))
uiclass, baseclass = pg.Qt.loadUiType(os.path.dirname(path2thisFile) + "/NLMD_v2_TSG.ui")

class MainWindow(uiclass, baseclass):
    def __init__(self, parent):
        super().__init__()
        self.setupUi(self)
        self.parent = parent
        
        self.setWindowIcon(QtGui.QIcon('icons/filter_icon.png'))    
        
        self.expStack = parent.Current_stack

        self.denoised_Stack = np.copy(parent.Current_stack)

        self.x = 0
        self.y = 0
        
        self.patch_size = 5
        self.patch_distance = 6
        self.param_h = self.slider_h.value() / 10_0.0
        
        self.label_distance.setText("Patch Distance: " + str(self.patch_distance))
        self.label_size.setText("Patch Size: " + str(self.patch_size))
        self.label_h.setText("Parameter h: " + str(self.param_h))
        
        if self.denoised.isChecked():
            self.denoised.toggle()
            
        self.denoised.setCheckable(False)

        self.crosshair_v1= pg.InfiniteLine(angle=90, movable=False, pen=self.parent.color5)
        self.crosshair_h1 = pg.InfiniteLine(angle=0, movable=False, pen=self.parent.color5)
        
        self.plotIt = self.profiles.getPlotItem()
        self.plotIt.addLine(x = self.expSeries.currentIndex)
        
        self.proxy1 = pg.SignalProxy(self.expSeries.scene.sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.proxy4 = pg.SignalProxy(self.expSeries.ui.graphicsView.scene().sigMouseClicked, rateLimit=60, slot=self.mouseClick)
        
        self.slider_size.valueChanged.connect(self.size_changed)
        self.slider_distance.valueChanged.connect(self.distance_changed)
        self.slider_h.valueChanged.connect(self.h_changed)
        
        self.expSeries.timeLine.sigPositionChanged.connect(self.drawCHORDprofiles)
        self.preview.clicked.connect(self.denoiseStack)
        self.denoised.stateChanged.connect(self.drawCHORDprofiles)
        
        self.Validate_button.clicked.connect(self.validate)
        
        self.img_number = len(self.expStack)
        self.prgbar = 0 # Outil pour la bar de progression
        self.progressBar.setValue(self.prgbar)
        self.progressBar.setRange(0, self.img_number-1)
        
        self.type = self.expStack.dtype

        self.expStack = self.check_type(self.expStack) # Convert data to float32 if needed
        self.denoised_Stack = self.check_type(self.denoised_Stack) # Convert data to float32 if needed
            
        self.displayExpStack(self.expStack)
        self.defaultdrawCHORDprofiles()
        
        app = QApplication.instance()
        screen = app.screenAt(self.pos())
        geometry = screen.availableGeometry()
        
        self.move(int(geometry.width() * 0.1), int(geometry.height() * 0.15))
        self.resize(int(geometry.width() * 0.7), int(geometry.height() * 0.6))
        self.screen = screen
        
        self.Validate_button.setEnabled(False)
        self.mouseLock.setVisible(False)

    def check_type(self,data): # Check if the data has type uint8 or uint16 and modify it to float32
        datatype = data.dtype

        if datatype == "float64":
            data = gf.convertToUint8(data)
            self.data = data.astype(np.float32)
        if datatype == "uint16":
            data = gf.convertToUint8(data)
            self.data = data.astype(np.float32)
        elif datatype == "uint8":
            self.data = data.astype(np.float32)
        elif datatype == "float32":
            self.data = data
            
        return self.data

    def size_changed(self):
        value = self.slider_size.value()
        self.patch_size = value
        self.label_size.setText("Patch Size: " + str(value))
        self.denoiseSlice()
    
    def distance_changed(self):
        value = self.slider_distance.value()
        self.threshold = value
        self.label_distance.setText("Patch Distance: " + str(value))
        self.denoiseSlice()
        
    def h_changed(self):
        value = self.slider_h.value()
        self.param_h = value / 10_0.0        
        self.label_h.setText("Parameter h: " + str(self.param_h))
        self.denoiseSlice()

    def denoiseSlice(self):
        a = gf.NonLocalMeanDenoising(self.expStack[0, :, :], self.param_h, True, self.patch_size, self.patch_distance)
        
        self.denoised_Stack[0, :, :] = a

        self.displayExpStack(self.denoised_Stack)
        
        self.denoised.setEnabled(False)

    def denoiseStack(self):
        self.denoised.setEnabled(False)
        self.preview.setEnabled(False)
        
        for i in range(0,len(self.expStack[:, 0, 0])): # Apply parameter on each slice
            a = gf.NonLocalMeanDenoising(self.expStack[i, :, :], self.param_h, True, self.patch_size, self.patch_distance)
            self.denoised_Stack[i, :, :] = a

            QApplication.processEvents()    
            self.ValSlice = i
            self.progression_bar()
                
        self.denoised.setCheckable(True)
        self.denoised.setEnabled(True)  
        self.denoised.setChecked(True)
        self.preview.setEnabled(True)
        
        self.displayExpStack(self.denoised_Stack)
        self.Validate_button.setEnabled(True)
        
    def validate(self):
        self.parent.Current_stack = np.copy(self.denoised_Stack)
        self.parent.StackList.append(self.denoised_Stack)
        
        Combo_text = '\u2022 Manual NLMD denoising. Parameter h : ' + str(self.param_h) + '. Patch size : ' + str(self.patch_size) + '. Patch distance : ' + str(self.patch_distance)  
        Combo_data = self.denoised_Stack
        self.parent.choiceBox.addItem(Combo_text, Combo_data)

        self.parent.displayExpStack(self.parent.Current_stack)
        
        self.drawCHORDprofiles()  
        
        self.parent.Info_box.ensureCursorVisible()
        self.parent.Info_box.insertPlainText("\n \u2022 NLMD denoising is achieved.")     
        
        self.close()

    def defaultdrawCHORDprofiles(self):
        self.profiles.clear()
        self.profiles.setBackground(self.parent.color2)
        
        self.profiles.getPlotItem().hideAxis('bottom')
        self.profiles.getPlotItem().hideAxis('left')
     
    def drawCHORDprofiles(self):
        try:
            self.profiles.clear()
            line = self.plotIt.addLine(x = self.expSeries.currentIndex)
            line.setPen({'color': (42, 42, 42, 100), 'width': 2})
            
            self.legend = self.profiles.addLegend(horSpacing = 30, labelTextSize = '10pt', colCount = 1, labelTextColor = 'black', brush = self.parent.color6, pen = pg.mkPen(color=(0, 0, 0), width=1))
            
            pen = pg.mkPen(color=self.parent.color4, width=5) # Color and line width of the profile
            self.profiles.plot(self.expStack[:, self.x, self.y], pen=pen, name='Undenoised') # Plot of the profile
            
            styles = {"color": "black", "font-size": "40px", "font-family": "Noto Sans Cond"} # Style for labels
            self.profiles.setLabel("left", "Grayscale value", **styles) # Import style for Y label
            self.profiles.setLabel("bottom", "Slice", **styles) # Import style for X label
            
            font=QtGui.QFont('Noto Sans Cond', 9)
            
            self.profiles.getAxis("left").setTickFont(font) # Apply size of the ticks label
            self.profiles.getAxis("left").setStyle(tickTextOffset = 20) # Apply a slight offset
            self.profiles.getAxis("bottom").setTickFont(font) # Apply size of the ticks label
            self.profiles.getAxis("bottom").setStyle(tickTextOffset = 20) # Apply a slight offset
            
            self.profiles.getAxis('left').setTextPen('k') # Set the axis in black
            self.profiles.getAxis('bottom').setTextPen('k') # Set the axis in black
            
            self.profiles.setBackground(self.parent.color2)
            self.profiles.showGrid(x=True, y=True)
            
            if self.denoised.isChecked():
                pen2 = pg.mkPen(color=self.parent.color5, width=5) # Color and line width of the profile

                self.profiles.plot(self.denoised_Stack[:, self.x, self.y], pen=pen2, name='Denoised')
        except:
            pass

    def progression_bar(self): # Fonction relative à la barre de progression
        self.prgbar = self.ValSlice
        self.progressBar.setValue(self.prgbar)

    def displayExpStack(self, Series):
        self.expSeries.ui.histogram.hide()
        self.expSeries.ui.roiBtn.hide()
        self.expSeries.ui.menuBtn.hide()
        
        self.expSeries.addItem(self.crosshair_v1, ignoreBounds=True)
        self.expSeries.addItem(self.crosshair_h1, ignoreBounds=True) 
        
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
        
    def mouseMoved(self, e):
        pos = e[0]

        if not self.mouseLock.isChecked():
            if self.expSeries.view.sceneBoundingRect().contains(pos):
    
                item = self.expSeries.view
                mousePoint = item.mapSceneToView(pos) 
                     
                self.crosshair_v1.setPos(mousePoint.x())
                self.crosshair_h1.setPos(mousePoint.y())

            self.x = int(mousePoint.x())
            self.y = int(mousePoint.y())
            
            if self.x >= 0 and self.y >= 0 and self.x < len(self.expStack[0, :, 0]) and self.y < len(self.expStack[0, 0, :]):
                self.drawCHORDprofiles()

    def mouseClick(self, e):
        pos = e[0]
        
        self.mouseLock.toggle()
        
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
            
        if self.x >= 0 and self.y >= 0 and self.x < len(self.expStack[0, :, 0])and self.y < len(self.expStack[0, 0, :]):
            self.drawCHORDprofiles()