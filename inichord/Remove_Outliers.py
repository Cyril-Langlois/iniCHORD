# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 23:26:20 2023

@author: clanglois1
"""
import os
from os.path import abspath

from inspect import getsourcefile
import numpy as np

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication

from inichord import General_Functions as gf

path2thisFile = abspath(getsourcefile(lambda:0))
uiclass, baseclass = pg.Qt.loadUiType(os.path.dirname(path2thisFile) + "/Remove_Outliers.ui")

class MainWindow(uiclass, baseclass):
    def __init__(self, parent):
        super().__init__()
        self.setupUi(self)
        self.parent = parent
        
        self.setWindowIcon(QtGui.QIcon('icons/remove_outliers_icon.png'))
                
        self.expStack = parent.Current_stack
        self.denoised_Stack = np.copy(parent.Current_stack)
        
        self.x = 0
        self.y = 0
        
        self.radius = 2
        self.threshold = 10
        
        self.label_radius.setText("Radius: " + str(self.radius))
        self.label_threshold.setText("Threshold: " + str(self.threshold))
        
        self.img_number = len(self.expStack)
        
        if self.denoised.isChecked():
            self.denoised.toggle()
            
        self.denoised.setCheckable(False)

        self.crosshair_v1= pg.InfiniteLine(angle=90, movable=False, pen=self.parent.color5)
        self.crosshair_h1 = pg.InfiniteLine(angle=0, movable=False, pen=self.parent.color5)
        
        self.plotIt = self.profiles.getPlotItem()
        self.plotIt.addLine(x = self.expSeries.currentIndex)
        
        self.proxy1 = pg.SignalProxy(self.expSeries.scene.sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.proxy4 = pg.SignalProxy(self.expSeries.ui.graphicsView.scene().sigMouseClicked, rateLimit=60, slot=self.mouseClick)
        
        self.slider_threshold.valueChanged.connect(self.threshold_changed)
        self.slider_radius.valueChanged.connect(self.radius_changed)
        self.expSeries.timeLine.sigPositionChanged.connect(self.drawCHORDprofiles)
        self.preview.clicked.connect(self.remOutStack)
        self.denoised.stateChanged.connect(self.drawCHORDprofiles)
        
        self.Validate_button.clicked.connect(self.validate)
        self.mouseLock.setVisible(False)
        
        self.prgbar = 0 # Outil pour la bar de progression
        self.progressBar.setValue(self.prgbar)
        self.progressBar.setRange(0, self.img_number-1)
        
        self.remOutSlice()
        self.displayExpStack(self.denoised_Stack)
        
        app = QApplication.instance()
        screen = app.screenAt(self.pos())
        geometry = screen.availableGeometry()
        
        self.move(int(geometry.width() * 0.1), int(geometry.height() * 0.15))
        self.resize(int(geometry.width() * 0.7), int(geometry.height() * 0.6))
        self.screen = screen
        
        self.defaultdrawCHORDprofiles()
        self.Validate_button.setEnabled(False)

#%% Functions 
    def radius_changed(self):
        self.radius = self.slider_radius.value()
        self.label_radius.setText("Radius: " + str(self.radius))
        self.remOutSlice()
    
    def threshold_changed(self):
        self.threshold = self.slider_threshold.value()
        self.label_threshold.setText("Threshold: " + str(self.threshold))
        self.remOutSlice()
    
    def remOutSlice(self):
        dummy, a = gf.remove_outliers(self.expStack[0, :, :], self.radius, self.threshold)
        
        self.denoised_Stack[0, :, :] = a 
        self.displayExpStack(self.denoised_Stack)
        self.denoised.setEnabled(False)
        
    def remOutStack(self):
        self.denoised.setEnabled(False)
        self.preview.setEnabled(False)
        
        for i in range(0,len(self.expStack[:, 0, 0])): # Apply parameters on each slice
            _, self.denoised_Stack[i, :, :] =  gf.remove_outliers(self.expStack[i,:,:], self.radius, self.threshold)

            QApplication.processEvents()    
            self.ValSlice = i
            self.progression_bar()
       
        self.drawCHORDprofiles()                

        self.denoised.setEnabled(True)         
        self.denoised.setCheckable(True)
        self.denoised.setChecked(True)
        self.preview.setEnabled(True)
        self.Validate_button.setEnabled(True)
        
        self.displayExpStack(self.denoised_Stack)
        
    def validate(self):
        self.parent.Current_stack = np.copy(self.denoised_Stack)
        self.parent.StackList.append(self.denoised_Stack)
        
        Combo_text = '\u2022 Outliers filtering. Radius : ' + str(self.radius) + ' ; Threshold : ' + str(self.threshold)
        Combo_data = self.denoised_Stack
        self.parent.choiceBox.addItem(Combo_text, Combo_data)
        
        self.parent.displayExpStack(self.parent.Current_stack)
        
        self.parent.Info_box.ensureCursorVisible()
        self.parent.Info_box.insertPlainText("\n \u2022 Remove outliers is achieved.") 
        
        self.close()

    def drawCHORDprofiles(self):
        try:
            self.profiles.clear()
            line = self.plotIt.addLine(x = self.expSeries.currentIndex)
            line.setPen({'color': (42, 42, 42, 100), 'width': 2})
            
            self.legend = self.profiles.addLegend(horSpacing = 30, labelTextSize = '10pt', colCount = 1, labelTextColor = 'black', brush = self.parent.color6, pen = pg.mkPen(color=(0, 0, 0), width=1))
            
            pen = pg.mkPen(color=self.parent.color4, width=5) # Color and line width of the profile
            self.profiles.plot(self.expStack[:, self.x, self.y], pen=pen, name='Before remove') # Plot of the profile
            
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

                self.profiles.plot(self.denoised_Stack[:, self.x, self.y], pen=pen2, name='After remove')
        except:
            pass
        
    def defaultdrawCHORDprofiles(self):
        self.profiles.clear()
        self.profiles.setBackground(self.parent.color2)
        
        self.profiles.getPlotItem().hideAxis('bottom')
        self.profiles.getPlotItem().hideAxis('left')

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