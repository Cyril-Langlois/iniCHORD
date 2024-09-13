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

from . import general_functions as gf
from skimage.metrics import mean_squared_error as mse
from skimage.restoration import denoise_tv_chambolle
from skimage.metrics import structural_similarity as ssim

path2thisFile = abspath(getsourcefile(lambda:0))
uiclass, baseclass = pg.Qt.loadUiType(os.path.dirname(path2thisFile) + "/Denoise_v1.ui")

class MainWindow(uiclass, baseclass):
    def __init__(self, parent):
        super().__init__()
        self.setupUi(self)
        self.parent = parent
        
        self.setWindowIcon(QtGui.QIcon('icons/filter_icon.png'))  
        
        self.expStack = parent.Current_stack
        
        self.Denoise_stack = np.copy(self.expStack)
        self.flag = parent.flag

        if self.flag == True:
            self.Reference = parent.Reference
            self.Slice1 = parent.Slice1
            self.displayRef(self.Reference)
            
            self.Proxy_button.setEnabled(False)
            
        else :
            self.Denoise_button.setEnabled(False)
            self.Denoise_button2.setEnabled(False)
            self.ChoiceBox.setEnabled(False)
            self.Choice_Idx.setEnabled(False)
            self.Spin_first.setEnabled(False)
            self.Spin_final.setEnabled(False)
            self.Spin_nbr.setEnabled(False)
            self.Validate_button.setEnabled(False)
            
        self.Proxy_button.clicked.connect(self.ImportProxyReference)
        
        self.Denoise_button.clicked.connect(self.AutoDenoisingStep)
        self.Denoise_button2.clicked.connect(self.StackDenoising)
        self.Validate_button.clicked.connect(self.ImportCurrentStack)
        
        self.x = 0
        self.y = 0
    
        self.crosshair_v1= pg.InfiniteLine(angle=90, movable=False, pen=self.parent.color5)
        self.crosshair_h1 = pg.InfiniteLine(angle=0, movable=False, pen=self.parent.color5)
        
        self.plotIt = self.profiles0.getPlotItem()
        self.plotIt.addLine(x = self.expSeries.currentIndex)
        
        self.proxy1 = pg.SignalProxy(self.expSeries.scene.sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.proxy4 = pg.SignalProxy(self.expSeries.ui.graphicsView.scene().sigMouseClicked, rateLimit=60, slot=self.mouseClick)
        
        self.expSeries.timeLine.sigPositionChanged.connect(self.drawCHORDprofiles)
        
        self.flagden = 0
        
        self.defaultdrawCHORDprofiles()
        self.displayExpStack(self.expStack)
        self.defaultIV()
        
        app = QApplication.instance()
        screen = app.screenAt(self.pos())
        geometry = screen.availableGeometry()
        
        self.move(int(geometry.width() * 0.1), int(geometry.height() * 0.15))
        self.resize(int(geometry.width() * 0.7), int(geometry.height() * 0.7))
        self.screen = screen
        
#%% Functions               
    def Init_ProgressBar(self): # Define Progress bar depending on the step
        if self.flagsearch == 1:
            self.img_number = len(self.sigma_range)
            self.prgbar = 0 # Outil pour la bar de progression
            self.progressBar.setValue(self.prgbar)
            self.progressBar.setRange(0, self.img_number-1)
        elif self.flagsearch == 0:
            self.img_number = len(self.expStack)
            self.prgbar = 0 # Outil pour la bar de progression
            self.progressBar.setValue(self.prgbar)
            self.progressBar.setRange(0, self.img_number-1)
            
    def progression_bar(self): # Progress bar function
        self.prgbar = self.ValSlice
        self.progressBar.setValue(self.prgbar)

    def ImportProxyReference(self): # Step to create a proxy of image reference
        value = self.AVG_box.value()
        self.AVG_value = value
        
        self.AVG_slices()
        self.displayExpStack(self.expStack)
        self.label_13.setText("Image reference")
        self.displayRef(self.Reference)
        
        self.Spin_first.setEnabled(True)
        self.Spin_final.setEnabled(True)
        self.Spin_nbr.setEnabled(True)
        
        self.Denoise_button.setEnabled(True)
        self.ChoiceBox.setEnabled(True)
        
    def AVG_slices(self): # Creation of a proxy of reference using a given number of images averaged together
        self.Slice1 = self.expStack[0,:,:]
        self.Reference = np.nanmean(self.expStack[0:self.AVG_value,:,:],axis=0)

    def AutoDenoisingStep(self): # Step to determine the best denoising parameter
        self.flagsearch = 1
        
        Spin_first_value = self.Spin_first.value()
        Spin_final_value = self.Spin_final.value()
        Spin_nbr_value = self.Spin_nbr.value()
        
        self.sigma_range = np.round(np.linspace(Spin_first_value, Spin_final_value, Spin_nbr_value, endpoint = True),4)
        
        self.Step_choice = self.ChoiceBox.currentText()
        
        if self.Step_choice == 'NLMD':            
            self.NLMD_approach()
            self.Compute_MSE_SSIM()
            self.drawParamProfiles()
            
            self.denSSIM = gf.NonLocalMeanDenoising(self.Slice1, self.idx_SSIM)
            self.denMSE = gf.NonLocalMeanDenoising(self.Slice1, self.idx_MSE)
            
        elif self.Step_choice == 'BM3D':        
            self.BM3D_approach()
            self.Compute_MSE_SSIM()
            self.drawParamProfiles()
            
            self.denSSIM = gf.BM3D(self.Slice1, self.idx_SSIM, isFast = True)
            self.denMSE = gf.BM3D(self.Slice1, self.idx_MSE, isFast = True)
            
        elif self.Step_choice == 'VSNR':            
            self.VSNR_approach()
            self.Compute_MSE_SSIM()            
            self.drawParamProfiles()
            
            filter_SSIM = [{'name':'Dirac', 'noise_level': self.idx_SSIM }] 
            filter_MSE = [{'name':'Dirac', 'noise_level': self.idx_MSE }] 
            
            self.denSSIM = gf.VSNR_funct(self.Slice1, filter_SSIM)
            self.denMSE = gf.VSNR_funct(self.Slice1, filter_MSE)
            
        elif self.Step_choice == 'TV Chambolle':
            self.TV_chamb_approach()
            self.Compute_MSE_SSIM()              
            self.drawParamProfiles()
            
            self.denSSIM = denoise_tv_chambolle(self.Slice1, self.idx_SSIM)
            self.denMSE = denoise_tv_chambolle(self.Slice1, self.idx_MSE)
            
        self.idx_SSIM_info = '%.3f'%(self.idx_SSIM)
        self.idx_MSE_info = '%.3f'%(self.idx_MSE)
        
        self.SSIM_info.setText('Parameter from SSIM: ' + self.idx_SSIM_info)
        self.MSE_info.setText('Parameter from MSE: ' + self.idx_MSE_info)
        
        self.denMerge = np.concatenate((self.denSSIM,self.denMSE),0)
        self.label_13.setText("SSIM and MSE denoising")
        self.displayRef(self.denMerge)
        
        self.Denoise_button2.setEnabled(True)
        self.Choice_Idx.setEnabled(True)

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

    def NLMD_approach(self): # Compute NLMD denoising for determination of best parameter
        self.Init_ProgressBar()
        self.Recap_denoise_stack = []
        
        self.Reference = self.check_type(self.Reference) # Convert data to float32 if needed
        self.Slice1 = self.check_type(self.Slice1) # Convert data to float32 if needed

        self.Reference2 = gf.NonLocalMeanDenoising(self.Reference, param_h = 0, fastAlgo = True, size = 5, distance = 6)
        
        for j in range(0,len(self.sigma_range)):
            denoised = gf.NonLocalMeanDenoising(self.Slice1, param_h = self.sigma_range[j], fastAlgo = True, size = 5, distance = 6)    
            self.Recap_denoise_stack.append(denoised)
            
            QApplication.processEvents() 
            self.ValSlice = j
            self.progression_bar()
            
    def BM3D_approach(self): # Compute BM3D denoising for determination of best parameter 
        self.Init_ProgressBar()

        self.Recap_denoise_stack = []
        self.Reference2 = gf.BM3D(self.Reference, 0, isFast = True)
        
        for j in range(0,len(self.sigma_range)):
            denoised = gf.BM3D(self.Slice1, self.sigma_range[j], isFast = True)   
            self.Recap_denoise_stack.append(denoised)
            
            QApplication.processEvents() 
            self.ValSlice = j
            self.progression_bar()

    def VSNR_approach(self): # Compute VSNR denoising for determination of best parameter 
        self.Init_ProgressBar()
        self.Recap_denoise_stack = []
        filter_ = [{'name':'Dirac', 'noise_level':0}] 
        self.Reference2 = gf.VSNR_funct(self.Reference,filter_)
        
        for j in range(0,len(self.sigma_range)):
            
            filter_ = [{'name':'Dirac', 'noise_level': self.sigma_range[j]}] 
            denoised = gf.VSNR_funct(self.Slice1,filter_)
                       
            self.Recap_denoise_stack.append(denoised)
            
            QApplication.processEvents() 
            self.ValSlice = j
            self.progression_bar()

    def TV_chamb_approach(self): # Compute TV Chambolle denoising for determination of best parameter 
        self.Init_ProgressBar()
        self.Recap_denoise_stack = []
        self.Reference2 = denoise_tv_chambolle(self.Reference, 1e-9)
    
        for j in range(0,len(self.sigma_range)):
            denoised = denoise_tv_chambolle(self.Slice1, self.sigma_range[j])
            self.Recap_denoise_stack.append(denoised)
            
            QApplication.processEvents() 
            self.ValSlice = j
            self.progression_bar()

    def Compute_MSE_SSIM(self): # Computation of MSE and SSIM value between images
        #  MSE Computation
        self.MSE = [mse(self.Reference2, img) for img in self.Recap_denoise_stack]
        self.min_MSE = np.nanmin(self.MSE)
        self.idx_MSE = self.sigma_range[np.nanargmin(self.MSE)]
        
        #  SSIM Computation
        self.SSIM = [ssim(self.Reference2, img, data_range=self.Reference2.max() - self.Reference2.min()) for img in self.Recap_denoise_stack]
        self.max_SSIM = np.nanmax(self.SSIM)
        self.idx_SSIM = self.sigma_range[np.nanargmax(self.SSIM)]

    def StackDenoising(self):
        self.flagsearch = 0
        
        self.Idx = self.Choice_Idx.currentText()
        
        if self.Idx == 'SSIM':
            self.IdxUsed = self.idx_SSIM
        elif self.Idx == 'MSE':
            self.IdxUsed = self.idx_MSE
        
        self.Step_choice = self.ChoiceBox.currentText()
        
        if self.Step_choice == 'NLMD':                
            self.Denoise_stack = np.copy(self.expStack)
            
            self.NLMD_denStack()
            self.displayExpStack(self.Denoise_stack)
            
        elif self.Step_choice == 'BM3D':                
            self.Denoise_stack = np.copy(self.expStack)
            
            self.BM3D_denStack()
            self.displayExpStack(self.Denoise_stack)

        elif self.Step_choice == 'VSNR':                
            self.Denoise_stack = np.copy(self.expStack)
            
            self.VNSR_denStack()
            self.displayExpStack(self.Denoise_stack)   

        elif self.Step_choice == 'TV Chambolle':              
            self.Denoise_stack = np.copy(self.expStack)
            
            self.TVcham_denStack()
            self.displayExpStack(self.Denoise_stack)  
            
        self.flagden = 1    
        self.drawCHORDprofiles()
        self.Validate_button.setEnabled(True)

    def NLMD_denStack(self): # Denoising using NLMD ; parameter is the denoising parameter value
        self.Init_ProgressBar()
        
        self.expStack = self.check_type(self.expStack) # Convert data to float32 if needed
              
        for i in range(0,len(self.expStack)):          
            self.Denoise_stack[i,:,:] = gf.NonLocalMeanDenoising(self.expStack[i,:,:], self.IdxUsed)

            QApplication.processEvents() 
            self.ValSlice = i
            self.progression_bar()
    
    def BM3D_denStack(self): # Denoising using BM3D ; parameter is the denoising parameter value
        self.Init_ProgressBar()
        for i in range(0,len(self.expStack)):
            self.Denoise_stack[i,:,:] = gf.BM3D(self.expStack[i,:,:], self.IdxUsed, isFast = True)
    
            QApplication.processEvents() 
            self.ValSlice = i
            self.progression_bar()
    
    def VNSR_denStack(self): # Denoising using VSNR (Dirac) ; parameter is the denoising parameter value
        self.Init_ProgressBar()
        filter_ = [{'name':'Dirac', 'noise_level': self.IdxUsed }] 
        
        for i in range(0,len(self.expStack)):
            self.Denoise_stack[i,:,:] = gf.VSNR_funct(self.expStack[i,:,:], filter_)

            QApplication.processEvents() 
            self.ValSlice = i
            self.progression_bar()
    
    def TVcham_denStack(self): # Denoising using TV Chambolle ; parameter is the denoising parameter value
        self.Init_ProgressBar()
        for i in range(0,len(self.expStack)):
            self.Denoise_stack[i,:,:] = denoise_tv_chambolle(self.expStack[i,:,:], self.IdxUsed)

            QApplication.processEvents() 
            self.ValSlice = i
            self.progression_bar()

    def ImportCurrentStack(self): # Extract denoised stack to the main gui
        if self.Idx == 'MSE':
            self.FlagIdx = self.idx_MSE
        elif self.Idx == 'SSIM':
            self.FlagIdx = self.idx_SSIM
        
        self.parent.Current_stack = self.Denoise_stack
        self.parent.StackList.append(self.Denoise_stack)
        
        Combo_text = '\u2022 Auto denoising using : ' + str(self.Step_choice) + '. Indice : ' + str(self.Idx) + '. Parameter : ' + str(self.FlagIdx)
        Combo_data = self.Denoise_stack
        self.parent.choiceBox.addItem(Combo_text, Combo_data)
        
        self.parent.displayExpStack(self.parent.Current_stack)
        
        self.parent.Info_box.ensureCursorVisible()
        self.parent.Info_box.insertPlainText("\n \u2022 Auto denoising is achieved")
        
        self.close()

    def defaultdrawCHORDprofiles(self):
        self.profiles.clear()
        self.profiles.setBackground(self.parent.color2)
        
        self.profiles.getPlotItem().hideAxis('bottom')
        self.profiles.getPlotItem().hideAxis('left')
        
        self.profiles0.clear()
        self.profiles0.setBackground(self.parent.color2)
        
        self.profiles0.getPlotItem().hideAxis('bottom')
        self.profiles0.getPlotItem().hideAxis('left')
        
        self.profiles2.clear()
        self.profiles2.setBackground(self.parent.color2)
        
        self.profiles2.getPlotItem().hideAxis('bottom')
        self.profiles2.getPlotItem().hideAxis('left')

    def drawParamProfiles(self):
        try:
            self.profiles.clear()
            self.profiles2.clear()
            
            pen = pg.mkPen(color=self.parent.color4, width=5) # Color and line width of the profile         

            self.profiles.plot(self.SSIM, pen=pen) # Plot of the SSIM Index
            self.profiles2.plot(self.MSE, pen=pen) # Plot of the MSE Index

            styles = {"color": "black", "font-size": "40px", "font-family": "Noto Sans Cond"} # Style for labels
            self.profiles.setLabel("left", "SSIM value", **styles) # Import style for Y label
            self.profiles.setLabel("bottom", "Discretization index", **styles) # Import style for X label
            
            self.profiles.getAxis('left').setTextPen('k') # Set the axis in black
            self.profiles.getAxis('bottom').setTextPen('k') # Set the axis in black
            self.profiles.setBackground(self.parent.color2)
            self.profiles.showGrid(x=True, y=True)

            self.profiles2.setLabel("left", "MSE value", **styles) # Import style for Y label
            self.profiles2.setLabel("bottom", "Discretization index", **styles) # Import style for X label
           
            self.profiles2.getAxis('left').setTextPen('k') # Set the axis in black
            self.profiles2.getAxis('bottom').setTextPen('k') # Set the axis in black 
            self.profiles2.setBackground(self.parent.color2)
            self.profiles2.showGrid(x=True, y=True)
            
        except:
            pass
        
    def drawCHORDprofiles(self):
        try:
            self.profiles0.clear()
            line = self.plotIt.addLine(x = self.expSeries.currentIndex)
            line.setPen({'color': (42, 42, 42, 100), 'width': 2})
            
            self.legend = self.profiles0.addLegend(horSpacing = 30, labelTextSize = '10pt', colCount = 1, labelTextColor = 'black', brush = self.parent.color6, pen = pg.mkPen(color=(0, 0, 0), width=1))
            
            pen = pg.mkPen(color=self.parent.color4, width=5) # Color and line width of the profile
            self.profiles0.plot(self.expStack[:, self.x, self.y], pen=pen, name='Undenoised') # Plot of the profile
            
            styles = {"color": "black", "font-size": "40px", "font-family": "Noto Sans Cond"} # Style for labels
            self.profiles0.setLabel("left", "Grayscale value", **styles) # Import style for Y label
            self.profiles0.setLabel("bottom", "Slice", **styles) # Import style for X label
            
            font=QtGui.QFont('Noto Sans Cond', 9)
            
            self.profiles0.getAxis("left").setTickFont(font) # Apply size of the ticks label
            self.profiles0.getAxis("left").setStyle(tickTextOffset = 20) # Apply a slight offset
            self.profiles0.getAxis("bottom").setTickFont(font) # Apply size of the ticks label
            self.profiles0.getAxis("bottom").setStyle(tickTextOffset = 20) # Apply a slight offset
            
            self.profiles0.getAxis('left').setTextPen('k') # Set the axis in black
            self.profiles0.getAxis('bottom').setTextPen('k') # Set the axis in black
            
            self.profiles0.setBackground(self.parent.color2)
            self.profiles0.showGrid(x=True, y=True)
            
            if self.flagden == 1:
                pen2 = pg.mkPen(color=self.parent.color5, width=5) # Color and line width of the profile
                self.profiles0.plot(self.Denoise_stack[:, self.x, self.y], pen=pen2, name='Denoised')
            
        except:
            pass

    def defaultIV(self):
        self.refSeries.ui.histogram.hide()
        self.refSeries.ui.roiBtn.hide()
        self.refSeries.ui.menuBtn.hide()
        
        view = self.refSeries.getView()
        view.setBackgroundColor(self.parent.color1)

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
        
    def displayRef(self, Series):
        self.refSeries.ui.histogram.hide()
        self.refSeries.ui.roiBtn.hide()
        self.refSeries.ui.menuBtn.hide()
        
        view = self.refSeries.getView()
        state = view.getState()        
        self.refSeries.setImage(Series) 
        view.setState(state)
        
        view.setBackgroundColor(self.parent.color1)
        ROIplot = self.refSeries.getRoiPlot()
        ROIplot.setBackground(self.parent.color1)
        
        self.refSeries.timeLine.setPen(color=self.parent.color3, width=15)
        self.refSeries.frameTicks.setPen(color=self.parent.color1, width=5)
        self.refSeries.frameTicks.setYRange((0, 1))

        s = self.refSeries.ui.splitter
        s.handle(1).setEnabled(True)
        s.setStyleSheet("background: 5px white;")
        s.setHandleWidth(5) 
        
        self.refSeries.autoRange()

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