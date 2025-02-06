# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:31:45 2023

@author: clanglois1
"""

import os
from os.path import abspath

from inspect import getsourcefile
import numpy as np

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication

import General_Functions as gf
import tifffile as tf
import time
from sklearn.cluster import MiniBatchKMeans
from skimage.measure import regionprops

path2thisFile = abspath(getsourcefile(lambda:0))
uiclass, baseclass = pg.Qt.loadUiType(os.path.dirname(path2thisFile) + "/Kmean.ui")

class MainWindow(uiclass, baseclass):
    def __init__(self, parent):
        super().__init__()
        self.setupUi(self)
        self.parent = parent
        
        self.setWindowIcon(QtGui.QIcon('icons/Grain_Icons.png'))   
        
        self.preview.setEnabled(False)
        
        self.defaultIV() # Default ImageView (when no series)
        
        self.x = 0
        self.y = 0

        self.crosshair_v1= pg.InfiniteLine(angle=90, movable=False, pen=self.parent.color5)
        self.crosshair_h1 = pg.InfiniteLine(angle=0, movable=False, pen=self.parent.color5)
        
        self.crosshair_v2= pg.InfiniteLine(angle=90, movable=False, pen=self.parent.color5)
        self.crosshair_h2 = pg.InfiniteLine(angle=0, movable=False, pen=self.parent.color5)
        
        self.proxy1 = pg.SignalProxy(self.expSeries.scene.sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.proxy4 = pg.SignalProxy(self.expSeries.ui.graphicsView.scene().sigMouseClicked, rateLimit=60, slot=self.mouseClick)
           
        self.proxy2 = pg.SignalProxy(self.Clustered_map.scene.sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.proxy5 = pg.SignalProxy(self.Clustered_map.ui.graphicsView.scene().sigMouseClicked, rateLimit=60, slot=self.mouseClick)
        
        try :
            self.expStack = parent.Current_stack
            self.StackDir = self.parent.StackDir
            
            self.preview.setEnabled(True)
            
            self.displayExpStack(self.expStack)
        except:
            pass
        
        self.Open_bttn.clicked.connect(self.loaddata) # Load image series or 2D array (for KAD map)
        self.preview.clicked.connect(self.Compute_Kmean)
        self.Validate_button.clicked.connect(self.validate)
        self.mouseLock.setVisible(False)
        
        self.Label_Kmean.setVisible(False)
        self.Clustered_map.setVisible(False)
        self.progressBar.setVisible(False)
        
        app = QApplication.instance()
        screen = app.screenAt(self.pos())
        geometry = screen.availableGeometry()
        
        self.move(int(geometry.width() * 0.1), int(geometry.height() * 0.15))
        self.resize(int(geometry.width() * 0.7), int(geometry.height() * 0.6))
        self.screen = screen

        self.Validate_button.setEnabled(False)

#%% Functions
    def loaddata (self):
        self.StackLoc, self.StackDir = gf.getFilePathDialog("Image stack (*.tiff)")  # Ask to open stack of images

        self.expStack = tf.TiffFile(self.StackLoc[0]).asarray() # Import the array
        self.expStack = np.flip(self.expStack, 1) # Flip the array
        self.expStack = np.rot90(self.expStack, k=1, axes=(2, 1)) # Rotate the array
        
        self.displayExpStack(self.expStack)
        
        self.preview.setEnabled(True)

    def Compute_Kmean(self):
        # Obtenir les dimensions du stack d'images
        num_images, height, width = self.expStack.shape

        # Réorganiser les données pour que chaque pixel soit représenté par un vecteur de ses valeurs à travers les images
        pixel_profiles = self.expStack.reshape(num_images, height * width).T

        # Définir le nombre de clusters
        num_clusters = int(self.Cluster_nbr_edit.text())

        # Appliquer l'algorithme K-means
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', batch_size=100)
        kmeans.fit(pixel_profiles)

        # Obtenir les labels des clusters
        labels = kmeans.labels_

        # Convertir les labels en une image segmentée
        self.Kmean_Stack = np.zeros((height, width))
        self.Kmean_Stack = labels.reshape(height, width)
        
        self.Label_Kmean.setVisible(True)
        self.Clustered_map.setVisible(True)
        
        self.Validate_button.setEnabled(True)
        
        self.displayKstack(self.Kmean_Stack)

    def validate(self):

        # Convert labeled image as integer
        Labels_int = np.zeros((len(self.Kmean_Stack),len(self.Kmean_Stack[0])), dtype = int)

        for i in range(0,len(self.Kmean_Stack)):
            for j in range(0,len(self.Kmean_Stack[0])):
                Labels_int[i,j] = int(self.Kmean_Stack[i,j])

        # Definition of the mean profiles for each label
        moyen_profil=np.zeros((len(regionprops(Labels_int)),len(self.expStack[:,0,0])))
        
        try :
            self.progressBar.setVisible(True)
            self.progressBar.setValue(0) # Set the initial value of the Progress bar at 0
            self.progressBar.setRange(0, len(self.expStack[:,0,0])) 
            self.progressBar.setFormat("Saving... %p%")
            
            for i in range (len(self.expStack[:,0,0])) :
                regions = regionprops(Labels_int, intensity_image=self.expStack[i,:,:])
                
                QApplication.processEvents()    
                self.ValSlice = i
                self.progression_bar()
                
                for j in range (len(regions)) :
                    moyen_profil[j][i]=regions[j].mean_intensity 
        except:
            self.popup_message("Kmean","Computation of clustered profiles failed. Check for data.",'icons/Grain_Icons.png')
            return
        
        # Creation of the clustered profiles list
        liste_clusters = np.copy(moyen_profil)

        self.liste = []

        for i in range (0,len(liste_clusters)):
            var = liste_clusters[i,:]
            self.liste.append(var)

        self.liste = np.dstack(self.liste)
        self.liste = np.swapaxes(self.liste, 0, 1)
        
        self.Labels_int = Labels_int
        
        # On recreer la carte des labels correctement
        'Ces 3 étapes sont faites pour eviter d avoir des labels non continus'
        unique_values = np.unique(self.Labels_int)

        # Créer une correspondance entre les valeurs originales et les nouvelles valeurs continues
        value_mapping = {old_value: new_value for new_value, old_value in enumerate(unique_values)}

        # Remplacer les valeurs originales par les nouvelles valeurs continues
        self.Labels_int = np.array([[value_mapping[value] for value in row] for row in self.Labels_int])
        
        ti = time.strftime("%Y-%m-%d__%Hh-%Mm-%Ss") # Absolute time 
        
        directory = "Kmean_" + ti # Name of the main folder
        PathDir = os.path.join(self.StackDir, directory)  # where to create the main folder
        os.mkdir(PathDir)  # Create main folder
        
        tf.imwrite(PathDir + '/Clustered_profiles.tiff', self.liste)
        tf.imwrite(PathDir + '/Labeled_map.tiff', np.rot90(np.flip(self.Labels_int, 0), k=1, axes=(1, 0)).astype('float32'))

        self.close()

    def progression_bar(self): # Function for the ProgressBar uses
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
        
    def displayKstack(self, Series):
        self.Clustered_map.addItem(self.crosshair_v2, ignoreBounds=True)
        self.Clustered_map.addItem(self.crosshair_h2, ignoreBounds=True) 
        
        self.Clustered_map.ui.histogram.show()
        self.Clustered_map.ui.roiBtn.hide()
        self.Clustered_map.ui.menuBtn.hide()
        
        view = self.Clustered_map.getView()
        state = view.getState()        
        self.Clustered_map.setImage(Series) 
        view.setState(state)
        view.setBackgroundColor(self.parent.color1)
        
        histplot = self.Clustered_map.getHistogramWidget()
        histplot.setBackground(self.parent.color1)
        
        histplot.region.setBrush(pg.mkBrush(self.parent.color5 + (120,)))
        histplot.region.setHoverBrush(pg.mkBrush(self.parent.color5 + (60,)))
        histplot.region.pen = pg.mkPen(self.parent.color5)
        histplot.region.lines[0].setPen(pg.mkPen(self.parent.color5, width=2))
        histplot.region.lines[1].setPen(pg.mkPen(self.parent.color5, width=2))
        histplot.fillHistogram(color = self.parent.color5)        
        histplot.autoHistogramRange()
        
        self.Clustered_map.setColorMap(pg.colormap.get('viridis'))
    
    def defaultIV(self):
        self.expSeries.clear()
        self.expSeries.ui.histogram.hide()
        self.expSeries.ui.roiBtn.hide()
        self.expSeries.ui.menuBtn.hide()
        
        view = self.expSeries.getView()
        view.setBackgroundColor(self.parent.color1)
        
        ROIplot = self.expSeries.getRoiPlot()
        ROIplot.setBackground(self.parent.color1)
        
    def mouseMoved(self, e):
        pos = e[0]
        sender = self.sender()
  
        if not self.mouseLock.isChecked():
            if self.expSeries.view.sceneBoundingRect().contains(pos)\
                or self.Clustered_map.view.sceneBoundingRect().contains(pos):    
                
                if sender == self.proxy1:
                    item = self.expSeries.view
                elif sender == self.proxy2:
                    item = self.Clustered_map.view
                
                mousePoint = item.mapSceneToView(pos) 
                     
                self.crosshair_v1.setPos(mousePoint.x())
                self.crosshair_h1.setPos(mousePoint.y())
                
                self.crosshair_v2.setPos(mousePoint.x())
                self.crosshair_h2.setPos(mousePoint.y())  
                
            try:
                self.x = int(mousePoint.x())
                self.y = int(mousePoint.y())
                
                self.printClick(self.x, self.y, sender)
            except:
                pass

    def mouseClick(self, e):
        pos = e[0]
        sender = self.sender()
        
        self.mouseLock.toggle()
        
        if self.expSeries.view.sceneBoundingRect().contains(pos)\
            or self.Clustered_map.view.sceneBoundingRect().contains(pos):
                
                if sender == self.proxy1:
                    item = self.expSeries.view
                elif sender == self.proxy2:
                    item = self.Clustered_map.view
                
                mousePoint = item.mapSceneToView(pos) 
    
                self.crosshair_v1.setPos(mousePoint.x())
                self.crosshair_h1.setPos(mousePoint.y())
                
                self.crosshair_v2.setPos(mousePoint.x())
                self.crosshair_h2.setPos(mousePoint.y())  
            
    def printClick(self, x, y, sender):
        try:
            self.Label_Kmean.setText("Kmean label: " + str(self.Kmean_Stack[x, y]))
        except:
            pass