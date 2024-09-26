# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:18:21 2024

@author: glhote1
"""

#%% Imports

from inspect import getsourcefile
from os.path import abspath

from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication

from inichord import general_functions as gf
import tifffile as tf
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
from scipy.fft import fft2, ifft2, fftshift

path2thisFile = abspath(getsourcefile(lambda:0))
uiclass, baseclass = pg.Qt.loadUiType(os.path.dirname(path2thisFile) + "/Extract_mosaic_ui_02.ui")

class MainWindow(uiclass, baseclass):
    def __init__(self,parent):
        super().__init__()

        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon('icons/Main_icon.png')) # Application of the main icon
        self.parent = parent

        self.OpenData.clicked.connect(self.loaddata) # Load data
        self.Xtract_series.clicked.connect(self.serie_extraction) # Extract the different series 
        self.Erroneous_bttn.clicked.connect(self.image_correction) # Replace image with problem
        self.Save_bttn.clicked.connect(self.save_data) # Save series in a given folder
        self.ProcessRef_bttn.clicked.connect(self.Ref_processing) # Process images references 
        
        self.defaultIV() # Default ImageView (when no image)
        
        self.flag_FFTCorrection = 0 # Initialization
        
        # disable buttons at the beginning
        self.nbrSeries.setEnabled(False)
        self.First_img.setEnabled(False)
        self.DriftCorr_choice.setEnabled(False)
        self.Xtract_series.setEnabled(False)
        self.Threshold.setEnabled(False)
        self.Erroneous_bttn.setEnabled(False)
        self.Save_bttn.setEnabled(False)
        
        app = QApplication.instance()
        screen = app.screenAt(self.pos())
        geometry = screen.availableGeometry()
        
        # Position (self.move) and size (self.resize) of the main GUI on the screen
        self.move(int(geometry.width() * 0.1), int(geometry.height() * 0.1))
        self.resize(int(geometry.width() * 0.8), int(geometry.height() * 0.6))
        self.screen = screen

    def Ref_processing(self):
        # Import reference images
        StackLoc, StackDir = gf.getFilePathDialog("Reference images") # Image importation
        self.image_ref = [] 
        
        for i in range(0,len(StackLoc)):
            Var = tf.TiffFile(StackLoc[i]).asarray() # Import the unit 2D array
            self.image_ref.append(Var) # Append every 2D array in a list named self.image_ref
            
        self.image_ref = np.dstack(self.image_ref) # Convert list of 3D array
        self.image_ref = np.swapaxes(self.image_ref, 0, 2) # Rearrangement of the axis
        self.image_ref = np.swapaxes(self.image_ref, 1, 2) # Rearrangement of the axis
            
        self.image_ref = self.image_ref[::-1]
        # Ask for the saving folder
        root = tk.Tk()
        root.withdraw()  # Masquer la fenêtre principale Tkinter
    
        # Ouvrir une boîte de dialogue pour choisir un dossier de sauvegarde
        dossier = filedialog.askdirectory(title="Choose a folder for reference images.")
        # Save in the folder with the good index

        for i in range(self.image_ref.shape[0]):
            # Générer le nom du fichier avec un numéro incrémenté
            chemin_complet = os.path.join(dossier, f"reference_{i}.tiff")
        
            # Sauvegarder le tableau 2D en tant que fichier TIFF
            tf.imwrite(chemin_complet, self.image_ref[i])
            
        # Finished message
        self.parent.popup_message("Extract mosaic","Reference images have been saved.",'icons/Main_icon.png')

    def loaddata(self):
        StackLoc, StackDir = gf.getFilePathDialog("Images") # Image importation
        
        self.image = [] # Initialization of the variable self.image (which will be the full image folder)
        
        self.progressBar.setValue(0) # Set the initial value of the Progress bar at 0
        self.progressBar.setRange(0, len(StackLoc)-1) 
        self.progressBar.setFormat("Loading images... %p%")

        for i in range(0,len(StackLoc)):
            Var = tf.TiffFile(StackLoc[i]).asarray() # Import the unit 2D array
            self.image.append(Var) # Append every 2D array in a list named self.image
            
            QApplication.processEvents()    
            self.ValSlice = i
            self.progression_bar()
            
        self.image = np.dstack(self.image) # Convert list of 3D array
        self.image = np.swapaxes(self.image, 0, 2) # Rearrangement of the axis
        self.image = np.swapaxes(self.image, 1, 2) # Rearrangement of the axis
        
        self.image_displayed = np.flip(self.image, 1) # Flip the array
        self.image_displayed = np.rot90(self.image_displayed, k=1, axes=(2, 1)) # Rotate the array
        
        self.displayExpStack(self.image_displayed) # Display the 3D array

        # enables buttons at the beginning
        self.nbrSeries.setEnabled(True)
        self.First_img.setEnabled(True)
        self.DriftCorr_choice.setEnabled(True)
        self.Xtract_series.setEnabled(True)
        
    def serie_extraction(self):
        self.series_nbr = self.nbrSeries.value() # Number of series involved
        self.First_image = self.First_img.value() # Number of the first image
        
        # To define if drift correction image has been acquired
        self.correction_choice = self.DriftCorr_choice.currentText()
        if self.correction_choice == "Yes":
            self.flag_corr = 1
        else :
            self.flag_corr = 0
            
        image_reduction = self.image[self.First_image-1:,:,:]
        self.images_dict = {} # Dictionnary creation

        if self.flag_corr == 1 :
            intervalle = self.series_nbr + 1
            indices_a_conserver = [i for i in range(image_reduction.shape[0]) if (i + 1) % intervalle != 0]
            image_reduction = image_reduction[indices_a_conserver]

        self.images_dict = {f"images_{i}": [] for i in range(self.series_nbr)}
            
        # Parcourir chaque image 2D dans le tableau 3D
        for index, image2D in enumerate(image_reduction):
            # Déterminer à quel groupe (liste) l'image doit être ajoutée
            groupe_index = index % self.series_nbr
            self.images_dict[f"images_{groupe_index}"].append(image2D)

        # Appliquer les opérations numpy à chaque liste du dictionnaire
        i= 0
        for key in self.images_dict:
            self.progressBar.setValue(0) # Set the initial value of the Progress bar at 0
            self.progressBar.setRange(0, len(self.images_dict)-1)
            self.progressBar.setFormat("Create series...")
            
            # Si la liste n'est pas vide
            if self.images_dict[key]:
                # Combiner les images de la liste le long du troisième axe avec dstack
                combined_image = np.dstack(self.images_dict[key])

                # Appliquer les deux swapaxes successifs
                transformed_image = np.swapaxes(combined_image, 0, 2)
                transformed_image = np.swapaxes(transformed_image, 1, 2)

                # Mettre à jour le dictionnaire avec l'image transformée
                self.images_dict[key] = transformed_image
            else:
                self.images_dict[key] = None  # Gérer les cas où il n'y a pas d'images dans le groupe

            QApplication.processEvents()    
            self.ValSlice = i
            self.progression_bar()
            
            i = i+1
        
        self.progressBar.setFormat("Series have been created")
        
        # Enables buttons
        self.Threshold.setEnabled(True)
        self.Erroneous_bttn.setEnabled(True)
        self.Save_bttn.setEnabled(True)

    def image_correction(self):
        self.Threshold_value = self.Threshold.value()
        self.Corrected_dict = {f"images_{i}": [] for i in range(self.series_nbr)}
        
        self.progressBar.setValue(0) # Set the initial value of the Progress bar at 0
        self.progressBar.setRange(0, len(self.images_dict)-1)
        self.progressBar.setFormat("Correct series... %p%")
        
        for i, nom in enumerate(list(self.images_dict.keys())):
            array_3d = self.images_dict[nom]
            distance = []
        
            # Comparer chaque image avec l'image de référence
            for j in range(0, array_3d.shape[0]):
                if j == 0:
                    image_reference = array_3d[j]
                else:
                    image_reference = array_3d[j-1]
                    
                decalage = self.Compute_shift(image_reference, array_3d[j])
                distance_decalage = np.linalg.norm(decalage)
                
                distance.append(distance_decalage)
        
            index = np.hstack(distance)
            index = index < self.Threshold_value
            
            # Parcourir l'array pour trouver les paires de 'False'
            for k in range(len(index) - 1):
                # Vérifier si deux éléments consécutifs sont 'False'
                if index[k] == False and index[k + 1] == False:       
                    array_3d[k] = array_3d[k - 1]
        
            self.Corrected_dict[f"images_{i}"].append(array_3d)
            
            QApplication.processEvents()    
            self.ValSlice = i
            self.progression_bar()

        self.progressBar.setFormat("Series have been corrected")
        self.flag_FFTCorrection = 1

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

    def save_data(self):
        if self.flag_FFTCorrection == 1:
            self.sauvegarder_tableaux(self.Corrected_dict)
        elif self.flag_FFTCorrection == 0:
            self.sauvegarder_tableaux(self.images_dict)
        
    def sauvegarder_tableaux(self,tableaux):
        # Initialiser Tkinter
        root = tk.Tk()
        root.withdraw()  # Masquer la fenêtre principale Tkinter
    
        # Ouvrir une boîte de dialogue pour choisir un dossier de sauvegarde
        dossier = filedialog.askdirectory(title="Choose a folder for image series.")
    
        self.progressBar.setValue(0) # Set the initial value of the Progress bar at 0
        self.progressBar.setRange(0, len(self.images_dict)-1) 
        self.progressBar.setFormat("Save series...%p%")
    
        # Vérifier si un dossier a été sélectionné
        if dossier:
            # Récupérer les clés du dictionnaire et les inverser
            cles_inverses = list(reversed(list(tableaux.keys())))
    
            # Boucle pour chaque clé et tableau dans le dictionnaire avec indices inversés
            for i, nom in enumerate(cles_inverses, start=1):
                tableau = tableaux[nom]
                # Définir le chemin complet du fichier avec le nom de la clé et l'indice inversé
                chemin_complet = os.path.join(dossier, f"serie_{i}.tiff")
    
                # Sauvegarder le tableau 3D au format TIFF avec le type de données d'origine
                tf.imwrite(chemin_complet, tableau)
                
                QApplication.processEvents()    
                self.ValSlice = i
                self.progression_bar()
        else:
            print("Aucun dossier sélectionné. Annulation de la sauvegarde.")
            
        self.progressBar.setFormat("Series have been saved")

    def progression_bar(self): # Function for the ProgressBar uses
        self.prgbar = self.ValSlice
        self.progressBar.setValue(self.prgbar)
        
    def defaultIV(self):
        self.expSeries.clear()
        self.expSeries.ui.histogram.hide()
        self.expSeries.ui.roiBtn.hide()
        self.expSeries.ui.menuBtn.hide()
        
        view = self.expSeries.getView()
        view.setBackgroundColor(self.parent.color1)
        
        ROIplot = self.expSeries.getRoiPlot()
        ROIplot.setBackground(self.parent.color1)
        
    def displayExpStack(self, series):   
        self.expSeries.ui.histogram.hide()
        self.expSeries.ui.roiBtn.hide()
        self.expSeries.ui.menuBtn.hide()
        
        view = self.expSeries.getView() # Extract view
        state = view.getState() # Extract state
        self.expSeries.setImage(series) # Add the wanted data inside the ImageView
        view.setState(state)
        
        view.setBackgroundColor(self.parent.color1) # Define the background color
        ROIplot = self.expSeries.getRoiPlot() # Extract the ROI
        ROIplot.setBackground(self.parent.color1) # Define the ROI background color
        
        font=QtGui.QFont('Noto Sans Cond', 8) # Set fontsize
        ROIplot.getAxis("bottom").setTextPen('k') # Apply color of the ticks label
        ROIplot.getAxis("bottom").setTickFont(font) # Apply size of the ticks label
        
        self.expSeries.timeLine.setPen(color=self.parent.color3, width=15) # Define timeLine line color and width
        self.expSeries.frameTicks.setPen(color=self.parent.color1, width=5) # Define ticks color and width
        self.expSeries.frameTicks.setYRange((0, 1))

        s = self.expSeries.ui.splitter
        s.handle(1).setEnabled(True) # Allow to change splitter height
        s.setStyleSheet("background: 5px white;") # Define splitter background color 
        s.setHandleWidth(5) # Define splitter width
