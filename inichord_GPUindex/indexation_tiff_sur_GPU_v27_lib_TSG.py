#! J:\Program Files\Python310\envs\Image\Scripts\python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 20:53:26 2023
@author: clanglois1
"""
import inichord_GPUindex.Xallo as xa
from inichord_GPUindex import Symetry as sy
from inichord_GPUindex import Fct_profil_modification_02 as fct
from inichord_GPUindex import general_functions as gf
from PyQt5.QtWidgets import QApplication

from pyquaternion import Quaternion
import cupy as cp
import h5py
import numpy as np
import tifffile as tf
import time

from PyQt5.QtWidgets import QMessageBox
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap

def DBopen(DB):
    '''    
    Parameters
    ----------
    DB : TYPE format .h5
        DESCRIPTION. database CHORDv3 non modifiée

    Returns
    -------
    listLabelNames : TYPE Python list
        DESCRIPTION. Contient les clés d'accès aux datasets contenant les quaternions dans le fichier h5'
    listChunksNames : TYPE Python list
        DESCRIPTION. Contient les clés d'accès aux datasets contenant les profils dans le fichier h
    listChunkArrays : TYPE Python list
        DESCRIPTION. chaque objet de la liste est un des datasets récupérés dans le fichier h5 qui contient les profils
    listLabelArray : TYPE Python list
        DESCRIPTION.
    rawProfileLength : TYPE
        DESCRIPTION. chaque objet de la liste est un des datasets récupérés dans le fichier h5 qui contient les quaternions

    '''
    
    # lecture du fichier de profils theoriques test
    f = h5py.File(DB, 'r')
    
    listKeys = gf.get_dataset_keys(f)
    listGroups = gf.get_group_keys(f)
    
    # création des listes contenant les paths des profils théoriques et orientation dans la base    

    # check de la longueur des profils dans la base
    rawProfileLength = 0
    
    for i in listGroups:
        if "Sampling" in f[i].attrs.keys():
            rawProfileLength = int(f[i].attrs["Sampling"])
            
    if rawProfileLength == 0:
        rawProfileLength = 360 # valeur par défaut si Sampling non trouvé dans la base
    
    listLabelNames = []
    listChunksNames = []
    
    listChunkArrays = []
    listLabelArray = []
    
    for i in listKeys:
        if "LabelChunk" in i:
            listLabelNames.append(i)
        elif "DataChunk" in i:
            listChunksNames.append(i)

    for i in range(len(listChunksNames)):
        listChunkArrays.append(np.asarray(f[listChunksNames[i]]))
        listLabelArray.append(np.asarray(f[listLabelNames[i]]))

        # listChunkArray est une liste dont les éléments sont des tableaux Numpy
        # de 250_000 * rawProfileLength, le tout rangé en ligne, donc de dim 1
    
    return listLabelNames, listChunksNames, listChunkArrays, listLabelArray, rawProfileLength     

#%% classes et fonctions liées à l'indexation

class IndexationGPUderiv:
    '''
    Le programme prend en entrée :

    le tableau Numpy rawImage est constitué de :
        # en colonne (selon axe 0) les profils.
        # en axe 1, la dimension de l'axe 1 est le nombre de lignes de l'image (sa hauteur)
        # en axe 2, la dimension de l'axe 2 (en profondeur) correspond à la largeur de l'image
    
    savePath est le nom du répertoire où stocker le fichier de résultat (sans slash à la fin)
    
    Database et CIFfile sont des chemins vers les fichiers correspondants,
    normalement obtenus en instanciant la classe "preindexation".
    nbSTACK est le nombre de profils de la stack que la carte est capable de prendre
    nbDB est le nb de profils de la base que la carte graphique peut prendre, 
    en conjonction avec nbSTACK
    
    dimPROF est la dimension d'un profil de la stack'
    
    
    Le but de la fonction est d'indexer sur GPU les profils de la stack qu'on lui donne
    à l'instanciation, l'indexation se met en route (grâce à __init__). La 
    fonction sauve le résultat dans un fichier h5 et également dans des sorties
    texte en quaternions et en Euler. 
    '''    

    def __init__(self, parent, Image, savePath, database, CIFfile, Workflow=[['Diff',0]], normType = "centered euclidian", nbSTACK=20_000, nbDB = 20_000, dimPROF = 180):
        self.parent = parent      

        # instruction pour afficher les tableaux de façon lisible
        np.set_printoptions(suppress=True, precision=5)
        self.mempool = cp.get_default_memory_pool()
        self.pinned_mempool = cp.get_default_pinned_memory_pool()
        
        with cp.cuda.Device(0):
            self.mempool.set_limit(fraction=1.0)    
        
        self.savePath = savePath   
       
        self.rawImage = Image    
        self.DB = database
        self.CIF = CIFfile

        self.nbSTACK = nbSTACK
        self.nbDB = nbDB
        self.dimPROF = dimPROF

        self.Workflow = Workflow
        self.normType = normType
        
        self.dimExpProfiles = self.rawImage.shape[0]
        self.height = self.rawImage.shape[1]
        
        if len(self.rawImage.shape) < 3:
            self.width = 1
        else:
            self.width = self.rawImage.shape[2]
        
        self.listChunksNames = []
        self.listLabelNames = []
        
        self.listChunkArrays = []
        self.listLabelArray = []
        self.testArrayList = []

        self.maxDistList = []
        self.rawIndicesList = []
        self.miniChunk_finalList = []
        self.whichDataChunkList = []

        self.SymM = sy.get_proper_matrices_from_CIF(CIFfile)
        self.SymQ = sy.get_proper_quaternions_from_CIF(CIFfile)
        
        self.actualProfLength = 0
        self.expChunkNB = 0
        self.ti = time.strftime("%Y-%m-%d__%Hh-%Mm-%Ss")
    
    def runIndexation(self):

        self.dataPrepDiff()     
        self.nbSTACK = int(self.nbSTACK * self.dimPROF / self.actualProfLength)              
        self.expPrepDiff()        

        self.initIndexation()

        self.indexationDiffspeed()
        
        self.postIndexation()
        
        self.quality_map_computation()
        
        self.savingRes()
        
        self.savingMTEX()
        
        self.savingATEX()

    def dataPrepDiff(self):
        
        self.parent.Info_box.ensureCursorVisible()
        self.parent.Info_box.insertPlainText("\n \u2022 Theoretical data preparation.")
        QApplication.processEvents()
        
        self.listLabelNames, self.listChunksNames, self.listChunkArrays, self.listLabelArray, self.rawProfileLength = DBopen(self.DB)
        
        # détermination du facteur de réduction
        actualRatio = self.rawProfileLength / self.dimExpProfiles
        if actualRatio < 1.0:
            self.parent.popup_message("Indexation","Increase theoretical profile length",'icons/Indexation_icon.png')
            return

        elif actualRatio == 1.0: # les profils theo et exp ont la même longueur
            reductionFactor = 1
        else:
            if (self.rawProfileLength % self.dimExpProfiles) == 0:
                reductionFactor = np.floor_divide(self.rawProfileLength, self.dimExpProfiles)
            else:
                self.parent.popup_message("Indexation","dim exp profiles must divide dim theoretical profiles",'icons/Indexation_icon.png')
                return
                
        self.actualProfLength = int(self.rawProfileLength / reductionFactor)
        
        self.listChunkArraysDiff = []
        
        for i in range(len(self.listChunkArrays)):
            self.a = fct.downSampleProfiles(self.listChunkArrays[i], reductionFactor) # réduction de 360 à 180 mais profils toujours en ligne (1D array)
            self.listChunkArrays[i] = self.a
            self.b = fct.reshapeProfilesInLine(self.a, self.actualProfLength)
            self.c = fct.Profile_modifier(self.b, self.Workflow, self.normType, axProf = 1)
            
            self.listChunkArraysDiff.append(self.c)
            
    def expPrepDiff(self):
        
        self.parent.Info_box.ensureCursorVisible()
        self.parent.Info_box.insertPlainText("\n \u2022 Experimental data preparation.")
        QApplication.processEvents()
        
        rawImage2D = self.rawImage.reshape(self.dimExpProfiles, self.height * self.width)
        
        diffImage2D = fct.Profile_modifier(rawImage2D, self.Workflow, self.normType, axProf = 0)
        self.expChunkNB = np.floor_divide(self.height * self.width, self.nbSTACK)
        self.remainSTACK = (self.height * self.width) % self.nbSTACK
        
        # pour les profils test, il y aura un nombre expChunkNB de chunks contenant nbSTACK*factorGPU pixels
        # il restera à passer une matrice de remainSTACK pixels (ceux de la fin de l'axe 1)
        
        i = 0
        for i in range(self.expChunkNB):
            self.testArrayList.append(diffImage2D[:,i*self.nbSTACK:(i+1)*self.nbSTACK])
        
        self.remain = 0
        if self.remainSTACK != 0:
            self.remain = 1
            self.testArrayList.append(diffImage2D[:, -self.remainSTACK:])
    
    def initIndexation(self):
        self.dbChunks = self.nbDB # 1/8 des profils d'un DataChunk. 62_500 sur UltraCalculus, 25_000 sur ordi perso
        self.loopDB = int(np.floor_divide(len(self.listChunkArrays[0]) / self.actualProfLength, self.dbChunks))

    def progression_bar(self): # Fonction relative à la barre de progression
        self.prgbar = self.ValSlice
        self.parent.progressBar.setValue(self.prgbar)

    def indexationDiffspeed(self):
        t1 = time.time()
        
        self.parent.Info_box.ensureCursorVisible()
        self.parent.Info_box.insertPlainText("\n \u2022 Indexation in progress.")
        QApplication.processEvents()
        
        self.prgbar = 0 # Progress bar initial value
        self.parent.progressBar.setValue(self.prgbar)
        progress_len = (self.expChunkNB + 1 * self.remain) * len(self.listChunksNames)
        self.parent.progressBar.setFormat("Indexation: %p%")
        self.parent.progressBar.setRange(0, progress_len) # Set the range according to the number of batches
        step = 0

        for y in range(self.expChunkNB + 1 * self.remain): # +1 pour prendre en compte le testChunk avec le reste
            
            normedGPUtest = cp.array(self.testArrayList[y])
        
            nbSTACKcurr =  len(self.testArrayList[y][0])  
        
            distDataChunk = np.zeros((len(self.listChunksNames), nbSTACKcurr))
            indDataChunk = np.zeros((len(self.listChunksNames), nbSTACKcurr))
            miniChunkInd = np.zeros((len(self.listChunksNames), nbSTACKcurr))
            
            for j in range(len(self.listChunksNames)):
                
                step = step + 1

                # on charge un chunk entier  en ligne que l'on remet en matrice
                listDist = np.zeros((self.loopDB, nbSTACKcurr))
                listInd = np.zeros((self.loopDB, nbSTACKcurr))
                
                QApplication.processEvents() 
                self.ValSlice = step
                self.progression_bar()
                                
                for k in range(self.loopDB): # on traite le DataChunk courant par petits bouts
           
                    normedGPU = cp.array(self.listChunkArraysDiff[j][k*self.dbChunks:(k+1)*self.dbChunks, :])

                    # calcul du tableau de distances
                    distances = cp.matmul(normedGPU, normedGPUtest)
                    del normedGPU

                    self.mempool.free_all_blocks()
                    self.pinned_mempool.free_all_blocks()

                    listDist[k, :] = cp.asnumpy(cp.max(distances, axis=0))
                    listInd[k,:] = cp.asnumpy(np.argmax(distances, axis=0))
                    del distances  
                    
                    # on libère la mémoire
                    self.mempool.free_all_blocks()
                    self.pinned_mempool.free_all_blocks()
                    
                maxInd = np.argsort(listDist, axis=0)
                miniChunkInd[j, :] = maxInd[-1, :]
                distDataChunk[j, :] = np.take_along_axis(listDist, maxInd, axis=0)[-1, :]
                indDataChunk[j, :] = np.take_along_axis(listInd, maxInd, axis=0)[-1, :]
                
            maxInd_array = cp.argsort(distDataChunk, axis=0)
            maxInd_array = cp.asnumpy(maxInd_array)
            
            self.maxDistList.append(np.take_along_axis(distDataChunk, maxInd_array, axis=0)[-1:, :])
            self.rawIndicesList.append(np.take_along_axis(indDataChunk, maxInd_array, axis=0)[-1:, :])
            self.miniChunk_finalList.append(np.take_along_axis(miniChunkInd, maxInd_array, axis=0)[-1:, :])
            self.whichDataChunkList.append(maxInd_array[-1:, :])
              
        del normedGPUtest
        self.mempool.free_all_blocks()
        self.pinned_mempool.free_all_blocks()
        t2 = time.time()
        self.Processing_time = t2-t1
        self.Processing_time = np.round(self.Processing_time,1)
        
        text = f"\n \u2022 Indexation time (s) : {self.Processing_time}."
        self.parent.Info_box.ensureCursorVisible()
        self.parent.Info_box.insertPlainText(text)
        QApplication.processEvents()

    def postIndexation(self):
        indexedStackList = []
        oriList = []
        Pr_list = []
        
        for y in range(self.expChunkNB + 1 * self.remain):
            indexedStackList.append(np.zeros((1, self.actualProfLength, len(self.testArrayList[y][0, :]) )))
            oriList.append(np.zeros((1, 4, len(self.testArrayList[y][0, :])))) # 4 because Quaternions 
        
        for n in range(self.expChunkNB + 1 * self.remain):
            
            for iTP in range(len(self.testArrayList[n][0, :])):
                
                DataChunk = fct.reshapeProfilesInLine(self.listChunkArrays[self.whichDataChunkList[n][0, iTP]], self.actualProfLength)
                DataOri = fct.reshapeProfilesInLine(self.listLabelArray[self.whichDataChunkList[n][0, iTP]], 4)
                
                indInDataChunk = int(self.rawIndicesList[n][0, iTP] + self.miniChunk_finalList[n][0, iTP] * self.dbChunks)
                
                Pr_list.append(indInDataChunk)
                
                ori = DataOri[indInDataChunk, :]
    
                profile = DataChunk[indInDataChunk, :]
                
                if self.normType == "euclidian":
                    profile = fct.normMatProfiles(profile, 0)
                elif self.normType == "centered euclidian":
                    profile = fct.centeredEuclidianNorm(profile, 0)
                elif self.normType == "centered_std":
                    profile = fct.centeredSTD(profile, 0)     
                
                indexedStackList[n][0, :, iTP] = profile
                oriList[n][0, :, iTP] = ori
                
                del profile
            
                self.mempool.free_all_blocks()
                self.pinned_mempool.free_all_blocks()           
        
        # concaténation
        nScoresStack = np.concatenate(indexedStackList, axis = -1)
        #Ajout GLDH
        
        nScoresDist = np.concatenate(self.maxDistList, axis = -1)
        nScoresOri = np.concatenate(oriList, axis = -1)
        
        whichDataChunkList = np.concatenate(self.whichDataChunkList,axis=1)
        Ref_Pr_list = np.reshape(np.vstack(Pr_list),(-1,1)).T
        
        # N° du profil théorique pour chaque pixel de l'image et pour chaque score
        Ref_Pr_list = Ref_Pr_list+(whichDataChunkList*self.loopDB*self.dbChunks)
        
        # N° du profil théorique pour chaque score (suivant X et Y)
        self.Ref_Pr_list2 = np.reshape(Ref_Pr_list,(1,self.height, self.width))
        
        self.nScoresStack = nScoresStack.reshape((1, self.actualProfLength, self.height, self.width))
                        
        self.nScoresDist = nScoresDist.reshape((1, self.height, self.width))
        self.nScoresOri = nScoresOri.reshape((1, 4, self.height, self.width))
        
        # Ajout GDLH pour récupération des profils avec traitement ==> A améliorer 
        
        self.Treatment_theo_prof = self.nScoresStack
        self.Treatment_theo_prof = self.Treatment_theo_prof.reshape((self.actualProfLength, self.height * self.width))
        self.Treatment_theo_prof = fct.Profile_modifier(self.Treatment_theo_prof, self.Workflow, self.normType, axProf = 0)
        self.Treatment_theo_prof = self.Treatment_theo_prof.reshape((1, self.actualProfLength, self.height , self.width))
        
        # Ajout GDLH pour récupération des profils expérimentaux avec traitement ==> A améliorer 
        self.testArrayList = np.hstack(self.testArrayList)
        self.testArrayList = self.testArrayList.reshape((self.actualProfLength , self.height , self.width))
        
        if self.normType == "euclidian":
            rawImage = fct.normMatProfiles(self.rawImage, 0)
        elif self.normType == "centered euclidian":
            rawImage = fct.centeredEuclidianNorm(self.rawImage, 0)
        elif self.normType == "centered_std":
            rawImage = fct.centeredSTD(self.rawImage, 0)
        
        self.rawImage = rawImage
        self.mempool.free_all_blocks()
        self.pinned_mempool.free_all_blocks()

    def quality_map_computation(self):
        self.wind_NCC = int(np.round(0.1*self.actualProfLength)) # 1/10 of the total length 
        # Computation of quality map
        self.qualmap = self.NCC_computation(self.nScoresStack[0,:,:,:],self.rawImage, batchsize = 5000, Windows = self.wind_NCC)
        
        self.quality_map = self.qualmap *100 # X100 to display in %

    def NCC_computation(self,Theo_stack,rawImage, batchsize = 5000, Windows = 18): # Normalized covariance correlation calculation
        var,batch_nbr = self.find_batch_nbr(len(Theo_stack[0]),len(Theo_stack[0][0]),batchsize) # Find optimal batch for cupy uses
        
        self.prgbar = 0 # Progress bar initial value
        self.parent.progressBar.setValue(self.prgbar)
        self.parent.progressBar.setFormat("Quality computation: %p%")
        self.parent.progressBar.setRange(0, int(batch_nbr)-1) # Set the range accordingly to the number of labels
        
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        
        incr = np.linspace(0,len(Theo_stack),Windows+1)
        corrcoeff = np.zeros((len(Theo_stack[0])*len(Theo_stack[0][0])))
        
        test_theo = Theo_stack.reshape((len(Theo_stack),len(Theo_stack[0])*len(Theo_stack[0][0])))
        test_exp = rawImage.reshape((len(rawImage),len(rawImage[0])*len(rawImage[0][0])))
        
        for i in range(int(batch_nbr)):
            QApplication.processEvents()    
            self.ValSlice = i
            self.progression_bar()
            
            NCC = np.zeros(var) # NCC for Normalized Covariance Correlation
            
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            
            for k in range(0,Windows):
                Theo_stack_wind = cp.asarray(test_theo[int(incr[k]) : int(incr[k+1]) ,var*i:var*(i+1)])
                rawImage_wind = cp.asarray(test_exp[int(incr[k]) : int(incr[k+1]) ,var*i:var*(i+1)])
                
                NCC_var = cp.corrcoef(Theo_stack_wind,rawImage_wind, rowvar = False)
                NCC_var2 = cp.diag(NCC_var[:len(NCC_var)//2, len(NCC_var)//2:])
                NCC_var2 = cp.asnumpy(NCC_var2)
                
                NCC = NCC + NCC_var2
                
                del(NCC_var,NCC_var2)
                
            NCC = NCC/Windows
            
            corrcoeff[var*i:var*(i+1)] = NCC
            
        corrcoeff2 = corrcoeff.reshape((len(Theo_stack[0]),len(Theo_stack[0][0])))
        
        return corrcoeff2

    def find_batch_nbr(self,height,width,batch):  # Find optimal batch for cupy uses
        i = 0
        while height*width % (batch-i) != 0:
            i += 1
        
        return batch-i, height*width/(batch-i)

    def savingRes(self):
        ti = time.strftime("%Y-%m-%d__%Hh-%Mm-%Ss")
        
        indexSTACK = h5py.File(self.savePath + '\indexScores_'+ ti + '.hdf5', 'a')
        
        group = indexSTACK.create_group('indexation')
        group.create_dataset(name='nScoresStack', data=self.nScoresStack)
        group.create_dataset(name='Treatment_theo_prof', data=self.Treatment_theo_prof) #Profil théo modifiés
        group.create_dataset(name='rawImage', data=self.rawImage)
        group.create_dataset(name='nScoresDist', data=self.nScoresDist)
        group.create_dataset(name='nScoresOri', data=self.nScoresOri)
        group.create_dataset(name='Ref_Pr_list3', data=self.Ref_Pr_list2)
        group.create_dataset(name='testArrayList', data=self.testArrayList) #Profil expé modifiés
        group.create_dataset(name='quality_map', data=self.quality_map) #Profil expé modifiés
                 
        group.attrs.create("profile length", self.actualProfLength)
        group.attrs.create("dbChunks", self.dbChunks)
        group.attrs.create("height", self.height)
        group.attrs.create("width", self.width)
        group.attrs.create("CIF path", self.CIF)
        group.attrs.create("stack path", self.savePath)
        group.attrs.create("database path", self.DB)
        group.attrs.create("normalization before indexation", self.normType)
        group.attrs.create("metric for Indexation", "cosine")
        group.attrs.create("nbSTACK", self.nbSTACK)
        group.attrs.create("nbDB", self.nbDB)
        
        indexSTACK.flush()
        indexSTACK.close()

        self.parent.Info_box.ensureCursorVisible()
        self.parent.Info_box.insertPlainText("\n \u2022 H5 file saved.")
        QApplication.processEvents()

    def savingMTEX(self):
        
        Quat = self.nScoresOri[-1,:,:,:]
        x = len(Quat[0])
        y = len(Quat[0][0])

        self.ti = time.strftime("%Y-%m-%d__%Hh-%Mm-%Ss")

        with open(self.savePath + '\indexGPU_'+ self.ti + '.quatCHORDv3-CTFxyConv.txt', 'w') as file:

            for i in range(x):
                for j in range(y):

                    index = 1    
                    if Quat[0,i,j] ==0 :
                        index = 0
                    file.write(str(index) + '\t' + str(j) + '\t' + str(i) + '\t' + str(Quat[0, i, j]) +
                    '\t' + str(Quat[1, i, j])  + '\t' + str(Quat[2, i, j])  + '\t' + str(Quat[3, i, j]) + '\n')
                    
        self.parent.Info_box.ensureCursorVisible()
        self.parent.Info_box.insertPlainText("\n \u2022 MTex file saved.")
        QApplication.processEvents()

    def savingATEX(self):
        
        Quat = self.nScoresOri[-1,:,:,:]
        
        # print("saving ATEX file...")
        Q = np.copy(Quat)
        Q = np.flip(Q, 1)
        Q = np.rot90(Q, k=1, axes=(2, 1))
        
        x = len(Q[0])
        y = len(Q[0][0])

        with open(self.savePath + '\indexGPU_'+ self.ti + '.quatCHORDv3-Euler-CTFxyConv.txt', 'w') as file:
            euler = np.zeros((3))
            
            
            for i in range(x):
                for j in range(y):

                    index = 1    
                    if Q[0,i,j] == 0:
                        index = 0
                    quatObj = Quaternion(Q[:, i, j])
                    quatObjinv = quatObj.inverse
                    euler = xa.Quat2Euler(quatObjinv.elements)
                    
                    
                    file.write(str(index) + '\t' + str(i) + '\t' + str(j) + '\t' + str(euler[0]) +
                    '\t' + str(euler[1])  + '\t' + str(euler[2]) + '\n')
                    
        self.parent.Info_box.ensureCursorVisible()
        self.parent.Info_box.insertPlainText("\n \u2022 ATEX file saved.")
        QApplication.processEvents()

#%% Pre-indexation

class preIndexation:
    """
    Classe permettant d'entrer en mémoire la stack à indexer, ainsi que le 
    fichier CIF correspondant (indexation monophasée) et la base de données. 
    Les dialogues "utilisateurs" se font au travers de la librairie 
    "General_functions".
    
    Ces infos figurent comme attributs de la classe preIndexation.
    """
    def __init__(self):
        # Icons sizes management for pop-up windows (QMessageBox)
        self.pixmap = QPixmap("icons/Main_icon.png")
        self.pixmap = self.pixmap.scaled(100, 100)
        
        self.StackLoc, self.StackDir = gf.getFilePathDialog("série d'images à indexer (*.tiff)")
        checkimage = tf.TiffFile(self.StackLoc[0]).asarray() # Check for dimension. If 2 dimensions : 2D array. If 3 dimensions : stack of images
 
        if checkimage.ndim != 3: # Check if the data is not an image series
            self.popup_message("IniCHORD","Please import a stack of images.",'icons/Main_icon.png')
            return 
        else:
            self.Stack = tf.TiffFile(self.StackLoc[0]).asarray()

        filepath,  self.CifDir = gf.getFilePathDialog('CIF selection')
        self.CifLoc = filepath[0]
        
        filepath, self.DatabaseDir = gf.getFilePathDialog('theoretical test profiles (*.crddb)')
        self.DatabaseLoc = filepath[0]

    def popup_message(self,title,text,icon):
        msg = QMessageBox()
        msg.setIconPixmap(self.pixmap)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setWindowIcon(QtGui.QIcon(icon))
        msg.exec_()