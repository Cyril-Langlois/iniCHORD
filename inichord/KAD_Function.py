# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:05:45 2023

@author: glhote1
"""

## Ce programme à pour objectif de permettre de sortir rapidement un KAM de distance d'une stack alignée
'A noter, la valeur de la carte de KAD ne sera pas nécessairement comprise entre 0 et 1 sauf pour le kernel "Homogeneous".'

# Importation et fonctions
import numpy as np

def centeredEuclidianNorm(prof, ax = 1):
    mean =np.mean(prof, axis = ax, keepdims=True)
    prof = prof - mean
    del mean
    norm = np.linalg.norm(prof, axis = ax, keepdims=True)
    prof = prof / norm
    
    return prof

def KAM_rapide(rawImage_1): 
    
    Pr_length = len(rawImage_1)
    
    rawImage_roll1 = np.roll(rawImage_1,1, axis = 1)
    rawImage_roll2 = np.roll(rawImage_1,1, axis = 2)
    rawImage_roll3 = np.roll(rawImage_1,-1, axis = 1)
    rawImage_roll4 = np.roll(rawImage_1,-1, axis = 2)
    rawImage_roll5 = np.roll(rawImage_1,(1,1), axis = (1,2))
    rawImage_roll6 = np.roll(rawImage_1,(1,-1), axis = (1,2))
    rawImage_roll7 = np.roll(rawImage_1,(-1,1), axis = (1,2))
    rawImage_roll8 = np.roll(rawImage_1,(-1,-1), axis = (1,2))
    
    # Reshape des profils
    
    rawImage_reshape = np.reshape(np.swapaxes(rawImage_1,0,2),(-1,Pr_length))
    rawImage_roll1_reshape = np.reshape(np.swapaxes(rawImage_roll1,0,2),(-1,Pr_length)).T
    rawImage_roll2_reshape = np.reshape(np.swapaxes(rawImage_roll2,0,2),(-1,Pr_length)).T
    rawImage_roll3_reshape = np.reshape(np.swapaxes(rawImage_roll3,0,2),(-1,Pr_length)).T
    rawImage_roll4_reshape = np.reshape(np.swapaxes(rawImage_roll4,0,2),(-1,Pr_length)).T
    rawImage_roll5_reshape = np.reshape(np.swapaxes(rawImage_roll5,0,2),(-1,Pr_length)).T
    rawImage_roll6_reshape = np.reshape(np.swapaxes(rawImage_roll6,0,2),(-1,Pr_length)).T
    rawImage_roll7_reshape = np.reshape(np.swapaxes(rawImage_roll7,0,2),(-1,Pr_length)).T
    rawImage_roll8_reshape = np.reshape(np.swapaxes(rawImage_roll8,0,2),(-1,Pr_length)).T
    
    # Calcul de la distance entre la carte de rÃ©fÃ©rence et chaques cartes "rolls"
    
    Matmul1 = []
    Matmul2 = []
    Matmul3 = []
    Matmul4 = []
    Matmul5 = []
    Matmul6 = []
    Matmul7 = []
    Matmul8 = []
    
    for i in range(0,len(rawImage_reshape)):
        Var_1 = np.matmul(rawImage_reshape[i,:],rawImage_roll1_reshape[:,i])
        Var_2 = np.matmul(rawImage_reshape[i,:],rawImage_roll2_reshape[:,i])
        Var_3 = np.matmul(rawImage_reshape[i,:],rawImage_roll3_reshape[:,i])
        Var_4 = np.matmul(rawImage_reshape[i,:],rawImage_roll4_reshape[:,i])
        Var_5 = np.matmul(rawImage_reshape[i,:],rawImage_roll5_reshape[:,i])
        Var_6 = np.matmul(rawImage_reshape[i,:],rawImage_roll6_reshape[:,i])
        Var_7 = np.matmul(rawImage_reshape[i,:],rawImage_roll7_reshape[:,i])
        Var_8 = np.matmul(rawImage_reshape[i,:],rawImage_roll8_reshape[:,i])
        
        Matmul1.append(Var_1)
        Matmul2.append(Var_2)
        Matmul3.append(Var_3)
        Matmul4.append(Var_4)
        Matmul5.append(Var_5)
        Matmul6.append(Var_6)
        Matmul7.append(Var_7)
        Matmul8.append(Var_8)
    
    Matmul1 = np.vstack(Matmul1)
    Matmul2 = np.vstack(Matmul2)
    Matmul3 = np.vstack(Matmul3)
    Matmul4 = np.vstack(Matmul4)
    Matmul5 = np.vstack(Matmul5)
    Matmul6 = np.vstack(Matmul6)
    Matmul7 = np.vstack(Matmul7)
    Matmul8 = np.vstack(Matmul8)
    
    Matmul1 = np.reshape(Matmul1,(len(rawImage_1[0][0]),len(rawImage_1[0])))
    Matmul2 = np.reshape(Matmul2,(len(rawImage_1[0][0]),len(rawImage_1[0])))
    Matmul3 = np.reshape(Matmul3,(len(rawImage_1[0][0]),len(rawImage_1[0])))
    Matmul4 = np.reshape(Matmul4,(len(rawImage_1[0][0]),len(rawImage_1[0])))
    Matmul5 = np.reshape(Matmul5,(len(rawImage_1[0][0]),len(rawImage_1[0])))
    Matmul6 = np.reshape(Matmul6,(len(rawImage_1[0][0]),len(rawImage_1[0])))
    Matmul7 = np.reshape(Matmul7,(len(rawImage_1[0][0]),len(rawImage_1[0])))
    Matmul8 = np.reshape(Matmul8,(len(rawImage_1[0][0]),len(rawImage_1[0])))
    
    Distance_JDG = (Matmul1 + Matmul2 + Matmul3 + Matmul4 + Matmul5 + Matmul6 + Matmul7 + Matmul8)/(9)
    
    Distance_JDG = np.rot90(Distance_JDG, k=1, axes=(0, 1))
    Distance_JDG = np.flip(Distance_JDG)
    Distance_JDG = np.fliplr(Distance_JDG)
    Distance_JDG = 1-Distance_JDG
    
    Distance_JDG[1:-1,0] = Distance_JDG[1:-1,1] # Col gauche
    Distance_JDG[1:-1,-1] = Distance_JDG[1:-1,-2] # Col droite
    Distance_JDG[0,:] = Distance_JDG[1,:]
    Distance_JDG[-1,:] = Distance_JDG[-2,:]
  
    del(Var_1,Var_2,Var_3,Var_4,Var_5,Var_6,Var_7,Var_8)
    del(rawImage_roll1,rawImage_roll2,rawImage_roll3,rawImage_roll4,rawImage_roll5,rawImage_roll6,rawImage_roll7,rawImage_roll8)
    del(rawImage_roll1_reshape,rawImage_roll2_reshape,rawImage_roll3_reshape,rawImage_roll4_reshape,rawImage_roll5_reshape,rawImage_roll6_reshape,rawImage_roll7_reshape,rawImage_roll8_reshape)
    del(Matmul1,Matmul2,Matmul3,Matmul4,Matmul5,Matmul6,Matmul7,Matmul8)

    return Distance_JDG

def Divided_KAD(rawImage_1, divider = 20):
    Conca_KAD = []
    list_i = []
    
    height = len(rawImage_1[0]) # Hauteur d'une image
    rec = 3 # Nombre de pixels pour le recouvrement

    for i in range(1,divider):
        if height%i > rec+1 and height/i < 1000: # Critères : stacks de moins de 1000 pixels de haut, et dernière stack avec suffisament de pixel (rec+1)
            list_i.append(i) # Extraction dans la liste des diviseurs ceux qui sont possibles
            
    min_i = np.min(list_i) # Selection du plus petit possible pour diviser le moins possible
    h_prime = height//min_i # Hauteur des stacks

    for i in range(0,min_i):       
        var = rawImage_1[:,h_prime*i:h_prime*(i+1)+rec,:] # Extraction de chaque sous regions
        var_distance = KAM_rapide(var) # Calcul du KAD sur chaque sous regions

        Conca_KAD.append(var_distance) # Stockage dans une liste des différentes cartes
      
    var = rawImage_1[:,h_prime*min_i:,:] # Extraction de la dernière sous regions
    var_distance = KAM_rapide(var) # Calcul du KAD sur la dernière portion du stack
        
    Conca_KAD.append(var_distance) # Ajout du dernier KAD
    lim = int(rec/2)
    
    xtract = np.zeros((len(rawImage_1[0]),len(rawImage_1[0][0]))) # Creation du tableau de concatenation des KAD

    # Remplissage de la fin du tableau
    End_shape = Conca_KAD[-1].shape
    xtract[-End_shape[0]:,:] = Conca_KAD[-1]

    # Remplissage du tableau en partant de la fin 
    for i in range(min_i,0,-1):
        xtract[(i-1)*h_prime :i*h_prime + lim+1] = Conca_KAD[i-1][:-lim]
        
    return xtract
