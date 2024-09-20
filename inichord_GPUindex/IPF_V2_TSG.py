# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 08:37:15 2023

@author: glhote1
"""
' See the version IPF_V2 for IPF display outside the iniCHORD GUI'

import os
import os.path
import h5py
import numpy as np

import matplotlib.pyplot as plt
import diffpy

import Dans_Diffraction as da
import inichord_GPUindex.Xallo as xa
from inichord_GPUindex import Symetry as sy

from orix.crystal_map import CrystalMap, PhaseList
from orix.quaternion import Rotation, symmetry
from orix import plot as orixPlot
from orix.vector import Vector3d

import tkinter as tk
from tkinter import filedialog, Tk

from inichord_GPUindex import general_functions as gf

root = tk.Tk()       # initialisation du dialogue
root.withdraw()

#%% fonctions
def Display_IPF_GUI(CIFpath, quats, IPF_view = "X"):
    
    phases = []
    phases.append(diffpy.structure.loadStructure(CIFpath))
    
    crys = da.functions_crystallography.readcif(CIFpath)
    PhaseName = crys["_chemical_formula_sum"]
    numSG = crys["_space_group_IT_number"]
    PG = symmetry.get_point_group(int(numSG), True).name
        
    if IPF_view == "X":
        IPFim, xmap = IPF_Z_GUI(quats, PhaseName, PG, phases, Ipf_dir = Vector3d.xvector())
    elif IPF_view == "Y":
        IPFim, xmap = IPF_Z_GUI(quats, PhaseName, PG, phases, Ipf_dir = Vector3d.yvector())
    elif IPF_view == "Z":
        IPFim, xmap = IPF_Z_GUI(quats, PhaseName, PG, phases, Ipf_dir = Vector3d.zvector())

    return IPFim

def IPF_Z_GUI(quats, PhaseName, PG, phases, Ipf_dir = Vector3d.zvector()):
    
    array2 = np.rot90(quats, 1, (1, 3))
    QuatCorr = np.rot90(array2, 3, (1, 2))
    # maintenant on a la largeur selon l'axe 3, la hauteur selon l'axe 2, et les quaternions selon l'axe 4 et le score suivant axe 0
        
    width = len(QuatCorr[0, 0, :, 0]) # largeur selon l'axe 1
    height = len(QuatCorr[0, :, 0, 0]) # hauteur selon l'axe 0
     
    # importation de la stack au format orix
    page = np.zeros((1,height * width, 7))
    k = 0
    for a in range(1):
        for i in range(height):
            for j in range(width):
                page[a,k, 0] = 1
                page[a,k, 1] = i
                page[a,k, 2] = j
                page[a,k, 3] = QuatCorr[a,i, j, 0]
                page[a,k, 4] = QuatCorr[a,i, j, 1]
                page[a,k, 5] = QuatCorr[a,i, j, 2]
                page[a,k, 6] = QuatCorr[a,i, j, 3]
        
                k += 1
        k = 0
 
    # ré-importation
    phase_id = page[:,:, 0]
    y = page[:,:, 1]
    x = page[:,:, 2]
    quats = page[:,:, 3:]
    
    # conversion Quaternion -> axe-angle car Orix ne permet pas l'import de quaternion expérimentaux !
    axes = np.zeros((1,len(quats[0]), 3))
    angles = np.zeros((1,len(quats[0]),1))
    
    for i in range(len(quats[0])):
        a, b = xa.QuaternionToAxisAngle(quats[0,i, :])
        axes[0,i, :] = a
        angles[0,i] = b
    
    rotations = []
    
    axes_i = axes[0,:,:]
    angles_i = angles[0,:,0]
    rotations_i = Rotation.from_axes_angles(axes_i, angles_i, degrees= True)
    rotations.append(rotations_i)
    
    phase_list = PhaseList(
        names=[PhaseName],
        point_groups=[PG],
        structures=phases[0])
    
    # Create a CrystalMap instance
    xmap2 = []
    
    xmap2_i = CrystalMap(rotations=rotations[0],phase_id=phase_id[0,:],x=x,y=y,phase_list=phase_list)
    xmap2_i.scan_unit = "um"
    xmap2.append(xmap2_i)

    # information pour utiliser le code couleur spécifique au groupe ponctuel
    pg_laue = xmap2[0].phases[1].point_group.laue
    ipf_key = orixPlot.IPFColorKeyTSL(pg_laue)
    
    o_Cu = []

    Var_o_Cu = xmap2[0][PhaseName].orientations
    o_Cu.append(Var_o_Cu)
    
    # Orientation colors
    Ipf_dir = Ipf_dir
    ipf_key = orixPlot.IPFColorKeyTSL(pg_laue, direction=Ipf_dir)
    
    rgb=[]

    rgb_i = ipf_key.orientation2color(o_Cu[0])
    rgb.append(rgb_i)
        
    # rgb est l'image IPF sous forme d'un tableau numpy correctement orienté
    # rgb est flatten, il faut reshaper
    rgb = np.dstack(rgb)
    rgb = np.reshape(rgb,(height, width, 3,1))

    rgb = np.swapaxes(rgb, 0, 3)

    rgb = np.swapaxes(rgb, 2, 3)
    rgb = np.flip(rgb, 0)

    # Affichage map 
    rgb = rgb[0,:,:,:]
    rgb = np.flip(rgb,1)
    rgb = np.rot90(rgb)
    
    return rgb, xmap2
