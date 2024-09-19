# -*- coding: utf-8 -*-
import math
import numpy as np
from pyquaternion import Quaternion

#____________________________________________________
# Calculs dans les cristaux
#____________________________________________________
	
def MetricTensor(a, b, c, alpha, beta, gamma): 
    """
    entrée des longueurs en Angtröms, entrée des angles en degrés
    
    Parameters:
        a (float): angle alpha
        b (float): angle beta
    
    """

    metricTensor = np.matrix(
    [[a * a, a * b * cos(toRad(gamma)), a * c * cos(toRad(beta))],
    [a * b * cos(toRad(gamma)), b * b, b * c * cos(toRad(alpha))],
    [a * c * cos(toRad(beta)), b * c * cos(toRad(alpha)), c * c]])

    # exemple d'utilisation :
    # Calcul d'un produit scalaire dans un cristal non orthonormée entre V1 et V2 :
    # V1.V2 = (V1.T * M * V2) avec .T la transposée du vecteur colonne, qui est
    # donc un vecteur horizontal
    
    return metricTensor

def Pc2o(a, b, c, alpha, beta, gamma): 
    # entrée des longueurs en Angtröms, entrée des angles en degrés
    # Retourne la matrice qui permet d'avoir les coordonnées d'un vecteur du cristal (c)
    # dans le repère orthonormé associé (o)
    # Le repère orthonorme est ici considere comme x // a ; z // c* et y // z*x
	# Le résultat est donc une matrice passive.
    
    Volume = (np.linalg.det(MetricTensor(a, b, c, alpha, beta, gamma)))**0.5
    # print("Volume de la maille : ", Volume)
    paramF = cos(toRad(beta)) * cos(toRad(gamma)) - cos(toRad(alpha))
    
    matrix = np.matrix(
    [[a, b * cos(toRad(gamma)), c * cos(toRad(beta))],
    [0.0, b * sin(toRad(gamma)), - c * paramF / sin(toRad(gamma))],
    [0.0, 0.0, Volume / a / b / sin(toRad(gamma))]])
    
    return matrix

def Po2c(a, b, c, alpha, beta, gamma):
    # entrée des longueurs en Angtröms, entrée des angles en degrés
    # Retourne la matrice qui permet d'avoir les coordonnées d'un vecteur du cristal (c)
    # dans le repère orthonormé associé (o). Le résultat est donc une matrice passive.
    # le calcul se fait tout simplement en inversant la matrice crystal-to-orthonormé
    # Le repère orthonorme est ici considere comme x // a ; z // c* et y // z*x
    
    return np.linalg.inv(Pc2o(a, b, c, alpha, beta, gamma))
 
#____________________________________________________
# Conversions entre représentations
#____________________________________________________


# Notes : La chaîne de conversion testée est dans cet ordre :
# - axisAngle <-> quaternion (cette conversion est universellement reconnue - une seule manière de faire)

# Ensuite, pour les quaternions
# Si l'on veut appliquer un quaternion sur un vecteur, on peut le faire de façon active :
#   p1' = q * p1 * q-1
# ou de façon passive :
#   p2 = q-1 * p1 * q
#  (ici, les indices 1 et 2 dénotent le référenciel 1 ou 2 dans lequel les coordonnées du vecteur sont obtenues)
# Dans le cas actif, le vecteur p est "prime" car il a bougé au sein du repère 1
# Dans le cas passif, le vecteur n'est pas "prime" car il n'a pas bougé ; par contre il y a un indice "2" car
# les coordonnées obtenues sont dans le référentiel 2 (obtenu par rotation du référentiel 1)

# Pour convertir le quaternion en matrice, il est convenu ici de transformer vers une matrice passive.
# Pour la transformation inverse, on passe d'une matrice passive à un quaternion dit "actif",
# c'est-à-dire qu'il faut employer la formule p2 = q-1 * p1 * q pour obtenir les coordonnées d'un vecteur dans le repère 2.


#-------------- conversions "aller"

def axisAngle2quaternion(axisAngle):

	# la conversion est celle des mathématiciens, qui correspond à celle de De Graef.
	# seul cas de figure spécial : si l'angle est nul
	# auquel cas le quaternion sera 1, 0, 0, 0
	# l'angle est entré en degrés
	# l'angle est en dernier dans la liste de quatre composante fournie en entrée
	
	# IMPORTANT : la librairie pyQuaternion normalise le quaternion, ce qui est très bien
	# IMPORTANT : l'angle de la rotation est compris entre 0° et 180° par convention

    axisAngle = axisAngleNormalize(axisAngle)
    
    if not (0.0 <= axisAngle[3] and axisAngle[3] <= 180.0): raise ValueError('the angle must be between 0.0 and 180.0 by convention')
	
    quat = Quaternion(axis = axisAngle[0:3], degrees = axisAngle [3])		

	# # calcul à la main, identique au calcul de pyQuaternion
	
	# c = cos (toRad(axisAngle[3]) / 2.0)
	# s = sin(toRad(axisAngle[3]) / 2.0)

	# axNorm = np.sqrt(axisAngle[0]**2 + axisAngle[1]**2 + axisAngle[2]**2)
	# quat2 = [c, axisAngle[0]*s / axNorm, axisAngle[1]*s / axNorm, axisAngle[2]*s / axNorm] 
    
    if close_enough(axisAngle[3], 0.0): 
        quat = Quaternion(scalar=1.0, vector=(0.0, 0.0, 0.0))

    return quat

def axisAngle2quaternion_nonorm(axisAngle):

	# la conversion est celle des mathématiciens, qui correspond à celle de De Graef.
	# seul cas de figure spécial : si l'angle est nul
	# auquel cas le quaternion sera 1, 0, 0, 0
	# l'angle est entré en degrés
	# l'angle est en dernier dans la liste de quatre composante fournie en entrée
	
	# IMPORTANT : la librairie pyQuaternion normalise le quaternion, ce qui est très bien
	# IMPORTANT : l'angle de la rotation est compris entre 0° et 180° par convention

    # axisAngle = axisAngleNormalize(axisAngle)
    
    if not (0.0 <= axisAngle[3] and axisAngle[3] <= 180.0): raise ValueError('the angle must be between 0.0 and 180.0 by convention')
	
    quat = Quaternion(axis = axisAngle[0:3], degrees = axisAngle [3])		

	# # calcul à la main, identique au calcul de pyQuaternion
	
	# c = cos (toRad(axisAngle[3]) / 2.0)
	# s = sin(toRad(axisAngle[3]) / 2.0)

	# axNorm = np.sqrt(axisAngle[0]**2 + axisAngle[1]**2 + axisAngle[2]**2)
	# quat2 = [c, axisAngle[0]*s / axNorm, axisAngle[1]*s / axNorm, axisAngle[2]*s / axNorm] 
    
    if close_enough(axisAngle[3], 0.0): 
        quat = Quaternion(scalar=1.0, vector=(0.0, 0.0, 0.0))

    return quat

def Quat2Euler(Quat):
    # unit test 09-09-2020
    # ici, le quaternion est donné sous forme de liste (ou numpy array) des quatre composantes
    # la fonction est issue des codes de M. De Graef, donc 
    # Il n'y a pas de symétrie qui joue ici.
    # attention ! routine modifiée pour éviter le concept de quaternion passif : axe = - axe par rapport à la fonction 
    # utilisée par De Graef
    
    # attention : pas de normalisation du quaternion pour l'instant

    Euler = np.zeros((3))
    epsijk = 1.0
    qq = Quat
    
    qq = -qq    # ces deux opérations sont pour transcrire le quaternion dans une forme mathématique classique.
    qq[0] = -qq[0] # Cela évite d'utiliser la notion de quaternion passif.
    
    q03 = qq[0]*qq[0] + qq[3]*qq[3]
    q12 = qq[1]*qq[1] + qq[2]*qq[2]
    chi = math.sqrt(q03*q12)

    if close_enough(chi, 0.0):
        if close_enough(q12,0.0):
            phi1 = math.atan2(-epsijk*2.0*qq[0]*qq[3], qq[0]*qq[0]-qq[3]*qq[3])
            Phi = 0.0
            phi2 = 0.0
        else:
            phi1 = math.atan2(2.0 * qq[1] * qq[2], qq[1] * qq[1] - qq[2] * qq[2])
            Phi = math.pi
            phi2 = 0.0
    else:
        chi = 1.0 / chi
        phi1 = math.atan2((-epsijk * qq[0] * qq[2] + qq[1]*qq[3])*chi, (-epsijk * qq[0] * qq[1] - qq[2]*qq[3])*chi)
        Phi = math.atan2(2.0/chi, q03-q12) 
        phi2 = math.atan2((epsijk * qq[0] * qq[2] + qq[1]*qq[3])*chi, (-epsijk * qq[0] * qq[1] + qq[2]*qq[3])*chi)

    Euler[0] = phi1
    Euler[1] = Phi
    Euler[2] = phi2

    if Euler[0] < 0.0: Euler[0] = math.fmod(Euler[0] + 100*math.pi, 2*math.pi)
    if Euler[1] < 0.0: Euler[1] = math.fmod(Euler[1] + 100*math.pi, math.pi)
    if Euler[2] < 0.0: Euler[2] = math.fmod(Euler[2] + 100*math.pi, 2*math.pi)

    # Conversion en degrés

    Euler[0] = toDeg(Euler[0])
    Euler[1] = toDeg(Euler[1])
    Euler[2] = toDeg(Euler[2])

    return Euler


def EulerToMatrix(AnglesEuler):
    # unit test 09-09-2020
    # fonction qui retourne la matrice de changement de repere construite a partir des angles d'Euler fournis en degrés
    # note : c'est la "vraie" matrice, sans le decalage angulaire propre a la representation sur la projection stereographique
    # on obtient une matrice passive. Pour la version active, prendre la transposée (ou l'inverse, c'est pareil)
	# c'est la matrice écrite dans la documentation de EMsoft, et dans le fichier rotations.f90


    phi1 = AnglesEuler[0]  # le tableau AnglesEuler contient en ligne les angles d'Euler en degrés, qui sont redistribues sur les variables phi1, PHI, phi2 pour plus de lisibilite
    Phi = AnglesEuler[1]
    phi2 = AnglesEuler[2]

    phi1 = toRad(phi1)  # passage en radians
    Phi = toRad(Phi)
    phi2 = toRad(phi2)

    a_11 = cos(phi1) * cos(phi2) - sin(phi1) * sin(phi2) * cos(Phi)
    a_12 = sin(phi1) * cos(phi2) + cos(phi1) * sin(phi2) * cos(Phi)
    a_13 = sin(phi2) * sin(Phi)
    a_21 = -cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(Phi)
    a_22 = -sin(phi1) * sin(phi2) + cos(phi1) * cos(phi2) * cos(Phi)
    a_23 = cos(phi2) * sin(Phi)
    a_31 = sin(phi1) * sin(Phi)
    a_32 = -cos(phi1) * sin(Phi)
    a_33 = cos(Phi)

    matrix = np.array([[a_11, a_12, a_13], [a_21, a_22, a_23], [a_31, a_32, a_33]])

    return matrix	

#-------------- conversions "retour"   

def OrientationMatrix2euler(om):
    # unit test 09-09-2020
    Euler = np.zeros((3))

    if not close_enough(abs(om[2,2]),1.0):
        Euler[1] = np.arccos(om[2,2])
        zeta = 1.0/ math.sqrt(1-om[2,2]*om[2,2])
        Euler[0] = np.arctan2(om[2,0]*zeta, -om[2,1]*zeta)
        Euler[2] = np.arctan2(om[0,2]*zeta, om[1,2]*zeta)
    else:
        if close_enough(om[2,2], 1.0):
            Euler[0] = np.arctan2(om[0,1], om[0,0])
            Euler[1] = 0.0
            Euler[2] = 0.0
        else:
            Euler[0] = -1.0 * np.arctan2(-om[0,1], om[0,0])
            Euler[1] = math.pi
            Euler[2] = 0.0

    if Euler[0] < 0.0: Euler[0] = math.fmod(Euler[0] + 100*math.pi, 2*math.pi)
    if Euler[1] < 0.0: Euler[1] = math.fmod(Euler[1] + 100*math.pi, math.pi)
    if Euler[2] < 0.0: Euler[2] = math.fmod(Euler[2] + 100*math.pi, 2*math.pi)

    # # attention : la reduction ci-dessous est valable dans le cubic pour coller avec Oxford.
    # # dans le cas général, on est modulo 2pi sur phi1, modulo pi sur Phi, et modulo 2pi sur phi2

    # if Euler[0] < 0.0: Euler[0] = math.fmod(Euler[0] + 100*math.pi, 2*math.pi)
    # if Euler[1] < 0.0 or Euler[1] > 0.5*math.pi: Euler[1] = math.fmod(Euler[1] + 100*math.pi, 0.5*math.pi)
    # if Euler[2] < 0.0 or Euler[2] > 0.5*math.pi: Euler[2] = math.fmod(Euler[2] + 100*math.pi, 0.5*math.pi)

    # Conversion en degrés

    Euler[0] = toDeg(Euler[0])
    Euler[1] = toDeg(Euler[1])
    Euler[2] = toDeg(Euler[2])

    return Euler

def EulerToQuat(AnglesEuler):
    # unit test 09-09-2020
    Quat = np.zeros((4))
    phi1 = AnglesEuler[0]  # le tableau AnglesEuler contient en ligne les angles d'Euler en degrés, qui sont redistribues sur les variables phi1, PHI, phi2 pour plus de lisibilite
    Phi = AnglesEuler[1]
    phi2 = AnglesEuler[2]

    epsijk = 1.0 # convention classique Morawiec pour la multiplication de quaternions

    phi1 = toRad(phi1)  # passage en radians
    Phi = toRad(Phi)
    phi2 = toRad(phi2)

    cPhi = cos(Phi / 2.0)
    sPhi = sin(Phi / 2.0)
    cm = cos(0.5 * (phi1 - phi2))
    sm = sin(0.5 * (phi1 - phi2))
    cp = cos(0.5 * (phi1 + phi2))
    sp = sin(0.5 * (phi1 + phi2))

    # # passive quaternion ; si on veut travailler avec les rotations actives, il faut prendre le conjugué du quaternion (voir Rowenhorst 2015)
    # Quat[0] = cPhi*cp
    # Quat[1] = -epsijk*sPhi*cm
    # Quat[2] = -epsijk*sPhi*sm
    # Quat[3] = -epsijk*cPhi*sp
  
    
    # active quaternion 
    Quat[0] = cPhi*cp
    Quat[1] = epsijk*sPhi*cm
    Quat[2] = epsijk*sPhi*sm
    Quat[3] = epsijk*cPhi*sp


    # first component must be positive
    if Quat[0] < 0.0: Quat = -Quat

    return Quat

def QuaternionToAxisAngle(quat):
    quat1 = Quaternion(quat)
    return quat1.axis, quat1.degrees

# ------------ sauts de chaîne

def QuaternionToMatrix(quat):
    quat1 = Quaternion(quat)
    return quat1.rotation_matrix.T # la transposée est ici nécessaire pour avoir une matrice passive

def OrientationMatrix2Quat(om):
    quat = Quaternion(matrix = om.T)
    if quat.scalar < 0.0: quat = -quat
    return quat

def axisAngle2OrientationMatrix(axisAngle):
    # unit test 2020-09-17
    # la fonction prend en entrée une liste de quatre éléments avec axe(3) puis l'angle(1) en degrés
    # le résultat est une matrice passive  
    quat =Quaternion(axis = axisAngle[:3], degrees = axisAngle[3])
    return quat.rotation_matrix.T

def OrientationMatrixToAxisAngle(om):
    # prend en entrée une matrice passive
    # retourne un quaternion actif
    # donc, si on veut obtenir les coordonnées d'un vecteur p1
    # dans un repère tourné, il faut utiliser la formule
    #   p2 = quat-1 * p1 * quat
    #  (ici, les indices 1 et 2 dénotent le référenciel 1 ou 2 dans lequel les coordonnées du vecteur sont obtenues)
    
    quat = Quaternion(matrix = om.T)
    return quat.axis, quat.degrees

def axisAngle2Euler(axisAngle):
    return Quat2Euler(axisAngle2quaternion(axisAngle))

def Euler2axisAngle(AnglesEuler):
    axisAngle = np.zeros(4)
    Quat = Quaternion(EulerToQuat(AnglesEuler))
    axisAngle[0:3] = Quat.axis
    axisAngle[3] = Quat.degrees 
    return axisAngle

#____________________________________________________
# Calculs de désorientation
#____________________________________________________

# ---------------  sans prise en compte des symétries

def disOfromQuatNoSym(quat1, quat2):
    # validé avec MTEX (exactement les mêmes résultats ; cela valide cette routine et également QuaternionToMatrix
    matrix01 = QuaternionToMatrix(quat1)
    matrix02 = QuaternionToMatrix(quat2)
    disOrientationMatrix = np.dot(matrix01, matrix02.T)
    # Warning : que l'on inverse l'ordre des matrices, ou la position de la
    # transposition, l'angle est le même, mais l'axe est renversé quand on permute
    # 
    # Interprétation de np.dot(matrix01, matrix02.T): si on part d'un vecteur 
    # définit dans le repère du cristal 2,alors on revient dans le repère de 
    # référence avec matrice02.T. Puis on repart dans le repère du cristal 01 
    # avec matrice01. La matrice np.dot(matrix01, matrix02.T) prend alors 
    # en entrée un vecteur dans le cristal 02 et nous donne ses coordonnées dans le cristal 01 

    disOh = 0.0  # initialisation du vecteur de rotation autour duquel la desorientation se fait
    disOk = 0.0
    disOl = 0.0

    misOrientation = np.arccos((np.trace(disOrientationMatrix) - 1.0) / 2.0)
    disOh = (disOrientationMatrix[1, 2] - disOrientationMatrix[2, 1]) / 2.0 / sin(misOrientation)  # les coordonnees du vecteur portant la rotation sont extraites de la matrice
    disOk = (disOrientationMatrix[2, 0] - disOrientationMatrix[0, 2]) / 2.0 / sin(misOrientation)
    disOl = (disOrientationMatrix[0, 1] - disOrientationMatrix[1, 0]) / 2.0 / sin(misOrientation)
    misOrientation = toDeg(misOrientation)

    return disOh, disOk, disOl, misOrientation


def disOfromQuatNoSymNoMat(quat1, quat2):
	quat1 = Quaternion(quat1)
	quat2 = Quaternion(quat2)
    # prend en entrée une liste de 4 valeurs avec le scalaire en premier
    # quat1 et quat2 sont des objets de la librairie pyQuaternions
 
	# Note sur le sens de multiplication :
	# Le calcul est compréhensible avec les matrices, comme dans la routine ci-dessus
	# avec cette interprétation :
    # Interprétation de np.dot(matrix01, matrix02.T): si on part d'un vecteur 
    # définit dans le repère du cristal 2,alors on revient dans le repère de 
    # référence avec matrice02.T. Puis on repart dans le repère du cristal 01 
    # avec matrice01. La matrice np.dot(matrix01, matrix02.T) prend alors 
    # en entrée un vecteur dans le cristal 02 et nous donne ses coordonnées dans le cristal 01
    # pour les quaternions, on applique la même logique en utilisant la façon passive d'appliquer
    # un quaternion à un vecteur
	
 
	disOQuat = quat2.inverse *  quat1 

	# ATTENTION : la librairie pyQuaternion renvoie un angle dans le domaine [-pi ; pi], et donc un angle pas forcément positif.
	# De plus, la convention de De Graef dit que les angles sont entre 0 et Pi, 
	# Dans ce cas, on inverse l'axe et l'angle vaut alors 2pi - angle.
	# Application de cette convention :
	
	omega = disOQuat.degrees
	
	axis = disOQuat.axis
	
	# dans le cas d'un angle entre -pi et 0, on renvoie dans l'intervalle [pi, 2pi] et on revient entre 0 et Pi
	# voir commentaire précédent  
	# if omega < 0: omega = omega + 360 
    # if omega > 180: 
    #     omega = 360 - omega
    #     axis = - axis

	if omega < 0: 
# 		print("angle négatif")
		omega = - omega # démarche équivalente
		axis = - axis
    
	return axis, omega


# ---------------  avec prise en compte des symétries

def disOfromQuatSymMat(quat1, quat2, listSymM):

	# en entrée, les deux quaternions sont exprimés sous forme de liste de 4 composantes
    # listSymM est une liste dont chaque composant est une matrice Numpy 3x3
	# pour pouvoir utiliser QuaternionToMatrix

    # attention ! Cette fonction ne sert que de comparatif au travail sur les quaternions
    # et elle ne fait jouer les ysmétries que d'un côté  seulement (sur un seul des deux grains)
    # Il peut y avoir plusieurs désorientations différentes qui ont le même angle mais des axes différents
    # mais équivalents par symétrie les uns aux autres.

    matrix01 = QuaternionToMatrix(quat1)
    matrix02 = QuaternionToMatrix(quat2)
    
    disOrientationMatrix = np.dot(matrix01, matrix02.T)

    disOh = 0.0  # initialisation du vecteur de rotation autour duquel la desorientation se fait
    disOk = 0.0
    disOl = 0.0

    disOrientation = 7000

    for m in range(len(listSymM)):
            LeftEquivalentO = np.dot(listSymM[m], disOrientationMatrix)  # on multiplie la matrice de desorientation par chacun des operateurs de symmetrie
 
            misOrientation = np.arccos((np.trace(LeftEquivalentO) - 1.0) / 2.0)  # l'angle de desorientation est calcule a partir de la trace de la matrice precedente

            componentH = (LeftEquivalentO[1, 2] - LeftEquivalentO[2, 1]) / 2.0 / sin(misOrientation)  # les coordonnees du vecteur portant la rotation sont extraites de la matrice
            componentK = (LeftEquivalentO[2, 0] - LeftEquivalentO[0, 2]) / 2.0 / sin(misOrientation)
            componentL = (LeftEquivalentO[0, 1] - LeftEquivalentO[1, 0]) / 2.0 / sin(misOrientation)

            if misOrientation < disOrientation:  # boucle pour verifier si l'angle de rotation trouve pour cet operateur est plus petit que le precedent
                disOrientation = misOrientation  # auquel cas on met a jour l'angle de desorientation et les composante sdu vecteur portant la rotation
                disOh = componentH
                disOk = componentK
                disOl = componentL
                final = LeftEquivalentO
                    
    disOrientation = toDeg(disOrientation)

    return disOh, disOk, disOl, disOrientation	
 
def disOfromQuatSymNoMat(quat1, quat2, listSymQ):

    # Cette fonction calcule la désorientation entre deux orientations représentées
    # par quat1 et quat2, en faisant jouer les symétries des deux cristaux, qui sont de même groupe propre
    # La fonction ne traite pas (encore) le cas où les opérations de symétrie ne sont pas les mêmes
    # pour les deux grains.
    # Cette fonction ne doit pas être utilisée pour calculer les variants d'orientation d'un seul quaternion
    # car on peut faire plus simple pour cela.
    # la solution renvoyée est la désorientation de plus petit angle dont l'axe est dans le triangle standard
    # du groupe propre en question.
    # La routine prend en compte les symétries des deux côtés (A et B) ainsi que la "switching symetry"
    # donc l'équivalence entre A -> B ou B -> A.
	# prend en entrée les quaternions obtenus par la bibliothèque pyquaternions
    # pour la liste des opérations de symétrie, il faut l'avoir aussi mise en format pyquaternions. 
	# On a donc les opérations de symétrie exprimée dans le repère ortho du cristal
    
    # quat1 = Quaternion(array = quat1)
    # quat2 = Quaternion(array = quat2)

    EquivalentO = Quaternion(array = np.zeros(4))
    resultO = Quaternion(array = np.zeros(4))
    
    if  type(quat1) != type(resultO):
        quat1 = Quaternion(quat1)
        quat2 = Quaternion(quat2)
    
    
    disOrientation = 7000.0

    ori = quat2.inverse * quat1 # 2024-06-13 : A checker : pourquoi on est obligé d'inverser les quaternions de Seb pour que ça marche ?

    for m in listSymQ:
        
        EquivalentO =  m * ori             # que l'on prenne m ou m.inverse
                                           # n'a pas d'importance car les deux figurent dans listSymM
        # la valeur absolue prend le rôle de la switching symetry entre A -> B ou B -> A
        misOrientation =  abs(EquivalentO.degrees) 
        axis = EquivalentO.axis
        
        # u = axis[0]
        # v = axis[1]
        # w = axis[2]
        
        # if (((0.0 <= w) and (w<=v) and (v <= u)) or ((0.0 <= -w) and (-w<=-v) and (-v <= -u))):
        #     if misOrientation <= disOrientation: 
        #         resultO = EquivalentO
        #         disOrientation = misOrientation

        if misOrientation <= disOrientation: 
            resultO = EquivalentO
            disOrientation = misOrientation
            # print(resultO.degrees, resultO.axis)
                
    omega = resultO.degrees
    axis = resultO.axis

# vérification des signes :
# on a le droit de changer omega ou l'axe car
# on peut prendre aussi bien A -> B que B -> A
# pour la désorientation
  
    # if axis[0] < 0:
    #     axis = - axis        

    omega = abs(omega)

    return axis, omega
    
def grainDisOrientation(quat1, quat2, listSymM):

    # Cette fonction calcule la désorientation entre deux orientations représentées
    # par quat1 et quat2, en faisant jouer les symétries des deux cristaux, qui sont de même groupe propre
    # La fonction ne traite pas (encore) le cas où les opérations de symétrie ne sont pas les mêmes
    # pour les deux grains.
    # Cette fonction ne doit pas être utilisée pour calculer les variants d'orientation d'un seul quaternion
    # car on peut faire plus simple pour cela.
    # la solution renvoyée est la désorientation de plus petit angle dont l'axe est dans le triangle standard
    # du groupe propre en question.
    # La routine prend en compte les symétries des deux côtés (A et B) ainsi que la "switching symetry"
    # donc l'équivalence entre A -> B ou B -> A.
	# prend en entrée les quaternions obtenus par la bibliothèque pyquaternions
    # pour la liste des opérations de symétrie, il faut l'avoir aussi mise en format pyquaternions. 
	# On a donc les opérations de symétrie exprimée dans le repère ortho du cristal
    
    # quat1 = Quaternion(array = quat1)
    # quat2 = Quaternion(array = quat2)
    
    EquivalentO = Quaternion(array = np.zeros(4))
    resultO = Quaternion(array = np.zeros(4))
    
    disOrientation = 7000.0

    ori = quat2.inverse * quat1

    for m in listSymM:
        for n in listSymM:
            EquivalentO =  m.inverse * ori * n # que l'on prenne m ou m.inverse
                                               # n'a pas d'importance car les deux figurent dans listSymM
            # la valeur absolue prend le rôle de la switching symetry entre A -> B ou B -> A
            misOrientation =  abs(EquivalentO.degrees) 
            axis = EquivalentO.axis
            u = axis[0]
            v = axis[1]
            w = axis[2]
            
            # if (((0.0 <= w) and (w<=v) and (v <= u)) or ((0.0 <= -w) and (-w<=-v) and (-v <= -u))):
            #     if misOrientation <= disOrientation: 
            #         resultO = EquivalentO
            #         disOrientation = misOrientation

            if misOrientation <= disOrientation: 
                resultO = EquivalentO
                disOrientation = misOrientation
                # print(resultO.degrees, resultO.axis)
                
    omega = resultO.degrees
    axis = resultO.axis

# vérification des signes :
# on a le droit de changer omega ou l'axe car
# on peut prendre aussi bien A -> B que B -> A
# pour la désorientation
  
    # if axis[0] < 0:
    #     axis = - axis        

    omega = abs(omega)

    return axis, omega


#____________________________________________________
# Outils pour les quaternions
#____________________________________________________

def vect2pyQuat(vec):
    # vec est un tableau numpy de trois éléments
    vecteur = np.zeros(4)
    vecteur[1:] = vec
    vecteur[0] = 0.0
    
    return Quaternion(array=vecteur)

def ActiveProduct(quat, vec):
    # quat doit être un objet pyQuaternion normalisé
    # vec est un tableau numpy de trois éléments
    Qvec = vect2pyQuat(vec)
    produit = quat*Qvec*quat.inverse
    if quat[0] < 0.0: produit = - produit

    return produit.imaginary

def PassiveProduct(quat, vec):
    # quat doit être un objet pyQuaternion normalisé
    # vec est un tableau numpy de trois éléments
    Qvec = vect2pyQuat(vec)
    produit = quat.inverse * Qvec * quat
    if quat[0] < 0.0: produit = - produit

    return produit.imaginary  

def listQuat2QuatObj(listQuat):

    # Cette routine convertit une liste de quaternions exprimés en 4 composantes
	# en une liste d'objet Quaternions normalisés au sens de la librairie pyQuaternion.

    list = []
    for i in listQuat:
        list.append(Quaternion(i).unit)
    return list

def Quat2RFZquat(quat, listSymQ):

    # routine de conversion donnant des résultats cohérents avec EMeqvrt
    # Note : si epsijk est réglé sur -1 dans EMsoft, on obtient le quaternion conjugué
	# Cette routine a pour but de renvoyer un quaternion donné en entrée vers la zone
	# fondamentale du cristal concerné
	# .
	# en entrée : un objet pyQuaternion et la liste des opérations de symétrie du cristal
	# exprimée en objet pyQuaternions dans le repère orthonormé du cristal.

    disOrientation = 7000
    RFZquat = Quaternion()  
    
    for m in listSymQ:

        # EquivalentO =   quat * m # on multiplie le quaternion qu'on veut passer en RFZ par chacun des opérateurs quaternions
        EquivalentO =   m * quat # note : la multiplication à gauche permet d'obtenir le même résultat que la routine EMsoft ConvertOrientation avec passage en RFZ
        misOrientation =  abs(EquivalentO.angle)
        
        if misOrientation < disOrientation:  # if pour verifier si l'angle de rotation trouve pour cet operateur est plus petit que le precedent
            disOrientation = misOrientation  # auquel cas on met a jour l'angle de desorientation et les composante sdu vecteur portant la rotation
            RFZquat = EquivalentO  
            if RFZquat[0] < 0: RFZquat = -RFZquat # on s'assure que le quaternion a une partie scalaire positive
            
    RFZquat = RFZquat.unit # des fois que ça ne soit pas fait en entrée, on normalise le quaternion qui sera renvoyé
    
    return RFZquat

#____________________________________________________
# Outils pour la projection stéréographique	
#____________________________________________________
	
def EulerToMatrixStereo(AnglesEuler):
    # le calcul de cette matrice est dediee a  l'affichage dans la projection
    # NE PAS UTILISER POUR DES CALCULS FORMELS !!!
    # la fonction accepte un tableau numpy avec une seule ligne et les trois angles Euler en degres le long de la ligne

    phi1 = AnglesEuler[0]
    Phi = AnglesEuler[1]
    phi2 = AnglesEuler[2]
    
    # modification propre a  l'affichage de la projection stereo
    # On rajoute 90 degres pour avoir l'axe X oriente vers la droite
    phi1 = 90 + phi1  

    phi1 = toRad(phi1)
    Phi = toRad(Phi)
    phi2 = toRad(phi2)

    a_11 = cos(phi1) * cos(phi2) - sin(phi1) * sin(phi2) * cos(Phi)
    a_21 = sin(phi1) * cos(phi2) + cos(phi1) * sin(phi2) * cos(Phi)
    a_31 = sin(phi2) * sin(Phi)
    a_12 = -cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(Phi)
    a_22 = -sin(phi1) * sin(phi2) + cos(phi1) * cos(phi2) * cos(Phi)
    a_32 = cos(phi2) * sin(Phi)
    a_13 = sin(phi1) * sin(Phi)
    a_23 = -cos(phi1) * sin(Phi)
    a_33 = cos(Phi)

    matrix = np.array([[a_11, a_12, a_13], [a_21, a_22, a_23], [a_31, a_32, a_33]])

    return matrix
	
def Cart2Sphe(Cartesian):   
	# fonction qui convertit les coordonnees cartesiennes d'un vecteur en ses coordonnees spheriques
	# les angles résultants sont exprimés en radians.
	
    X = Cartesian[0]     
    Y = Cartesian[1]
    Z = Cartesian[2]

    theta = 0.0

    rho = (X ** 2 + Y ** 2 + Z ** 2) ** 0.5
    theta = np.arccos(Z / rho)
    phi = np.arccos(X / (X ** 2 + Y ** 2) ** 0.5)
    if Y <= 0: phi = 2 * math.pi - np.arccos(X / (X ** 2 + Y ** 2) ** 0.5)

    Spherical = np.array([[rho], [theta], [phi]])

    return Spherical	
	
def Sphe2Proj(Spherical):  # fonction qui convertit les coordonnées spheriques d'un vecteur en ses coordonnees dans la projection stereo

    strike = 270 - toDeg(Spherical[2, 0])  # ATTENTION : ce "270-" n'est là que pour coller aux angles predefini dans la projection stereo
    dip = toDeg(Spherical[1, 0])

    return strike, dip

#____________________________________________________
# Outils divers
#____________________________________________________

def operationsPropresList(listOpGE):
    # listOpGE est une liste qui contient des tableaux Numpy 3x4 correspondant
    # aux opérations de symétrie. Attention : il y a la partie translatoire aussi !!
    
    OpPropres = []
    # élimination des rotations impropres
    for i in range(len(listOpGE)):
        if np.linalg.det(listOpGE[i][:, :3]) > 0.0: OpPropres.append(listOpGE[i][:, :3].flat)
        
    # élimination des doublons    
    tupled_lst = set(map(tuple, OpPropres))
    tupled_lst = list(tupled_lst)
    lst2 = []
    for i in range(len(tupled_lst)):
        lst2.append(np.asarray(tupled_lst[i]).reshape((3, 3)))
    
    return lst2

def cos(a):
	# entrer l'angle en radians
    b = np.cos(a)
    return b

def sin(a):
	# entrer l'angle en radians
    b = np.sin(a)
    return b

def toRad(e):
    f = float(e)/360.0*2.0*math.pi
    return f

def toDeg(e):
    f = float(e)/2.0/math.pi*360.0
    return f

def close_enough(a,b):
    epsilon = 0.00000001
    res = False
    if abs(a-b) < epsilon: res = True
    return res


def axisAngleNormalize(axisAngle):
    u = axisAngle[0]
    v = axisAngle[1]
    w = axisAngle[2]
    norm = (u**2 + v**2 + w**2)**0.5
    u = u / norm
    v = v / norm
    w = w / norm
    normalized = axisAngle
    normalized[0] = u
    normalized[1] = v
    normalized[2] = w
    
    return normalized

def convertAll(data, representation):
    
    axisAngle = np.zeros(4)
    
    if representation == 'axisAngle':
        quat = Quaternion(axis = data[:3], degrees = data[3])
        mat = axisAngle2orientationMatrix(data)
        euler = Quat2Euler(quat)
        axisAngle = data
        print(axisAngle)
        print(euler)
        print(mat)
        print(quat)
        
    elif representation == 'quaternion':
        quat = Quaternion(data)
        axisAngle[ :3] = quat.axis
        axisAngle[3] = quat.degrees
        mat = QuaternionToMatrix(quat)
        euler = Quat2Euler(quat)
        print(axisAngle)
        print(euler)
        print(mat)
        print(quat)
        
    elif representation == 'euler':
        mat = EulerToMatrix(data)
        quat = Quaternion(array = EulerToQuat(data))
        axisAngle[ :3] = quat.axis
        axisAngle[3] = quat.degrees 
        euler = data
        print(axisAngle)
        print(euler)
        print(mat)
        print(quat)
        
    else:
        print("choose keyword between axisAngle, quaternion or euler")
        quat = Quaternion()
        euler = np.zeros(3)
        mat = np.zeros((3, 3))
        
    
    return axisAngle, euler, mat, quat
                            
        
        
