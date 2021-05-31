import cv2
import numpy as np

# PAS = Pression Artérielle Systolique
# PAD = Pression Artérielle Diastolique

# Retourne la PAS et la PAD détectées sur une image d'un appareil de mesure de tension artérielle
def detecterTensions(image):
    _, PAS, PAD = detecterTensionsDebug(image)
    return PAS, PAD

# Retourne la PAS et la PAD détectées sur une image d'un appareil de mesure de tension
# Retourne également une liste d'image qui correspond aux différentes étapes intermédiaires de calcul
def detecterTensionsDebug(image):
    detecteurTension = DetecteurTension(400) # l'image traitée sera réduite à une image de 400 pixels de haut
    
    # variables relatives au traitement d'image
    alpha = 1.0
    blurAmount = 3
    threshold = 83
    adjustment = 11
    erode = 4
    iterations = 3
    
    detecteurTension.setImage(image)
    detecteurTension.setParametres(alpha, blurAmount, threshold, adjustment, erode, iterations)
    _, PAS, PAD = detecteurTension.detecterTensions()
    
    return (_,PAS,PAD)

# Classe qui gère la détection de tensions dans une images
class DetecteurTension:
    def __init__(self, height):
        self.height = height
        self.width = 0
        self.original = None # image originale
        self.R = 1 # ratios (largeur/hauteur) des images aux différentes étapes de traitement lors des recadrages
        self.xR = 0 # décallage en x utilisé lors des recadrages
        self.yR = 0  # décallage en y utilisé lors des recadrages
        self.inversee = False
    
    # redimensionne une image à une hauteur choisi
    # retourne l'image et largeur retourne -1 si la hauteur ne permet pas de redimensionner
    def resize_to_height(self, img, height):
        r = img.shape[0] / float(height)
        if img.shape[0] > 0 and img.shape[1] > 0:
            dim = (int(img.shape[1] / r), height)
            res = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            return res, dim[0]
        else:
            return img, -1
        
    # définir l'image dans laquelle on va chercher les chiffres de tension
    def setImage(self, image):
        self.original = image
        self.R = 1
        self.xR = 0
        self.yR = 0
        self.resized, self.width = self.resize_to_height(self.original, self.height)
        self.R = (self.original.shape[0] / self.resized.shape[0]) # hOrigine / height
        
    # défini les paramètres commme le niveau de flou pour le traitement de la photo
    def setParametres(self, alpha, blurAmount, threshold, adjustment, erode, iterations):
        self.alpha = alpha
        self.blurAmount = blurAmount
        self.threshold = threshold
        self.adjustment = adjustment
        self.erode = erode
        self.iterations = iterations
        
        # surexpose l'image
    def exposure(self, img, alpha):
        alpha = float(alpha)
        return cv2.multiply(img, np.array([alpha]))
    
    # inverse les couleurs de l'image
    def inverse_colors(self, img):
        img = (255 - img)
        return img
    
    # transforme l'image en niveau de gris
    def echelleGris(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # applique un flou à l'image
    def blur(self, img, blurAmount):
        return cv2.GaussianBlur(img, (blurAmount, blurAmount), 0)
    
    # applique un seuillage à l'image
    def thresh(self, img, threshold, adjustment):
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, threshold, adjustment)

    # "errode" une image
    def errode(self, img, erode, iterations):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode,erode))
        eroded = cv2.erode(img, kernel, iterations = iterations)
        return eroded
    
    # applique les premiers traitements à l'image, soit, dans l'ordre : 
    # - surexposition
    # - niveau de girs
    # - flou
    # - seuillage
    # - inversion des couleurs
    def traitement1(self, img):
        exp = self.exposure(img, self.alpha)
        grise = self.echelleGris(exp)
        floue = self.blur(grise, self.blurAmount)
        seuillee = self.thresh(floue, self.threshold, self.adjustment)
        inversee = self.inverse_colors(seuillee)
        
        return inversee
    
    # prend une image seuillee en noir et blanc et ne garde que les formes dont le contour
    # ressemble à peu près aux dimensions de chiffres.
    # * retourne l'image traitées avec tout les contours détectés
    # * retourne l'image avec uniquement les contours de chiffres selectionnes
    # TODO : resize avec le x et y min et max des contours gardés + quelques(5?) pixels de marge
    def selectionnerContourChiffres(self, img):
        # detection de contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours
        contoursNombres = []
        
        
        # transformer une image avec un seuil canal en image rgb (mais en noir et blanc)
        hImage,wImage = img.shape
        imgCP = np.zeros((hImage,wImage,3), np.uint8)
        imgCP[:,:,0] = img
        imgCP[:,:,1] = img
        imgCP[:,:,2] = img
        
        if len(contours) > 0:
            # on supprime tout les contours dont les dimensions ne ressemblent pas à celles d'un chiffres
            # comme par exemple les contours trop grands ou trop petits, trop vides, trop longs...
            for contour in contours:
                [x, y, w, h] = cv2.boundingRect(contour)
                aspect = float(w) / h
                size = w * h
                imgContour = img[y:y+h, x:x+w]
                h,w = imgContour.shape
                nbPixBlancs = np.sum(imgContour==255)
                propPixBlanc = nbPixBlancs/(h*w) # proportion de pixel blancs sur le contour en cours de traitement
                
                # dessiner sur l'image en noir et blanc les contours (même si on ne les garde, on peut avoir 
                # une idée de tous les contours détectes à l'origine)
                cv2.rectangle(imgCP, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                # enlever les contours trop grands
                if(size > (wImage*hImage)/8):
                    continue
                
                # enlever les contours trop petits :
                if(size < (hImage*wImage)/300):
                    #print("contourTropPetit : " + str(size) + " vs : " + str((hImage*wImage)/350))
                    continue
                                
                # enlever les contours pas assez remplis
                if(propPixBlanc<0.2):
                    continue
                
                if(propPixBlanc<0.5) and aspect >3 and size > (hImage*wImage)/20:
                    continue
                
                # enlever les contours qui font plus de la moitié de la hauteur de l'image
                if h>hImage/2:
                    continue
                
                # enlever les contours trop longs
                if aspect < 0.1:
                    continue
                
                # enlever les contours carres trop petits (par exemple les coeurs)
                if(aspect < 1.3 and aspect > 0.7 and size < (wImage*hImage)/100):
                    continue
                
                contoursNombres.append(contour)
        
            # garder dans l'image originale uniqumement les zones qui correspondent aux 
            # contours qu'on à gardé
            mask = np.zeros((hImage,wImage,1), np.uint8)
            
            # x et y min et max pour resize autour des contours
            xMin = wImage
            yMin = hImage
            xMax = 0
            yMax = 0
            
            
        if(len(contoursNombres) > 0):
            # sélectionner les contours de chiffre dans l'image grâce à un mask
            for contour in contoursNombres:
                [x, y, w, h] = cv2.boundingRect(contour) 
                cv2.rectangle(mask, (x, y), (x+ w, y + h), (255), -1)
                
                if x < xMin:
                    xMin = x
                if y < yMin:
                    yMin = y
                if x+w > xMax:
                    xMax = x+w
                if y+h > yMax:
                    yMax = y+h
            
            masked = cv2.bitwise_and(img, img, mask = mask)
        
            # recadrer avec les x et y min et max avant d'eroder
            # prendre 5 pix de marge
            xMin -= 5
            yMin -= 5
            xMax += 5
            yMax += 5
            if xMin < 0:
                xMin = 0
            if yMin < 0:
                yMin = 0
            if xMax > wImage:
                xMax = wImage
            if yMax > hImage:
                yMax = hImage
                
            print("xMin: " + str(xMin) + " yMin: " + str(yMin) + " xMax: " + str(xMax) + " yMax: " + str(yMax))
            imgRecadre = masked[yMin:yMax, xMin:xMax]
            imgResized, _ = self.resize_to_height(imgRecadre, self.height)
        
            # eroder pour que les segments d'un chiffres se colent et forment une seule forme
            erodee = self.inverse_colors(self.errode(self.inverse_colors(imgResized), self.erode, self.iterations))
            return imgCP, erodee
        else:
            return imgCP, np.zeros((hImage,wImage), np.uint8)
    
    # pour chaque contour détectés dans une image essaye de déterminer le chiffre qui est dedans
    # rassemble les chiffres par nombres en fonction de leur position
    # * retourne une image avec de debug avec les zones analyées 
    # * retourne une liste de nombres
    # ? inclure la taille des nombres et leur pos? parfois la fréquence et comprise sur l'écran mais avec une taille plus petite
    def detecterNombres(self, img):
        
        nombresTrouves = []
        nombresRetour = []
        
        # structure 7 segments :   
        #       _____
        #      |  A  |
        #   __  ‾‾‾‾‾  __
        #  | F|       | B|
        #  |  |       |  |
        #   ‾‾  _____  ‾‾
        #      |  G  |
        #   __  ‾‾‾‾‾  __
        #  | E|       | C|
        #  |  |       |  |
        #   ‾‾  _____  ‾‾
        #      |  D  |
        #       ‾‾‾‾‾    
        
        DIGITS_LOOKUP = {
        0: [1, 1, 1, 1, 1, 1, 0],
        2: [1, 1, 0, 1, 1, 0, 1],
        3: [1, 1, 1, 1, 0, 0, 1],
        4: [0, 1, 1, 0, 0, 1, 1],
        5: [1, 0, 1, 1, 0, 1, 1],
        6: [1, 0, 1, 1, 1, 1, 1],
        7: [1, 1, 1, 0, 0, 0, 0],
        70: [1, 1, 1, 0, 0, 1, 0], #deuxième 7
        8: [1, 1, 1, 1, 1, 1, 1],
        9: [1, 1, 1, 1, 0, 1, 1]
        }

        propDigit = [0.6,0.3,0.3,0.6,0.3,0.3,0.6]
        
        hImage,wImage = img.shape
        
        imgCP = np.zeros((hImage,wImage,3), np.uint8)
        imgCP[:,:,0] = img
        imgCP[:,:,1] = img
        imgCP[:,:,2] = img
        
        #selectionner les contours, à ce stade, chaque contour devrait être un nombre
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours
            
        if len(contours) > 0:
            # pour chaque contour déterminer le chiffre qui est dedans et le rajouter avec ces positions dans le tableau chiffres[]
            chiffres = []
            for contour in contours:
                [x, y, w, h] = cv2.boundingRect(contour)
                aspect = float(w) / h
                size = w * h
                
                #si size < 1/50 de la taille totale, c'est pas un nombre de
                if size < (hImage*wImage)/60:
                    continue
                
                cv2.rectangle(imgCP, (x, y), (x + w, y + h), (0, 0, 255), 2)

                cv2.rectangle(imgCP, (x + int(w/2) - 2, y + int(h/2) - 2), (x + int(w/2) + 2, y + int(h/2) + 2), (255,0,0), 2)
                
                if aspect < 0.4: # TODO : VERIFIER RATIO pour qu'un chiffre soit un 1
                    chiffres.append(["1",x+w/2,y+h/2])
                else: #trouver quel chiffre c'est selon les 7 segments voir schema au dessus pour les lettres des segments
                    cropped = img[y:y+h, x:x+w]
                    # chaque zone correspond à un segment
                    zoneA = cropped[int(0):int(0.2*h), int(0.375*w):int(0.625*w)]
                    zoneB = cropped[int(0.2*h):int(0.4*h), int(0.66*w):int(w)]
                    zoneC = cropped[int(0.6*h):int(0.8*h), int(0.66*w):int(w)]
                    zoneD = cropped[int(0.8*h):int(h), int(0.375*w):int(0.625*w)]
                    zoneE = cropped[int(0.6*h):int(0.8*h), int(0):int(0.33*w)]
                    zoneF = cropped[int(0.2*h):int(0.4*h), int(0):int(0.33*w)]
                    zoneG = cropped[int(0.4*h):int(0.6*h), int(0.375*w):int(0.625*w)]
                    
                    # dessinner les zones sur l'image
                    cv2.rectangle(imgCP, (x + int(0.375*w), y), (x + int(0.625*w), y + int(0.2*h)), (0,255,0),3)
                    cv2.rectangle(imgCP, (x + int(0.66*w), y + int(0.2*h)), (x + int(w), y + int(0.4*h)), (0,255,0),3)
                    cv2.rectangle(imgCP, (x + int(0.66*w), y + int(0.6*h)), (x + int(w), y + int(0.8*h)), (0,255,0),3)
                    cv2.rectangle(imgCP, (x + int(0.375*w), y + int(0.4*h)), (x + int(0.625*w), y + int(0.6*h)), (0,255,0),3)
                    cv2.rectangle(imgCP, (x, y + int(0.2*h)), (x + int(0.33*w), y + int(0.4*h)), (0,255,0),3)
                    cv2.rectangle(imgCP, (x, y + int(0.6*h)), (x + int(0.33*w), y + int(0.8*h)), (0,255,0),3)
                    cv2.rectangle(imgCP, (x + int(0.375*w), y +int(0.8*h) ), (x + int(0.625*w), y + int(h)), (0,255,0),3)
                    
                    zones = [zoneA, zoneB, zoneC, zoneD, zoneE, zoneF, zoneG]
                    binSeg = []
                    indice = 0
                    # pour chaque zone (qui correspond donc à un segment) regarder si elle est remplie
                    # chaque zone dois contenir une prop de pixel blanc min pour être remplie
                    # cette prop change en fonction de la zone car selon les afficheurs les segments des côtés sont plus ou moins penchés
                    for zone in zones:
                        h2,w2 = zone.shape
                        nbWhite = np.sum(zone==255) 
                        if(nbWhite > 0):
                            # print((nbWhite)/(h*w))
                            if(nbWhite/(h2*w2)>propDigit[indice]):
                                binSeg.append(1)
                            else:
                                binSeg.append(0)
                        else:
                            # print(0)
                            binSeg.append(0)
                        indice +=1
                    # print(binSeg)
                    outputNumbers = None
                    if(binSeg==DIGITS_LOOKUP[0]):
                        outputNumbers = "0 "
                    elif(binSeg==DIGITS_LOOKUP[2]):
                        outputNumbers = "2 "
                    elif(binSeg==DIGITS_LOOKUP[3]):
                        outputNumbers = "3 "
                    elif(binSeg==DIGITS_LOOKUP[4]):
                        outputNumbers = "4 "
                    elif(binSeg==DIGITS_LOOKUP[5]):
                        outputNumbers = "5 "
                    elif(binSeg==DIGITS_LOOKUP[6]):
                        outputNumbers = "6 "
                    elif(binSeg==DIGITS_LOOKUP[7]):
                        outputNumbers = "7 "
                    elif(binSeg==DIGITS_LOOKUP[70]):
                        outputNumbers = "7 "
                    elif(binSeg==DIGITS_LOOKUP[8]):
                        outputNumbers = "8 "
                    elif(binSeg==DIGITS_LOOKUP[9]):
                        outputNumbers = "9 "
                    if outputNumbers!=None:
                        chiffres.append([str(outputNumbers),x+w/2,y+h/2])
                
            # * detecter quels chiffres appartiennent au même nombre
            # marge de hauteur pour que les chiffres soient considérés sur la même ligne
            # si la position du chiffre 1 est comprise entre la position du chiffre 2 + marge et la position du chiffre 2 - marge alors ils sont dans le même nombre
            marge = hImage/16
            
            # TODO : rajouter une marge de largeur, si deux chiffres sont trop éloignés en largeur alors ils ne sont pas dans le même nombre
            
            nombres = []
            indiceNombre = -1
            #print(len(chiffres))
            for chiffre in chiffres:
                #print(chiffre[0])
                indiceNombre = -1
                if len(nombres) == 0:
                    nouveauNombre = []
                    nouveauNombre.append(chiffre)
                    nombres.append(nouveauNombre)
                else:
                    done = False
                    for nombre in nombres:
                        indiceNombre +=1
                        # on compare les coord y du chiffre avec celles du premier chiffre d'un des nombres qu'on teste
                        if chiffre[2] > nombre[0][2] - marge and chiffre[2] < nombre[0][2] + marge: #chiffre[2]  ->  coord y
                            #même nombre
                            #print("append to : ")
                            #print(nombres[indiceNombre])
                            nombres[indiceNombre].append(chiffre)
                            done = True
                            break
                    if(not done):       
                        #print("nouveau nombre")
                        #print(nombres[indiceNombre])
                        nouveauNombre = []
                        nouveauNombre.append(chiffre)
                        nombres.append(nouveauNombre)  
            #print(nombres)
            
            # trier les chiffres d'un nombre de gauche à droite
            for nombre in nombres:
                yMoyNombre = 0
                for chiffre in nombre:
                    yMoyNombre += chiffre[2]
                yMoyNombre /= len(nombre)
                nbChiffre = len(nombre)
                nbFinal = 0
                for i in range(nbChiffre):
                    minX = nombre[0]
                    for chiffre in nombre:
                        if chiffre[1] < minX[1]:
                            minX = chiffre
                    nbFinal += int(minX[0])*pow(10,nbChiffre-1-i)
                    nombre.remove(minX)
                #print(nbFinal)
                nombresTrouves.append([nbFinal,yMoyNombre])
                
            # on enlève les nombres bcp trop petit ou  bcp trop grand (ex : 200 de pression c'est pas possible, 10 non plus)
            nombresRetour = nombresTrouves.copy()
            for nombre in nombresTrouves:
                #print(nombre)
                if(nombre[0] < 50 or nombre[0] > 250):
                    nombresRetour.remove(nombre)
            #print("")
        return imgCP, nombresRetour
    
    # detecte les contours dans une zone de l'image image et renvoie les coord x, y et la les dimensions w, h
    def plusGrosContour(self, img):
        
        # TODO : prendre un mix des deux plus gros contour? (image 15)
        hImage,wImage,_ = img.shape
        
        traitee = self.traitement1(img)
        
        # detection de contours
        contours, _ = cv2.findContours(traitee, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours
        
        # [xF, yF, wF, hF] = [0,0,0,0] # plus gros contour
        # [xF2, yF2, wF2, hF2] = [-1,-1,-1,-1] # deuxième plus gros contour
        
        # if len(contours) > 0:
        #     [xF, yF, wF, hF] = cv2.boundingRect(contours[0])
        #     for contour in contours:
        #         [x, y, w, h] = cv2.boundingRect(contour)
        #         aspect = float(w) / h
        #         size = w * h
        #         if size > 0.95 * hImage * wImage :# Suppprimer les contours si ils font une trop grosse partie de l'image
        #             continue
        #         if(size > wF * hF):
        #             [xF2, yF2, wF2, hF2] = [xF, yF, wF, hF]
        #             [xF, yF, wF, hF] = cv2.boundingRect(contour)
        #         elif (size > wF2 * hF2):
        #             [xF2, yF2, wF2, hF2] = cv2.boundingRect(contour)
                    
        #     x = xF if xF < xF2 else xF2
        #     y = yF if yF < yF2 else yF2
        #     w = (xF + wF - x) if (xF + wF) > (xF2 + wF2) else (xF2 + wF2 - x)
        #     h = (yF + hF - y) if (yF + hF) > (yF2 + hF2) else (yF2 + hF2 - y)
        #     return x, y, w, h
        # else:
        #     return -1, -1, -1, -1
        
        [xF, yF, wF, hF] = [0,0,0,0] # plus gros contour en largueur
        [xF2, yF2, wF2, hF2] = [0,0,0,0] # plus contour en hauteur
        
        if len(contours) > 0:
            [xF, yF, wF, hF] = cv2.boundingRect(contours[0])
            for contour in contours:
                [x, y, w, h] = cv2.boundingRect(contour)
                aspect = float(w) / h
                size = w * h
                if size > 0.95 * hImage * wImage :# Suppprimer les contours si ils font une trop grosse partie de l'image
                    continue
                if(w > wF):
                    [xF, yF, wF, hF] = cv2.boundingRect(contour)
                if (h > hF2):
                    [xF2, yF2, wF2, hF2] = cv2.boundingRect(contour)
                    
            x = xF if xF < xF2 else xF2
            y = yF if yF < yF2 else yF2
            w = (xF + wF - x) if (xF + wF) > (xF2 + wF2) else (xF2 + wF2 - x)
            h = (yF + hF - y) if (yF + hF) > (yF2 + hF2) else (yF2 + hF2 - y)
            
            if w * h < 0.95 * hImage * wImage:
                return x, y, w, h
            elif wF * hF > wF2 * hF2:
                return xF, yF, wF, hF
            else:
                return xF2, yF2, wF2, hF2
        else:
            return -1, -1, -1, -1
    
    # cherche la tension sys et dias dans une zone spécifique de l'écran
    # * retourne la PAS et la PAD si elle sont trouvees
    # * retourne les images de debug
    def detecterTensionDansZone(self, indice):
        # deux cas a gérer : écritures noires et écritures blanches
        # dans un cas il faut faire une inversion des couleurs au debut, dans l'autre cas non
        
        # ! Pour l'instant on gère que les écritures noires.
        
        imagesDebug = []
        PAS = 0
        PAD = 0
        
        traitee = self.traitement1(self.resized)
        imagesDebug.append(("traitee " + str(indice),traitee))
        
        img, erodee = self.selectionnerContourChiffres(traitee)
        
        imagesDebug.append(("contours " + str(indice),img))
        
        img, nombres = self.detecterNombres(erodee)
        imagesDebug.append(("nombres " + str(indice),img))
        
        # * plusieurs cas :
        # - il y a moins de deux nombres détectes, dans ce cas on n'a pas trouvé la PAS et la PAD, on renvoie 0
        # - il y a deux nombres exactement détectes, le plus grand seras la PAS et le plus petit sera la PAD
        # - il y a 3 nombres, on compare les pos en y (parmis les 3 nombres, il peut y avoir les deux PA + la FC), et on prend les deux nombres du haut
        # - il y a plus de 3 nombres, c'est trop, on renvoie 0
        nbNombres = len(nombres)
        if(nbNombres == 2):
            if(nombres[0] > nombres[1]):
                PAS = nombres[0][0]
                PAD = nombres[1][0]
            else:
                PAS = nombres[1][0]
                PAD = nombres[0][0]
        elif(nbNombres == 3):
            maxY = nombres[0]
            for nombre in nombres:
                if nombre[1] > maxY[1] : # nombre[1] est la coord en Y
                    max = nombre
            nombres.remove(maxY)
            if(nombres[0] > nombres[1]):
                PAS = nombres[0][0]
                PAD = nombres[1][0]
            else:
                PAS = nombres[1][0]
                PAD = nombres[0][0]
        else:
            print(nombres)
        
        return imagesDebug, PAS, PAD
    
    # détecte la PAS et la PAD dans une image
    # * retourne la PAS et la PAD si elle sont trouvees
    def detecterTensions(self):
        self.setImage(self.original)
        imagesDebug = [] # images affichées à l'écran pour débugger
        
        imagesDebug.append(('Originale',self.resized))
        
        # cherche PAS et PAD dans l'image entière, si ça ne fonctionne pas, prendre
        # la zone la plus importante de l'image (selon la détection de contour)
        # et relancer la recherche (faire ça 3 fois ?)
        
        PAS = 0
        PAD = 0
        
        iterations = 0
        
        x = 0
        y = 0
        w = self.resized.shape[1]
        h = self.height
        
        while(iterations < 5): # ? nombre d'itérations à vérifier
            imgs, PAS, PAD = self.detecterTensionDansZone(iterations)
            for img in imgs:
                imagesDebug.append(img)
            if PAS != 0 and PAD != 0:
                break;
            
            x, y, w, h = self.plusGrosContour(self.resized)
            
            if(x == -1):
                break;  
            
            xM = int(x * self.R + self.xR)
            yM = int(y * self.R + self.yR)
            wM = int(w * self.R)
            hM = int(h * self.R)
                        
            cropped = self.original[yM:yM+hM, xM:xM+wM]
            self.resized, c = self.resize_to_height(cropped, self.height)
                        
            if (c == -1):
                break;
            
            self.xR = xM
            self.yR = yM
            self.R = self.R * (h / self.height)

            iterations += 1
        
        # si rien de détecté on tente d'inverser les couleurs   
        if(PAD == 0 and self.inversee == False):
            self.inversee = True
            imagesDebug.clear()
            self.original = self.inverse_colors(self.original)
            imagesDebug, PAS, PAD = self.detecterTensions()
            
        
        return imagesDebug, PAS, PAD