import cv2
import numpy as np

def detecterTension(image):
    
    detecteurTension = DetecteurTension(400)
    
    # variables relatives au traitement d'image
    alpha = 1.0
    blurAmount = 3
    threshold = 83
    adjustment = 11
    erode = 4
    iterations = 3
    
    detecteurTension.setImage(image)
    detecteurTension.setParametres(alpha, blurAmount, threshold, adjustment, erode, iterations)
    PAS, PAD = detecteurTension.detecterTensions()
    
    return (PAS,PAD)

class DetecteurTension:
    def __init__(self, height):
        self.height = height
        self.width = 0
        self.original = None
        self.R = []
        self.xR = []
        self.yR = []
    
    # redimensionne une image à une hauteur choisi, retourne l'image et largeur retourne -1 si la hauteur ne permet pas de resize
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
        self.R.clear()
        self.xR.clear()
        self.yR.clear()
        self.resized, self.width = self.resize_to_height(self.original, self.height)
        self.R.append(self.original.shape[0] / self.resized.shape[0])
        
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
    
    # prend une image seuillee en noir et blanc et ne garde que les forment dont le contour
    # ressemble à peu près aux dimensions de chiffres.
    # * retourne l'image originale avec tout les contours détectés
    # * retourne l'image avec uniquement les contours de chiffres selectionnes
    def selectionnerContourChiffres(self, img):
        # detection de contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours
        contoursNombres = []
        
        # TODO : PRENDRE LE X, Y min et max prendre quelques pixels de plus de toutes les côtés et resize?
        
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
                
                # dessiner sur l'image noir et blanc les contours (même si on ne les garde pas pour avoir une idée 
                # de tous les contours détectes à l'origine)
                cv2.rectangle(imgCP, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                #enlever les contours trop grands
                if(size > (wImage*hImage)/8):
                    continue
                
                #sauter les contours pas assez remplis
                imgContour = img[y:y+h, x:x+w]
                h,w = imgContour.shape
                nbWhite = np.sum(imgContour==255)
                if(nbWhite/(h*w)<0.2):
                    continue
                
                if(nbWhite/(h*w)<0.5) and aspect >4:
                    continue
                
                #enlever les contours qui font plus de la moitié de la hauteur de l'image
                if h>hImage/2:
                    continue
                
                #enlever les contours trop longs
                if aspect < 0.1:
                    continue
                
                #enlever les contours trop petits :
                if(size < (hImage*wImage)/350):
                    continue
                
                #enlever les contours carres trop petits (par exemple les coeurs)
                if(aspect < 1.3 and aspect > 0.7 and size < (wImage*hImage)/100):
                    continue
                
                contoursNombres.append(contour)
        
            # garder dans l'image originale uniqumement les zones qui correspondent aux 
            # contours qu'on à gardé
            mask = np.zeros((hImage,wImage,1), np.uint8)
            for contour in contoursNombres:
                [x, y, w, h] = cv2.boundingRect(contour) 
                cv2.rectangle(mask, (x, y), (x+ w, y + h), (255), -1)
            
            masked = cv2.bitwise_and(img, img, mask = mask)
            # copymasked = np.zeros((self.height,self.width,3), np.uint8)
            # copymasked[:,:,0] = masked[:,:,0]
            # copymasked[:,:,1] = masked[:,:,0]
            # copymasked[:,:,2] = masked[:,:,0]
        
            # eroder pour que les segments d'un chiffres se colent et forment une seule forme
            erodee = self.inverse_colors(self.errode(self.inverse_colors(masked), self.erode, self.iterations))
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
                
                if aspect < 0.5: #un 1
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
        
        [xF, yF, wF, hF] = [0,0,0,0]
        
        if len(contours) > 0:
            [xF, yF, wF, hF] = cv2.boundingRect(contours[0])
            for contour in contours:
                [x, y, w, h] = cv2.boundingRect(contour)
                aspect = float(w) / h
                size = w * h
                if size > 0.95 * hImage * wImage :#? vérifier le 0.95 (Suppprimer les contours si ils font une rop grosse partie de l'image)
                    continue
                if(size > wF * hF):
                    [xF, yF, wF, hF] = cv2.boundingRect(contour)
            return xF, yF, wF, hF
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
        
        while(iterations < 3):
            imgs, PAS, PAD = self.detecterTensionDansZone(iterations)
            for img in imgs:
                imagesDebug.append(img)
            if PAS != 0 and PAD != 0:
                break;
            
            x, y, w, h = self.plusGrosContour(self.resized)
            
            if(x == -1):
                break;  
            
            
            # hOrigine,wOrigine,_ = self.original.shape
            # hImage,wImage,_ = self.resized.shape
            # xM = int((x*wOrigine)/wImage)
            # yM = int((y*hOrigine)/hImage)
            # wM = int((w*wOrigine)/wImage)
            # hM = int((h*hOrigine)/hImage)
            if iterations == 0 :
                xM = int(x*self.R[0])
                yM = int(y*self.R[0])
                wM = int(w*self.R[0])
                hM = int(h*self.R[0])
                
                cropped = self.original[yM:yM+hM, xM:xM+wM]
                self.resized, _ = self.resize_to_height(cropped, self.height)
                
                self.xR.append(x)
                self.yR.append(y)
                self.R.append(h/self.height)
            elif iterations == 1: #! TEMP
                
                xM = int(((x*self.R[1]) + self.xR[0])*self.R[0])
                yM = int(((y*self.R[1]) + self.yR[0])*self.R[0])
                wM = int((w*self.R[1])*self.R[0])
                hM = int((h*self.R[1])*self.R[0])
                
                print(str(x) + " " + 
                      str(y) + " " + 
                      str(self.R[1]) + " " +
                      str(self.xR[0]) + " " + 
                      str(self.yR[0]) + " " + 
                      str(self.R[0]))
                
                print(str(xM) + " " +
                      str(yM) + " " +
                      str(wM) + " " +
                      str(hM))
                cropped = self.original[yM:yM+hM, xM:xM+wM]
                self.resized, c = self.resize_to_height(cropped, self.height)
                
                if c == -1:
                    break;
                
                self.xR.append(xM)
                self.yR.append(yM)
                self.R.append(h/self.height)
                
                      
            iterations += 1
               
        
        return PAS, PAD