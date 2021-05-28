import cv2
import numpy as np

def scannerDocument(image):
    # constantes
    height = 500
    
    originale = image.copy()
    
    retour = originale
    
    r = image.shape[0]/height
    image = resize(image, height)
    
    res = resize(image, height)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)
    
    edged = inverse_colors(edged)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    eroded = cv2.erode(edged, kernel, iterations = 2)
    eroded = inverse_colors(eroded)
        
    contours, _ = cv2.findContours(eroded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = sorted(contours, key= cv2.contourArea, reverse=True)[:5]
    
    cpImg = image.copy()
    
    if(len(contours) != 0):
        
        found = False
    
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            print(len(approx))
            
            
            if(len(approx) == 4): # si le contour a 4 coté on garde
                screenCnt = approx
                found = True
                break;
            
            
        if(found):
            cv2.drawContours(cpImg,[screenCnt], -1, (255, 0, 0), 4)
            
            warped = four_point_transform(originale, screenCnt.reshape(4,2) * r)
    
            retour = warped
    return retour
    
# inverse les couleurs de l'image
def inverse_colors(img):
    img = (255 - img)
    return img

def resize(img, height):
    return cv2.resize(img, (int(img.shape[1]/(img.shape[0]/height)),height), interpolation=cv2.INTER_AREA)        

# pyimagesearch
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect