import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse





print("This is a Simple Scanner")
print("OpenCV version used: {}".format(cv2.__version__))

# show image from a file
def show_file_image(img, title):
    image = cv2.imread(img, cv2.IMREAD_COLOR)
    image_resize = cv2.resize(image, (0,0), fx=0.25, fy=0.25)
    if image is None:
        sys.exit('Could not open or find the image!')

    # plt.imshow(image, cmap='gray')
    cv2.imshow("Source image", image_resize)
    cv2.waitKey(0)
    return image


# show live image from a webacam camera
def show_live_image():
    camera = cv2.VideoCapture(0)
    while True:
        result, image = camera.read()
        cv2.imshow("Live Camera", image)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit

    cv2.destroyAllWindows()
        
    return None

    

# ---- Get the Arucu markers ----
# 1. specify aruco dictionary
# 2. specify parameters
# 3. detect markers


def detect_aruco_markers(image):
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)    
    cv2.aruco.drawDetectedMarkers(image, corners, ids, (0,255,0))
    cv2.imshow("Image with Aruco markers", image)
  
    return corners, ids, rejectedImgPoints


# 2. Compute the homography

def compute_homography(h_points_live_image, h_points_sample_image):

    '''
    HOMOGRAPHY MATRIX

    xd        xs      
    yd  = H * ys
    1         1
    '''

    print("h_points_live_image: {}".format(h_points_live_image))  
    print("h_points_sample_image: {}".format(h_points_sample_image))  

    # H = cv2.findHomography(source_image, destination_image)
    return  None #return the final image after apply the homography



# detect aruco markers in an live image
def detect_live_image_aruco():

    camera = cv2.VideoCapture(0)
    sample_image = '/home/pabs/PIV/datasets/InitialDataset/templates/template2_fewArucos.png'

    # get the points from the sample image

    h_points_sample_image = []
    image = cv2.imread(sample_image, cv2.IMREAD_COLOR)
    corners, ids, rejected_points = detect_aruco_markers(image)
    for i in range(len(ids)):
        h_points_sample_image.append([(corners[i][0][0,0],corners[i][0][0,1]), ids[i][0]])

    # real image computation
    while True:
        result, image = camera.read()
        corners, ids, rejected_points = detect_aruco_markers(image)
        h_points_live_image = []

        # get the key points to compute the homography
        if ids is not None:
            for i in range(len(ids)):
                h_points_live_image.append([(corners[i][0][0,0],corners[i][0][0,1]), ids[i][0]])

        # h_points_sample_image = dete

        compute_homography(h_points_live_image, h_points_sample_image) #compute homography

        if cv2.waitKey(1) == 27: 
            break  # esc to quit

    cv2.destroyAllWindows()
        
    return None




# 4. Get the positions of the markers in the new image








###############################################################
#
#   
#                          MAIN PROGRAM
#
#
###############################################################



'''
CAMERA MODES:

    0: webcam
    1: image file

'''


CAMERA_MODE = 1

    
# get the source image
source_image = 0

template_image = cv2.imread('/home/pabs/PIV/datasets/InitialDataset/templates/template2_fewArucos.png',0)

if CAMERA_MODE == 1:
    source_img = detect_live_image_aruco()
elif CAMERA_MODE == 2:
    source_img ='/home/pabs/PIV/datasets/InitialDataset/templates/real_image_1.png'
    source_img = show_file_image(source_img, "Source image")
else:
    print("Invalid camera mode")

print("source_img: {}".format(source_img))
corners, ids, rejection_points = detect_aruco_markers(source_img)





# get the aruco marker of the template -> 4 points coordinates
# get the arucor marker of the source image -> 4 points coordinates
# get the homography matrix


# # compute the homography
# H = compute_homography(source_img, destination_image)
# print("Homography matrix: {}".format(H))


# except:
#   print("An exception occurred")
 