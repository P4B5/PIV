import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse

from numpy.lib.function_base import _parse_input_dimensions





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
    cv2.imshow("Image with Aruco markers", image) # show the image with the markers
  
    return corners, ids, rejectedImgPoints


# 2. Compute the homography

def compute_homography(h_points_live_image, h_points_sample_image):

    '''
    HOMOGRAPHY MATRIX

    xd        xs      
    yd  = H * ys
    1         1
    '''

    print("h_points_input_image: {}".format(h_points_live_image))  
    print("h_points_template_image: {}".format(h_points_sample_image))  

    if len(h_points_live_image) == len(h_points_sample_image):
        # get the points to compute the homography
        xs = []
        ys = []
        xd = []
        yd = []
        for i in range(len(h_points_live_image)):
            xd.append(h_points_live_image[i][0][0])
            yd.append(h_points_live_image[i][0][1])
            xs.append(h_points_sample_image[i][0][0])
            ys.append(h_points_sample_image[i][0][1])

        # compute the homography
        xs = np.array(xs)
        ys = np.array(ys)
        xd = np.array(xd)
        yd = np.array(yd)
        H = np.array([[np.sum(xs*xd), np.sum(xs*yd), np.sum(xs)], [np.sum(ys*xd), np.sum(ys*yd), np.sum(ys)], [np.sum(xd), np.sum(yd), len(xs)]])
        H = np.linalg.inv(H)
        H = H/H[2,2]

        # print(H)
        # print("H: {}".format(H))


        # H = cv2.findHomography(source_image, destination_image)
        return  H #return the final image after apply the homography

    else:
        return None


# GET THE HOMOGRAPHY FROM A FIXED IMAGET
def detect_aruco_image():

    # 1. get the sample image
    template_image = '/home/pabs/PIV/datasets/InitialDataset/templates/template2_fewArucos.png'
    input_image = '/home/pabs/PIV/datasets/InitialDataset/templates/real_image_1.png'

    # get the points from the template image

    h_points_template_image = []
    h_points_input_image = []


    template_image = cv2.imread(template_image, cv2.IMREAD_COLOR)
    # template_image = cv2.resize(template_image, image.shape[1::-1])
    
    corners, ids, rejected_points = detect_aruco_markers(template_image)
    for i in range(len(ids)):
        h_points_template_image.append([(corners[i][0][0,0],corners[i][0][0,1]), ids[i][0]])

    input_image = cv2.imread(input_image, cv2.IMREAD_COLOR)
    corners, ids, rejected_points = detect_aruco_markers(input_image)
        
    # get the key points to compute the homography
    if ids is not None:
        for i in range(len(ids)):
            h_points_input_image.append([(corners[i][0][0,0],corners[i][0][0,1]), ids[i][0]])

        # h_points_sample_image = dete

    H = compute_homography(h_points_input_image, h_points_template_image) #compute homography
    print("H: {}".format(H))

    # get the new image
    # print("image shape: {}".format(image.shape))
    # print("sample_image.shape: {}".format(sample_image.shape))
    blank_image = np.zeros(input_image.shape, np.uint8)

 
    # if H is not None:
    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            point = np.array([i, j, 1])
            point = np.dot(H, point)
            point = point/point[2]
            point = point.astype(int)
            print("coordinates after apply H {} {}".format((i,j),point))
            blank_image[i, j] = input_image[i, j]


    #             print("i: {}, j: {}".format(i,j))
    #             x = np.array([j,i,1])
    #             x = np.dot(H,x)
    #             x = x/x[2]
    #             x = x.astype(int)
    #             if x[0] >= 0 and x[0] < template_image.shape[0] and x[1] >= 0 and x[1] < template_image.shape[1]:
    #                 print("correct value!")
    #                 blank_image[i,j] = input_image[x[1],x[0]]

    cv2.imshow("input image copy", input_image)
    cv2.waitKey(0)


    # cv2.destroyAllWindows()

    return None










# detect aruco markers in an live image
def detect_live_image_aruco():

    camera = cv2.VideoCapture(0) 
    # instead of camera pass the image file
    sample_image = '/home/pabs/PIV/datasets/InitialDataset/templates/template2_fewArucos.png'

    # get the points from the sample image

    h_points_sample_image = []
    sample_image = cv2.imread(sample_image, cv2.IMREAD_COLOR)
    result, image = camera.read()
    sample_image = cv2.resize(sample_image, image.shape[1::-1])
    
    corners, ids, rejected_points = detect_aruco_markers(sample_image)
    for i in range(len(ids)):
        h_points_sample_image.append([(corners[i][0][0,0],corners[i][0][0,1]), ids[i][0]])

    # real image computation
    while True:
        result, image = camera.read()
        # print(image.shape)
        corners, ids, rejected_points = detect_aruco_markers(image)
        h_points_live_image = []

        # get the key points to compute the homography
        if ids is not None:
            for i in range(len(ids)):
                h_points_live_image.append([(corners[i][0][0,0],corners[i][0][0,1]), ids[i][0]])

        # h_points_sample_image = dete

        H = compute_homography(h_points_live_image, h_points_sample_image) #compute homography
        print("H: {}".format(H))

        # get the new image
        # print("image shape: {}".format(image.shape))
        # print("sample_image.shape: {}".format(sample_image.shape))
        # blank_image = np.zeros(image.shape, np.uint8)

        # if H is not None:
        #     for i in range(blank_image.shape[0]):
        #         for j in range(blank_image.shape[1]):
        #             print("i: {}, j: {}".format(i,j))
        #             x = np.array([j,i,1])
        #             x = np.dot(H,x)
        #             x = x/x[2]
        #             x = x.astype(int)
        #             blank_image[i,j] = image[x[1],x[0]]

        # cv2.imshow("New image", blank_image)

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


CAMERA_MODE = 3

    
# get the source image
source_image = 0

template_image = cv2.imread('/home/pabs/PIV/datasets/InitialDataset/templates/template2_fewArucos.png',0)

if CAMERA_MODE == 1:
    source_img = detect_live_image_aruco()
elif CAMERA_MODE == 2:
    source_img ='/home/pabs/PIV/datasets/InitialDataset/templates/real_image_1.png'
    source_img = show_file_image(source_img, "Source image")
elif CAMERA_MODE == 3:
    detect_aruco_image()
else:
    print("Invalid camera mode")

# print("source_img: {}".format(source_img))
# corners, ids, rejection_points = detect_aruco_markers(source_img)





# get the aruco marker of the template -> 4 points coordinates
# get the arucor marker of the source image -> 4 points coordinates
# get the homography matrix


# # compute the homography
# H = compute_homography(source_img, destination_image)
# print("Homography matrix: {}".format(H))


# except:
#   print("An exception occurred")
 