import sys
from tkinter.tix import InputOnly
from typing import final
import cv2
import cv2.aruco as aruco
import os
import numpy as np


'''

--- GROUP NUMBER 13 ---

Jonas Rogde Jørgensen           -- ist1101351
Afonso dos Santos Caetano Dias  -- ist87139
Pablo Castellanos Lopez         -- ist1101077

PIV PROJECT 2021/22

execution: pivproject2021.py <task> <path_to_template> <path_to_output_folder> <path_to_input_folder>

REQUIRED PACKAGES TO BE INSTALLED TO RUN THE CODE:
pip install python-opencv
pip install opencv-contrib-python
pip install numpy


TASKS:
1. COMPUTE HOMOGRAHY USING ARUCO MARKERS [x]
2. COMPUTE HOMOGRAHY WITHOUT ARUCO MARKERS [x]
4. COMPUTE HOMOGRAHY USING 2 RGB CAMERAS

'''


print("OpenCV version used: {}".format(cv2.__version__))


###############################################################
#
#   
#                          FUCTIONS
#
#
###############################################################




# get the aruco markers from an image
def detect_aruco_markers(image):
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)    
    aruco.drawDetectedMarkers(image, corners, ids, (0,255,0))
    return corners, ids, rejectedImgPoints


# copute the homography
def compute_homography(h_points_input_image, h_points_template_image):

    '''
    HOMOGRAPHY MATRIX

    xd        xs      
    yd  = H * ys
    1         1
    '''

    # get the P matrix
    P = []
    for i in range(len(h_points_input_image)):
        xd_1 = h_points_input_image[i][0]
        yd_1 = h_points_input_image[i][1]
        xs_1 = h_points_template_image[i][0]
        ys_1 = h_points_template_image[i][1]
        P.append([xs_1, ys_1, 1, 0, 0, 0, -xs_1*xs_1, -ys_1*xd_1, -xd_1])
        P.append([0, 0, 0, xs_1, ys_1, 1, -xs_1*ys_1, -yd_1*ys_1, -yd_1])
   
    P = np.array(P)
    [U, S, Vt] = np.linalg.svd(P) #singular value decomposition
    H = Vt[-1].reshape(3, 3) #last column of Vt is the last row of Vt
    H = H/H[-1,-1] #normalize H
    H = np.linalg.inv(H) #invert H
    return H



# detect aruco markers in an input image
def detect_image_aruco(template_image, input_image_raw, output_path):

    h_points_template_image = []
    h_points_input_image = []
    
    input_image = cv2.imread(input_image_raw)
    input_image_copy = cv2.imread(input_image_raw)
    template_image = cv2.imread(template_image, cv2.IMREAD_COLOR)
 
    # get the points from the input image
    corners, ids_input, rejected_points = detect_aruco_markers(input_image)
    if ids_input is not None:
        for i in range(len(ids_input)):
            for j in range (4):
                h_points_input_image.append([(corners[i][0][j,0],corners[i][0][j,1]), ids_input[i][0]])
  
        # get the points from the template image
        corners, ids, rejected_points = detect_aruco_markers(template_image)
        for i in range(len(ids)):
            for j in range (4):
                if ids[i][0] in ids_input:
                    h_points_template_image.append([(corners[i][0][j,0],corners[i][0][j,1]), ids[i][0]])

        # sort the points
        h_points_input_image_sorted = sorted(h_points_input_image, key=lambda x: x[1])
        h_points_template_image_sorted = sorted(h_points_template_image, key=lambda x: x[1])

        # compute the homography
        if len(h_points_input_image_sorted) >= 4:
            H = compute_homography(h_points_input_image_sorted, h_points_template_image_sorted) #compute homography
            # warp the input image
            if H is not None:
                final_img = cv2.warpPerspective(input_image_copy, H, template_image.shape[1::-1])
                name_of_image = os.path.basename(input_image_raw)
                name_of_image = name_of_image
                output_path = output_path + "/" + name_of_image
                print("image: {} ===> {} ".format(input_image_raw, output_path))    
                print("===========================================================")
                cv2.imwrite(output_path, final_img)



###############################################################
#
#   
#                 DETECT IMPORTANT POINTS IMAGES
#
#
###############################################################


def get_binary_image(image):
    # image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
    thresh, image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
    # cv2.imshow('image', im_bw)
    # cv2.waitKey(0)
    return image
    

# function for initializing input images
def image_init(input1, input2, rows, cols):
    img1 = cv2.imread(input1)
    img2 = cv2.imread(input2)

    img1= cv2.resize(img1, (cols, rows))
    img2= cv2.resize(img2, (cols , rows))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    return gray1, gray2

# function to detect key points in an image
def key_point_detector(image):
    img = image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv2.drawKeypoints(gray,kp,img)
    cv2.imshow('image', img)
    cv2.waitKey(0)


def adaptative_gaussian_thres(image):
    th3 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,20)
    return th3

# function to detect edges in an image
def canny_edge_detector(image):
    edges = cv2.Canny(image,240,400,L2gradient=2)
    return edges

# get the countours of the paper
def get_contours(image):
    img = image
    ret, thresh = cv2.threshold(image, 127, 255, 0)
    contours,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0,255,0), 3)
    cv2.imshow('image', img)
    cv2.waitKey(0)

    pass

# function to find matches between keypoints of two images
def find_matches(img1, img2):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()

    matches = bf.match(des1, des2)
    matches = sorted(matches,key=lambda x:x.distance)

    return kp1, kp2, matches


def visualize_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)


def median_filter(image):
    image = cv2.medianBlur(image,5)
    return image


def gaussian_bluring(image):
    image = cv2.GaussianBlur(image,(5,5),0)
    return image

def averaging_filter(image):
    kernel = np.ones((3,3),np.float32)/10
    image = cv2.filter2D(image,-1,kernel)
    return image

def detect_corners(image):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)

    img[dst > 0.01 * dst.max()] = [0,0,255]

    return img


def save_image(input_image_raw, output_path, image_to_save):
    name_of_image = os.path.basename(input_image_raw)
    name_of_image = name_of_image
    output_path = output_path + "/" + name_of_image
    print("image: {} ===> {} ".format(input_image_raw, output_path))    
    print("===========================================================")
    cv2.imwrite(output_path, image_to_save)


def draw_rectangle(image):
    ret,thresh = cv2.threshold(image,127,255,0)
    im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]
    M = cv2.moments(cnt)
    print( M )

    area = cv2.contourArea(cnt)

    perimeter = cv2.arcLength(cnt,True)

    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    epsilon = 0.1*cv2.arcLength(cnt,True)
    pprox = cv2.approxPolyDP(cnt,epsilon,True)

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image,[box],0,(0,0,255),2)

def harris_corner_detector(image):

    thresh = 200
    # Detector parameters
    blockSize = 2
    apertureSize = 3
    k = 0.04
    # Detecting corners
    dst = cv2.cornerHarris(image, blockSize, apertureSize, k)
    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    dst_norm_scaled = cv2.convertScaleAbs(dst_norm)
    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i,j]) > thresh:
                cv2.circle(dst_norm_scaled, (j,i), 5, (255), 2)

    return dst_norm_scaled


def orb_detector(img1, img2):

    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    visualize_image(img3)


# function to draw image with matching key points
def draw_matches(img1, img2, kp1, kp2, matches, rows, cols,output_path):
    # Initialize output image
    out = np.zeros((rows, 2*cols, 3), dtype='uint8')

    # Place the first image to the left
    out[:rows,:cols] = np.dstack([img1])

    # Place the next image to the right of it
    out[:rows,cols:] = np.dstack([img2])


    img1_pt = np.zeros((len(matches), 2), dtype=np.float32)
    img2_pt = np.zeros((len(matches), 2), dtype=np.float32)

    for i, mat in enumerate(matches):
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        img1_pt[i, :] = kp1[mat.queryIdx].pt
        img2_pt[i, :] = kp2[mat.trainIdx].pt

        # filter matches points
        cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2) + cols, int(y2)), 4, (0, 255, 0), 1)

        cv2.line(out, (int(x1), int(y1)), (int(x2) + cols, int(y2)), (0, 0, 255), 1)


    # remove points from image 2 which are out of the paper
    # detect corners of the paper
    # know if a point is inside the area of the paper
    # remove correspondant points from image 1

    # compute the homgraphy
    H, mask = cv2.findHomography(img2_pt, img1_pt, cv2.RANSAC)
    
    if H is not None:
        final_img = cv2.warpPerspective(img2, H, (cols, rows)) # warp the input image
        save_image(input_image_raw, output_path, final_img) #save the new image on the output folder

    # cv2.imshow('output', out)
    # cv2.waitKey(0)


###############################################################
#
#   
#                          MAIN PROGRAM
#
#
###############################################################


# ----- EXECTUION AS THE DELIVER SPECIFICATIONS ----

if len(sys.argv) == 5:
    
    task = sys.argv[1]
    template_image = sys.argv[2]
    output_path = sys.argv[3]
    input_folder = sys.argv[4]

    if task == '1':
        # check if the output directory exists
        if os.path.isdir(output_path) == False:
            os.mkdir(output_path) #create the output directory

        # COMPUTE THE HOMOGRAPHY
        for image in os.listdir(input_folder):
            input_image_raw = input_folder + "/" + image     
            source_img = detect_image_aruco(template_image, input_image_raw, output_path)

    if task == '2':
        if os.path.isdir(output_path) == False:
            os.mkdir(output_path) #create the output directory

        print("--> task 2")
        for image in os.listdir(input_folder):
            input_image_raw = input_folder + "/" + image     
            
            # dimension of images
            rows = 1000
            cols = 600

            # initialize the images
            img1_init, img2_init = image_init(template_image, input_image_raw, rows, cols)

            # draw matches
            kp1, kp2, matches = find_matches(img1_init, img2_init)
            draw_matches(img1_init, img2_init, kp1, kp2, matches, rows, cols, output_path)


elif len(sys.argv) == 6:
    print("task 4")

else:
    print("ERROR: wrong number of arguments")
    sys.exit(1) 