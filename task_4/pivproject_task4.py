import sys
from tkinter import image_names
from tkinter.tix import InputOnly
from typing import final
from webbrowser import get
import cv2
import cv2.aruco as aruco
import os
import numpy as np
import matplotlib.pyplot as plt



'''

--- GROUP NUMBER 13 ---

Jonas Rogde Jørgensen           -- ist1101351
Afonso dos Santos Caetano Dias  -- ist87139
Pablo Castellanos Lopez         -- ist1101077

PIV PROJECT 2021/22

execution: 
python pivproject2021.py <task> <path_to_template> <path_to_output_folder> <path_to_input_folder>
python pivproject2021.py 4 <path_to_template>  <path_to_output_folder>  <input_folder_1> <input_folder_2>

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
                save_image(final_img, output_path)



###############################################################
#
#   
#                 DETECT IMPORTANT POINTS IMAGES
#
#
###############################################################


def save_image(input_image_raw, output_path, image_to_save):
    name_of_image = os.path.basename(input_image_raw)
    name_of_image = name_of_image
    output_path = output_path + "/" + name_of_image
    print("image: {} ===> {} ".format(input_image_raw, output_path))    
    print("===========================================================")
    cv2.imwrite(output_path, image_to_save)
    

def visualize_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)


# function for initializing input images
def image_init(input1, input2, rows, cols):
    img1 = cv2.imread(input1)
    img2 = cv2.imread(input2)

    img1= cv2.resize(img1, (cols, rows))
    img2= cv2.resize(img2, (cols , rows))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    return gray1, gray2

# initialize two imput images + template
def image_init_2(template, input1, input2, rows, cols):
    img1 = cv2.imread(input1)
    img2 = cv2.imread(input2)
    tmp = cv2.imread(template)

    img1= cv2.resize(img1, (cols, rows))
    img2= cv2.resize(img2, (cols , rows))
    tmp = cv2.resize(tmp, (cols , rows))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    # return tmp, img1, img2
    return tmp, gray1, gray2



# function to find matches between keypoints of two images
def find_matches(img1, img2):
    sift = cv2.SIFT_create(nfeatures=300, contrastThreshold=0.1, edgeThreshold=0.01, sigma=3)

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)

    matches = sorted(matches,key=lambda x:x.distance)

    return kp1, kp2, matches




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

    cv2.imshow('output', out)
    cv2.waitKey(0)
    return img1_pt, img2_pt





def orb_detector(img1, img2):

    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=400, edgeThreshold=1)
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

    print(kp1)
    print(kp2)
    print(matches)

    img1_pt = np.zeros((len(matches), 2), dtype=np.float32)
    img2_pt = np.zeros((len(matches), 2), dtype=np.float32)

    for i, mat in enumerate(matches):
        img1_pt[i, :] = kp1[mat.queryIdx].pt
        img2_pt[i, :] = kp2[mat.trainIdx].pt

    print(img1_pt)
    print(img2_pt)

    return img1_pt, img2_pt
    # return kp1, kp2, matches



def get_homography(img1_pt, img2_pt, image,rows, cols):
    H, mask = cv2.findHomography(img2_pt, img1_pt, cv2.RANSAC)

    print(mask)

    if H is not None:
        final_img = cv2.warpPerspective(image, H, (cols, rows)) # warp the input image
        # cv2.imshow('final_img', final_img)
        # cv2.waitKey(0)
        return final_img
        # save_image(input_image_raw, output_path, final_img) #save the new image on the output folder

    return None
  
def MSE_compare(img1, img2, rows, cols):
    MSE = 0
    for i in range(rows):
        for j in range(cols):
            MSE = MSE + abs(img1[i][j] - img2[i][j])
    return MSE


def MSE_compare2(img1, img2, rows, cols):
    MSE = 0
    for i in range(rows-1):
        for j in range(cols-1):
            MSE = MSE + abs((img1[i][j] + img1[i+1][j] + img1[i][j+1] + img1[i+1][j+1]) - (img2[i][j] + img2[i+1][j] + img2[i][j+1] + img2[i+1][j+1]))
    return MSE


###############################################################
#
#   
#                          MAIN PROGRAM
#
#
###############################################################


# ----- EXECTUION AS THE DELIVER SPECIFICATIONS ----


task = sys.argv[1]           # get the task from the command line
template_image = sys.argv[2] # get the template image from the command line
output_path = sys.argv[3]    # get the output path from the command line


if len(sys.argv) == 5:
    input_folder = sys.argv[4]   # get the input folder from the command line
    if task == '1':
        print("--> task 2")
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
            rows = 1200
            cols = 800

            # initialize the images
            img1_init, img2_init = image_init(template_image, input_image_raw, rows, cols)

            # draw matches
            kp1, kp2, matches = find_matches(img1_init, img2_init)
            draw_matches(img1_init, img2_init, kp1, kp2, matches, rows, cols, output_path)


elif len(sys.argv) == 6:
    print("--> task 4")
    input_folder_1 = sys.argv[4]   # get the input folder from the command line
    input_folder_2 = sys.argv[5]   # get the input folder from the command line

    print("task: ", task)
    print("template image: ", template_image)
    print("output_path: ", output_path)
    print("input folder 1: ", input_folder_1)
    print("input folder 2: ", input_folder_2)

    # python pivproject2021.py 4 /home/pabs/PIV/task_4/TwoCameras/ulisboatemplate.jpg /home/pabs/PIV/task_4/TwoCameras/ulisboa2/output  /home/pabs/PIV/task_4/TwoCameras/ulisboa2/phone2 /home/pabs/PIV/task_4/TwoCameras/ulisboa2/photo2
    # python pivproject2021.py 4 /home/pabs/PIV/task_4/GoogleGlass/template_glass.jpg /home/pabs/PIV/task_4/TwoCameras/ulisboa2/output  /home/pabs/PIV/task_4/GoogleGlass/glass /home/pabs/PIV/task_4/GoogleGlass/nexus
    # python pivproject2021.py <task> <path_to_template> <path_to_output_folder> <path_to_input_folder>
    # python pivproject_task4.py 4 /home/pabs/PIV/task_4/TwoCameras/ulisboatemplate.jpg /home/pabs/PIV/task_4/TwoCameras/ulisboa1/output  /home/pabs/PIV/task_4/TwoCameras/ulisboa1/phone /home/pabs/PIV/task_4/TwoCameras/ulisboa1/photo
    
    
    # create the output directory
    if os.path.isdir(output_path) == False:
        os.mkdir(output_path) #create the output directory

    image_lst_1 = []
    image_lst_2 = []

    for image1 in os.listdir(input_folder_1):
        image_lst_1.append(input_folder_1 + "/" + image1)

    for image2 in os.listdir(input_folder_2):
        image_lst_2.append(input_folder_2 + "/" + image2)  

    image_lst_1.sort()
    image_lst_2.sort()

    dataset_len = 0
    if (len(image_lst_1) <= len(image_lst_2)):
        dataset_len = len(image_lst_1)
    elif(len(image_lst_1) > len(image_lst_2)):
        dataset_len = len(image_lst_2)

    for i in range(dataset_len):
        # dimension of images
        rows = 800
        cols = 400

        # initialize the images
        template, img1_init, img2_init = image_init_2(template_image, image_lst_1[i],image_lst_2[i] , rows, cols)

     
        img1_pt, img2_pt = orb_detector(img1_init,img2_init)
        h_image = get_homography(img1_pt, img2_pt, img2_init, rows, cols)
        visualize_image(h_image)

        # sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # sharpen = cv2.filter2D(h_image, -1, sharpen_kernel)
        # visualize_image(sharpen)

        kp1_1, kp2_1, matches_1 = find_matches(template, h_image)
        img1_pt, img2_pt = draw_matches(template,h_image, kp1_1, kp2_1, matches_1, rows, cols, output_path)
        h_image = get_homography(img1_pt, img2_pt, h_image,rows, cols)
        visualize_image(h_image)

else:
    print("ERROR: wrong number of arguments or arguments format")
    sys.exit(1) 
