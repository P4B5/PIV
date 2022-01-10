
import sys
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
        xd_1 = h_points_input_image[i][0][0]
        yd_1 = h_points_input_image[i][0][1]
        xs_1 = h_points_template_image[i][0][0]
        ys_1 = h_points_template_image[i][0][1]
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
#                          MAIN PROGRAM
#
#
###############################################################


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
        print("task 2")
    if task == '3':
        print("task 3")
   
elif len(sys.argv) == 6:
    print("task 4")

else:
    print("ERROR: wrong number of arguments")
    sys.exit(1) 






