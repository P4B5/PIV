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

def compute_homography(h_points_live_image, h_points_template_image):

    '''
    HOMOGRAPHY MATRIX

    xd        xs      
    yd  = H * ys
    1         1
    '''

    print("h_points_input_image: {}".format(h_points_live_image))  
    print("h_points_template_image: {}".format(h_points_template_image))  

    # get the points from the input image
    xs_1 = h_points_live_image[0][0][0]
    xs_2 = h_points_live_image[1][0][0]
    xs_3 = h_points_live_image[2][0][0]
    xs_4 = h_points_live_image[3][0][0]
    ys_1 = h_points_live_image[0][0][1]
    ys_2 = h_points_live_image[1][0][1]
    ys_3 = h_points_live_image[2][0][1]
    ys_4 = h_points_live_image[3][0][1]

    # get the points from the template image
    xd_1 = h_points_template_image[0][0][0]
    xd_2 = h_points_template_image[1][0][0]
    xd_3 = h_points_template_image[2][0][0]
    xd_4 = h_points_template_image[3][0][0]
    yd_1 = h_points_template_image[0][0][1]
    yd_2 = h_points_template_image[1][0][1]
    yd_3 = h_points_template_image[2][0][1]
    yd_4 = h_points_template_image[3][0][1]

    P = np.array([
        [xs_1, ys_1, 1, 0, 0, 0, -xd_1*xs_1, -ys_1*xd_1, -xd_1],
        [0, 0, 0, xs_1, ys_1, 1, -xs_1*ys_1, -yd_1*ys_1, -yd_1],
        [xs_2, ys_2, 1, 0, 0, 0, -xd_2*xs_2, -ys_2*xd_2, -xd_2],
        [0, 0, 0, xs_2, ys_2, 1, -xs_2*ys_2, -yd_2*ys_2, -yd_2],
        [xs_3, ys_3, 1, 0, 0, 0, -xd_3*xs_3, -ys_3*xd_3, -xd_3],
        [0, 0, 0, xs_3, ys_3, 1, -xs_3*ys_3, -yd_3*ys_3, -yd_3],
        [xs_4, ys_4, 1, 0, 0, 0, -xd_4*xs_4, -ys_4*xd_4, -xd_4],
        [0, 0, 0, xs_4, ys_4, 1, -xs_4*ys_4, -yd_4*ys_4, -yd_4]
        ])

    
    [U, S, Vt] = np.linalg.svd(P)
    print(Vt)
    H = Vt[-1].reshape(3, 3)
    H = H/H[-1,-1]
    # H = np.linalg.inv(H)

    # H =[[0.322458225645459,	10.5437738227079,	-2943.68591951408],
    #     [-4.16259475079465,	3.99892442704985,	4310.31501556172],
    #     [-9.69860740545962e-05,	0.00199300996332708,	1]]
   
    return H




    # if len(h_points_live_image) == len(h_points_sample_image):
    #     # get the points to compute the homography
    #     xs = []
    #     ys = []
    #     xd = []
    #     yd = []
    #     for i in range(len(h_points_live_image)):
    #         xd.append(h_points_live_image[i][0][0])
    #         yd.append(h_points_live_image[i][0][1])
    #         xs.append(h_points_sample_image[i][0][0])
    #         ys.append(h_points_sample_image[i][0][1])

    #     # compute the homography
    #     xs = np.array(xs)
    #     ys = np.array(ys)
    #     xd = np.array(xd)
    #     yd = np.array(yd)
    #     H = np.array([[np.sum(xs*xd), np.sum(xs*yd), np.sum(xs)], [np.sum(ys*xd), np.sum(ys*yd), np.sum(ys)], [np.sum(xd), np.sum(yd), len(xs)]])
    #     H = np.linalg.inv(H)
    #     H = H/H[2,2]

    #     # print(H)
    #     # print("H: {}".format(H))


    # #     # H = cv2.findHomography(source_image, destination_image)
    #     return  H #return the final image after apply the homography

    # else:
    #     return None


# GET THE HOMOGRAPHY FROM A FIXED IMAGET
def detect_aruco_image():

    # 1. get the sample image
    template_image = '/home/pabs/PIV/datasets/InitialDataset/templates/template2_fewArucos.png'
    input_image_raw = '/home/pabs/PIV/datasets/InitialDataset/templates/real_image_2.png'

    # get the points from the template image

    h_points_template_image = []
    h_points_input_image = []


    template_image = cv2.imread(template_image, cv2.IMREAD_COLOR)
    template_image = cv2.resize(template_image, (0,0), fx=0.25, fy=0.25)


    corners, ids, rejected_points = detect_aruco_markers(template_image)
    print("corners: {}".format(corners))
    for i in range(len(ids)):
            h_points_template_image.append([(corners[i][0][0,0],corners[i][0][0,1]), ids[0][0]])


    print(h_points_template_image)

    input_image = cv2.imread(input_image_raw, cv2.IMREAD_COLOR)
    input_image_raw = cv2.imread(input_image_raw, cv2.IMREAD_COLOR)
    input_image = cv2.resize(input_image, (0,0), fx=0.5, fy=0.5)
    input_image_raw = cv2.resize(input_image_raw, (0,0), fx=0.5, fy=0.5)
  
    corners, ids, rejected_points = detect_aruco_markers(input_image)
        
    # get the key points to compute the homography
    if ids is not None:
        for i in range(len(ids)):
            h_points_input_image.append([(corners[i][0][0,0],corners[i][0][0,1]), ids[i][0]])

        # h_points_sample_image = dete
   
    H = compute_homography(h_points_input_image, h_points_template_image) #compute homography
    print("H: {}".format(H))

    # return None

    # get the new image
    # print("image shape: {}".format(image.shape))
    # print("sample_image.shape: {}".format(sample_image.shape))
    blank_image = np.zeros(template_image.shape, np.uint8)

    # if (p.x < Xmin or p.x > Xmax or p.y < Ymin or p.y > Ymax):
    #     return None 
    


 
    if H is not None:
        for i in range(input_image_raw.shape[0]):
            for j in range(input_image_raw.shape[1]):
                p = np.array([j, i, 1])
                point = np.dot(H, p)
                # point = H @ np.array([i,  j, 1]).transpose()
                point = point/point[-1]
                point = point.astype(int)
                if point[0] >= 0 and point[0] < blank_image.shape[0] and point[1] >= 0 and point[1] < blank_image.shape[1]:
                    print("coordinates after apply H {} --->> {}".format((j,i),point))
                    blank_image[point[0],point[1]] = input_image_raw[j, i]
           

    cv2.imshow("input image copy", blank_image)
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

        print("image shape: {}".format(image.shape))
        print("sample_image.shape: {}".format(sample_image.shape))
        blank_image = np.zeros(image.shape, np.uint8)

        if H is not None:
            for i in range(blank_image.shape[0]):
                for j in range(blank_image.shape[1]):
                    print("i: {}, j: {}".format(i,j))
                    x = np.array([j,i,1])
                    x = np.dot(H,x)
                    x = x/x[2]
                    x = x.astype(int)
                    blank_image[i,j] = image[x[1],x[0]]

        cv2.imshow("New image", blank_image)

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
 