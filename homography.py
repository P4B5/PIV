import cv2
import numpy as np
# import matplotlib as plt
import sys
import matplotlib.pyplot as plt
# from datasets import templates


print("This is a Simple Scanner")
print("OpenCV version used: {}".format(cv2.__version__))

# show image from a file
def show_file_image(img, title):
    image = cv2.imread(img,0)
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
        print(image)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit

    cv2.destroyAllWindows()
        
    return None

# 2. Compute the homography

def compute_homography(source_image, destination_image):

    H = cv2.findHomography(source_image, destination_image)
    print(H)

    '''
    HOMOGRAPHY MATRIX

    xd        xs      
    yd  = H * ys
    1         1
    
    
    '''


    return None #return the final image after apply the homography



# Mat H = findHomography(objectPointsPlanar, imagePoints);






# 3. Get the Arucu markers






# 4. Get the positions of the markers in the new image








###############################################################
#
#   
#                          MAIN PROGRAM
#
#
###############################################################



CAMERA_MODE = 2

try:
    
    # get the source image
    source_image = 0
    
    if CAMERA_MODE == 1:
        source_img = show_live_image()
    elif CAMERA_MODE == 2:
        source_img ='/home/pabs/PIV/datasets/InitialDataset/templates/template1_manyArucos.png'
        source_img = show_file_image(source_img, "Source image")
    else:
        print("Invalid camera mode")


    # get the destination image
    image_resolution = source_img.shape
    destination_image = np.zeros(image_resolution)

    print(source_img)
    print("Image resolution: {}".format(image_resolution))
    show_file_image(destination_image, "Destination image")


    # compute the homography
    compute_homography(source_img, destination_image)


except:
  print("An exception occurred")
 