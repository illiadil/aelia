from numba import jit, njit
import numpy as np
import cv2
import math

@njit(cache=False)
def lnir(Mask,W):
    pos_i=0
    pos_j=0
    h = 1
    for i in range(0,Mask.shape[1]-W):
        try:
            j = list(Mask[i,:]).index(1)
        except:
            continue
        iterations = 0
        while j < Mask.shape[0]-h: # j is x
            for h_iteration in range(h,Mask.shape[0]-j):
                rect = Mask[i:i+W,j:j+h_iteration]
                if np.sum(rect) == W*h_iteration:
                    pos_i, pos_j = i,j
                    h = h_iteration
                else:
                    j+=h
                    break
            iterations=iterations+1
            if iterations > Mask.shape[0]:
                break
    shape = [(pos_j,pos_i), (pos_j+h, pos_i+W)]
    return shape
    
@njit(cache=False)
def coordinate_projection(point,alpha):
    x, y = point[0] , point[1]
    alpha_rad = math.radians(alpha)
    a = x*math.cos(alpha_rad)-y*math.sin(alpha_rad)
    b = x*math.sin(alpha_rad)+y*math.cos(alpha_rad)
    return ( int(a), int(b) )

def rotate(rotateImage, angle): 
	# Taking image height and width 
	imgHeight, imgWidth = rotateImage.shape[0], rotateImage.shape[1] 

	# Computing the centre x,y coordinates 
	# of an image 
	centreY, centreX = imgHeight//2, imgWidth//2

	# Computing 2D rotation Matrix to rotate an image 
	rotationMatrix = cv2.getRotationMatrix2D((centreY, centreX), angle, 1.0) 

	# Now will take out sin and cos values from rotationMatrix 
	# Also used numpy absolute function to make positive value 
	cosofRotationMatrix = np.abs(rotationMatrix[0][0]) 
	sinofRotationMatrix = np.abs(rotationMatrix[0][1]) 

	# Now will compute new height & width of 
	# an image so that we can use it in 
	# warpAffine function to prevent cropping of image sides 
	newImageHeight = int((imgHeight * sinofRotationMatrix) +
						(imgWidth * cosofRotationMatrix)) 
	newImageWidth = int((imgHeight * cosofRotationMatrix) +
						(imgWidth * sinofRotationMatrix)) 

	# After computing the new height & width of an image 
	# we also need to update the values of rotation matrix 
	rotationMatrix[0][2] += (newImageWidth/2) - centreX 
	rotationMatrix[1][2] += (newImageHeight/2) - centreY 

	# Now, we will perform actual image rotation 
	rotatingimage = cv2.warpAffine( 
		rotateImage.astype(np.uint8), rotationMatrix, (newImageWidth, newImageHeight)) 

	return rotatingimage 

def check_all_angles(image,angle_step,W):
    #rotation angle in degree
    angle_with_max = 0
    max_h = -1
    rectangle_for_max = [0,0,0,0]
    for angle in range(0,180,angle_step):
        rotated_img = rotate(image.astype(int),angle)
        rectangle = lnir(rotated_img, W)
        h = rectangle[1][0]-rectangle[0][0]
        if h > max_h:
            max_h = h
            angle_with_max = angle
            rectangle_for_max = rectangle
    rotated_img = rotate(image,angle_with_max)
    offset_projection = math.floor(rotated_img.shape[0]/2)
    rectangle_for_max = [(x-offset_projection,offset_projection-y) for (x,y) in rectangle_for_max]
    points = [rectangle_for_max[0],(rectangle_for_max[0][0],rectangle_for_max[1][1]),rectangle_for_max[1],(rectangle_for_max[1][0],rectangle_for_max[0][1])]
    points_projected = [coordinate_projection(points[i],-1*angle_with_max) for i in range(4)]
    offset_original = math.floor(image.shape[0]/2)
    return [(x+offset_original,offset_original-y) for (x,y) in points_projected]