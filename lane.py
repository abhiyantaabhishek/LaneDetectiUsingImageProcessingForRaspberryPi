import cv2
import numpy as np
import RPi.GPIO as GPIO

from time import sleep
from picamera import PiCamera
from picamera.array import PiRGBArray
import time

#def setup():

Forward=14
Backward=15
sleeptime=1
enable=18

Forward1=3
Backward1=4
enable1=12

GPIO.setmode(GPIO.BCM)
GPIO.setup(Forward, GPIO.OUT)
GPIO.setup(Backward, GPIO.OUT)
GPIO.setup(Forward1, GPIO.OUT)
GPIO.setup(Backward1, GPIO.OUT)


GPIO.setup(2,GPIO.OUT)
pwm=GPIO.PWM(2,50)
#pwm=GPIO.PWM(3,100)
pwm.start(0)
GPIO.setup(enable, GPIO.OUT)
my_pwm=GPIO.PWM(enable,1000)
my_pwm.start(0)
GPIO.setup(enable1, GPIO.OUT)
my_pwm1=GPIO.PWM(enable1,1000)
my_pwm1.start(0)

def birdeye(img):

    h, w = img.shape[:2]
	
    src = np.float32([[w, h-10],    # br
                      [0, h-10],    # bl
                      [0, h*2//3],   # tl
                      [w, h*2//3]])  # tr
    dst = np.float32([[w, h],       # br
                      [0, h],       # bl
                      [0, 0],       # tl
                      [w, 0]])      # tr

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    return warped, M, Minv
        
def ROI(img):
    height, width= img.shape
    vertices = np.array([[(0, height),
                              (0, height*2/3),
                              (width-50, height*2/3),
                              (width , height)]],
                            dtype=np.int32)
    #line_img = np.zeros(shape=(img_h, img_w))
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
    cv2.imshow("mask",img)


def forward():
    GPIO.output(Forward, GPIO.HIGH)
    GPIO.output(Forward1, GPIO.HIGH)



def change(angle):
    pwm.ChangeDutyCycle(angle)
    #time.sleep(1)
    
def nothing(x):
    pass

# Create a black image, a window
#img = np.zeros((300,512,3), np.uint8)

#im =cv2.imread("data//test_images//solidWhiteCurve.jpg")
#cap = cv2.VideoCapture("data//test_videos//solidWhiteRight.mp4")
#img_h, img_w = im[0].shape[0], im[0].shape[1]
cv2.namedWindow('image')
cv2.createTrackbar('v1','image',8,255,nothing)
cv2.createTrackbar('v2','image',8,255,nothing)
cv2.createTrackbar('v3','image',0,255,nothing)
cv2.createTrackbar('threshold1','image',138,255,nothing)
cv2.createTrackbar('threshold2','image',220,255,nothing)
cv2.createTrackbar('rho','image',1,255,nothing)
cv2.createTrackbar('theta','image',1,255,nothing)
cv2.createTrackbar('threshold_h','image',50,255,nothing)
cv2.createTrackbar('min_line_len','image',15,255,nothing)
cv2.createTrackbar('max_line_gap','image',5,255,nothing)
cv2.createTrackbar('thickness','image',5,255,nothing)

def display_image(im):
	err=0;
	v1 = cv2.getTrackbarPos('v1','image')
	v2 = cv2.getTrackbarPos('v2','image')
	v3 = cv2.getTrackbarPos('v3','image')
	threshold1 = cv2.getTrackbarPos('threshold1', 'image')
	threshold2 = cv2.getTrackbarPos('threshold2', 'image')
	rho = cv2.getTrackbarPos('rho', 'image')
	theta = cv2.getTrackbarPos('theta', 'image')
	threshold_h = cv2.getTrackbarPos('threshold_h', 'image')
	min_line_len = cv2.getTrackbarPos('min_line_len', 'image')
	max_line_gap = cv2.getTrackbarPos('max_line_gap', 'image')
	thickness = cv2.getTrackbarPos('thickness', 'image')
	
	img_h, img_w = im[0].shape[0], im[0].shape[1]
	height, width, channels = im.shape
	##    cv2.imshow('im1',im)
	gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	##    cv2.imshow('im2', gray)
	blur = cv2.GaussianBlur(gray,(2*v1+1,2*v2+1),v3)
	##    cv2.imshow('im3', blur)
	edge = cv2.Canny(blur,threshold1=threshold1, threshold2=threshold2)
	cv2.imshow('im4', edge)
	edge=ROI(edge)
	
	cv2.imshow('edges',edge)
	birdeye_binary, M, Minv = birdeye(edge)
	#birdeye_binary= edge
	cv2.imshow('birdeye_binary',birdeye_binary)
	n_windows = 9
	height, width = birdeye_binary.shape
	
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(birdeye_binary[height*2 // 3:, :], axis=0)
	
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((birdeye_binary, birdeye_binary, birdeye_binary)) * 255  
	
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = len(histogram) // 2
	leftx_base = np.argmax(histogram[:midpoint]) #left portion     #  np.argmax: Returns the indices of the maximum values alon
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint      #right prtion for the correct indiced midpoint is added
	
	# Set height of windows
	window_height = np.int(height / n_windows)
	
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = birdeye_binary.nonzero()
	nonzero_y = np.array(nonzero[0]) #it ll store all the non zero values
	nonzero_x = np.array(nonzero[1])
	
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	
	margin =  50 # width of the windows +/- margin
	minpix = 100  # minimum number of pixels found to recenter window
	
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []
	
	# Step through the windows one by one
	for window in range(n_windows):
	# Identify window boundaries in x and y (and right and left)
		win_y_low = height - (window + 1) * window_height #window measurments
		win_y_high = height - window * window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		
		# Draw the windows on the visualization image
		cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
		cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
		
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) #those non zero if come under the ramges
		                  & (nonzero_x < win_xleft_high)).nonzero()[0]

		good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low)
	                   & (nonzero_x < win_xright_high)).nonzero()[0]
	
	# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
		    leftx_current = np.int(np.mean(nonzero_x[good_left_inds]))
		if len(good_right_inds) > minpix:
		    rightx_current = np.int(np.mean(nonzero_x[good_right_inds]))
		try:
			if len(good_left_inds) <100:
			   leftx_current != np.int(np.mean(nonzero_x[good_left_inds]))
		except:
        
        		print ("non")
		try:
			if len(good_right_inds) <100:
		    	   rightx_current != np.int(np.mean(nonzero_x[good_right_inds]))
		except:
        
        		print ("non")	
# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)
	
	
	
	
	# Extract left and right line pixel positions
	line_lt_all_x, line_lt_all_y = nonzero_x[left_lane_inds], nonzero_y[left_lane_inds]
	line_rt_all_x, line_rt_all_y = nonzero_x[right_lane_inds], nonzero_y[right_lane_inds]
	#left_fit_pixel = np.polyfit(line_lt_all_y, line_lt_all_x, 2)
	#right_fit_pixel = np.polyfit(line_rt_all_y, line_rt_all_x, 2)
	try:
		left_fit_pixel = np.polyfit(line_lt_all_y, line_lt_all_x, 2)
		right_fit_pixel = np.polyfit(line_rt_all_y, line_rt_all_x, 2)
		print(left_fit_pixel)
	
		target = (left_fit_pixel+right_fit_pixel)/2

		#target = (left_fit_pixel+right_fit_pixel)/2
		ploty = np.linspace(0, height-1, height)
		#left_fitx = left_fit_pixel[0] * ploty ** 2 + left_fit_pixel[1] * ploty + left_fit_pixel[2]
		#right_fitx = right_fit_pixel[0] * ploty ** 2 + right_fit_pixel[1] * ploty + right_fit_pixel[2]
		left_fitx = np.polyval(left_fit_pixel, ploty)              #putting the value of y we get x value for left side
		right_fitx = np.polyval(right_fit_pixel, ploty)
		tagret_fitx = np.polyval(target, ploty)
		out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [ 0, 0,255]
		out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]
		#target_x_y=
		for point in set(zip(left_fitx,ploty)):
		    cv2.circle(out_img,(int(point[0]),int(point[1])), 2, (0,0,255), -1)
		for point in set(zip(right_fitx,ploty)):
		    cv2.circle(out_img,(int(point[0]),int(point[1])), 2, (0,0,255), -1)
		for point in set(zip(tagret_fitx,ploty)):
	    	    cv2.circle(out_img,(int(point[0]),int(point[1])), 2, (0,0,255), -1)

		for point in set(zip(tagret_fitx,ploty)):
		    if(int(point[1])==height-50):
		        cv2.circle(out_img,(int(point[0]),int(point[1])), 10, (0,255,0), -1)
		cv2.circle(out_img,(width//2,height-50), 10, (0,255,0), -1)
		err = width//2-int(point[0])
		
	except:
		print ("nothing")	

	
	dewarped_out_img = cv2.warpPerspective(out_img, Minv, (width, height))

	blend_im = cv2.addWeighted(src1=dewarped_out_img, alpha=0.8, src2=im, beta=0.5, gamma=0.)
	
	cv2.imshow('out_img', out_img)
	
	cv2.imshow('im5', blend_im)
	cv2.imshow('dewarped_out_img', dewarped_out_img)
 

	return err;
  
def change_slow(desired_angle,present_angle):
    
    x=float(present_angle)
    if x<desired_angle:
       
        while(x<=desired_angle):
            
            x+=0.1
            change(x)
        
            #print "i", x
##            change(x)
            #pwm.ChangeDutyCycle(desired_angle)
            sleep(0.01)
            
    
    elif x>desired_angle:
        
        while(x>=desired_angle):
            x-=0.1
##            change(x)
        
            #print "i", x
            change(x)
            #pwm.ChangeDutyCycle(desired_angle)
            sleep(0.01)
               
    return desired_angle 

def err_generator(err,present_angle ):
    print "called"
    print "err",err
    if err>10 :
        desired_angle=float((110/18)+2)
        print "120"
        present_angle=change_slow(desired_angle,present_angle)
	my_pwm1.ChangeDutyCycle(55)
	my_pwm.ChangeDutyCycle(70)
   
    if err<-10:
        desired_angle=float((90/18)+2)
        print "80"
        present_angle=change_slow(desired_angle,present_angle)
	my_pwm1.ChangeDutyCycle(70)
	my_pwm.ChangeDutyCycle(55)

   
    if err>-10 and err<10:
        desired_angle=float((100/18)+2)
        print "100"
        present_angle=change_slow(desired_angle,present_angle)  
	my_pwm1.ChangeDutyCycle(40)
	my_pwm.ChangeDutyCycle(40)  
    return present_angle
    

fast= input("tell me the speed")
my_pwm1.ChangeDutyCycle(fast)
my_pwm.ChangeDutyCycle(fast)
forward()

if __name__=='__main__':
    #setup()
    state ={'left':1,'right':2,'idle':0}
    present_state=state['idle']
    call=0
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))
    present_angle=float((100/18)+2) 
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        im =frame.array
        err=display_image(im)
        if(present_state==state['idle']):
            call=1
            if(err>90):
                present_state=state['right']
            if(err<90):
                present_state=state['left']
        elif(present_state==state['right']):
            if(err<0):
                present_state=state['idle']
        elif(present_state==state['left']):
            if(err>0):
                present_state=state['idle']
            
        
        try:
            #if(call==1):
            present_angle =err_generator(err,present_angle)
                #call=0
            im =frame.array
        except KeyboardInterrupt:
            destroy()
        rawCapture.truncate(0)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    
    
        

    
def destroy():
    GPIO.cleanup()
cv2.destroyAllWindows()    

