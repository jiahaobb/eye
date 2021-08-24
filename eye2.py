import numpy as np 
import math 
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
# from sympy import *
'''
 input: r1, r2, c1, c2 
 
             c1         c2
  -|-----------------------> col
   |         |           |
 r1| - - - - + - *****   |
   |      	 | *       * |
   |      	 |*         *|
   |      	  *         *
   |           *       * 
 r2|- - - - - -  ***** 
   |    
   |              
   v
 row
'''
# output: traced back theta-phi 

# all in unit of ``mm''  

# input parameter s
# eye_info = r1, r2, c1, c2
class eye_env:
	def __init__(self, eye_info, img):
		self.f = 50.0
		self.r1, self.r2, self.c1, self.c2 = eye_info
# cornea radical length 
		self.d = 11.0
		self.d_r = self.d/2

# pixel size 
		self.p = 36 * 24 / 22300000 * 10000

# read image 
		self.img = img

# img size 
		self.img_row = self.img.shape[0]
		self.img_col = self.img.shape[1]

# parameters for ellipse
		self.a = 7.8
		self.b = 7.8
		self.c = 8.028

		self.temp_img = np.zeros([self.img_row, self.img_col])
		self.theta_s = [] 
		self.phi_s = []
		self.scale = 100/180
		self.out = np.zeros([int(180*self.scale), int(360*self.scale), 3])
		self.eye_mask = np.zeros([self.img_row, self.img_col])
		self.r1_img = r1 
		self.r2_img = r2 
		self.c1_img = c1
		self.c2_img = c2 

# shift the center to the center of the image 
		self.r1 = self.r1 - int(self.img_row/2)
		self.r2 = self.r2 - int(self.img_row/2)
		self.c1 = self.c1 - int(self.img_col/2)
		self.c2 = self.c2 - int(self.img_col/2)

# calcualte the center of the eye in the image 
		self.center = (int((self.r1+self.r2)/2), int((self.c1+self.c2)/2))
		self.r_min = (self.r2 - self.r1)/2
		self.r_max = (self.c2 - self.c1)/2

# calculate the real position of the eye to the camera
		self.dy = abs(self.center[0]) * self.d_r / self.r_max
		self.dx = abs(self.center[1]) * self.d_r / self.r_max
		self.dz = self.f * (self.d / self.p)
	def xy2tp(self, x0_, y0_):
		dz = self.dz
		dx = self.dx
		dy = self.dy 
		d_r = self.d_r
		x0 = x0_ / self.r_min * self.d_r
		y0 = y0_ / self.r_min * self.d_r
		z0 = math.sqrt(8.028*8.028 - x0*x0 - y0*y0)

		# record the corresponding pixel value


		# hteta-phi mapping 
		sigma1 = math.sqrt(-4*dz*dz*x0*x0 - 4*dz*dz*y0*y0 + 257.795136*dz*dz - 25882.98205*dz*z0 + 6444.8784*x0*x0 + 6444.8784*y0*y0 + 6444.8784*z0*z0)
		sigma2 = 2 * (dz*dz - 10*dz*z0 + 25*x0*x0 + 25*y0*y0 + 25*z0*z0)

		x_1 = x0 - 5*x0*(-sigma1 - 2*dz*z0 + 10*x0*x0 + 10*y0*y0 + 10*z0*z0)/sigma2
		x_2 = x0 - 5*x0*( sigma1 - 2*dz*z0 + 10*x0*x0 + 10*y0*y0 + 10*z0*z0)/sigma2
		y_1 = y0 - 5*y0*(-sigma1 - 2*dz*z0 + 10*x0*x0 + 10*y0*y0 + 10*z0*z0)/sigma2
		y_2 = y0 - 5*y0*( sigma1 - 2*dz*z0 + 10*x0*x0 + 10*y0*y0 + 10*z0*z0)/sigma2

		temp = sigma1
		sigma1 = 10*x0*x0 + 10*y0*y0 + 10*z0*z0 - 2*dz*z0 - temp 
		sigma2 = 10*x0*x0 + 10*y0*y0 + 10*z0*z0 - 2*dz*z0 + temp 
		sigma3 = 2 * (dz*dz - 10*dz*z0 + 25*x0*x0 + 25*y0*y0 + 25*z0*z0)

		z_1 = z0 + (dz*sigma1)/sigma3 - (5*z0*sigma1)/sigma3
		z_2 = z0 + (dz*sigma2)/sigma3 - (5*z0*sigma2)/sigma3

		x = x_1
		y = y_1 
		z = z_1

		if z < 0:
			x = x_2
			y = y_2
			z = z_2

		y_copy = y
		y = z 
		z = y_copy

		# calculate theta and phi in radian 
		theta_ = math.acos(z/8.028)
		try:
			phi_ = math.atan(y/x)
		except ZeroDivisionError:
			phi_ = 0.0

		if x <= 0 and y <= 0:
			phi_ = phi_ + math.pi
		elif x <= 0 and y >= 0:
			phi_ = phi_ + math.pi
		elif x >= 0 and y <= 0:
			phi_ = phi_ + 2*math.pi

		# transfer to degree measurement
		theta_ = theta_ / math.pi * 180
		if theta_ < 0:
			theta_ = 0
		elif theta_ > 359:
			theta_ = 359

		phi_ = phi_ / math.pi * 180 + 90

		# swap up side down 
		phi_ = 360 - phi_
		theta_ = 180 - theta_
		
		if phi_ < 0:
			phi_ = 0
		elif phi_ > 359:
			phi_ = 359

		# theta_ = math.floor(theta_)
		# phi_ = math.floor(phi_)
		# print(theta_, phi_)
		return theta_, phi_


	def gen_map(self):


		# print estimated distance from camera to eye
		# print("dx: ", self.dx, "mm")
		# print("dy: ", self.dy, "mm")
		# print("dz: ", self.dz/1000, "m")

		# construct the output of theta-phi representation

		# for point within the ellipse
		for row in range(self.r1, self.r2):
			y0_ = row - self.center[0]
			for col in range(self.c1, self.c2):
				# print(row, col)
				r_img = row + int(self.img_row/2)
				c_img = col + int(self.img_col/2)
				pixel = self.img[r_img, c_img, :]
				x0_ = col - self.center[1]
				# print(self.c1, self.c2, self.r1, self.r2)
				# print(self.img.shape)
				# print("self.center[0]: ", self.center[0], "self.center[1]: ", self.center[1], "x0: ", x0_, ", y0: ", y0_)
				val = (y0_*y0_) / (self.r_min*self.r_min) + (x0_*x0_) / (self.r_min*self.r_min)

				# continue if it is not a point within the ellipse
				if val > 1:
					# print(val)
					continue
				
				# compute real location in camera coordinate 
				theta_0, phi_0 = self.xy2tp(x0_-0.5, y0_-0.5)
				theta_1, phi_1 = self.xy2tp(x0_+0.5, y0_+0.5)
				theta_0, phi_0 = math.floor(theta_0), math.floor(phi_0)
				theta_1, phi_1 = math.ceil(theta_1), math.ceil(phi_1)
				self.out[int(theta_0*self.scale):int(theta_1*self.scale), int(phi_0*self.scale):int(phi_1*self.scale), :] = pixel
		return self.out
	# print("theta range(0,180):", min(theta_s), max(theta_s))
	# print("phi range(0,180):", min(phi_s), max(phi_s))

def img2posmap(eye_info, img, threshold = -1):
	r1, r2, c1, c2 = eye_info
	ee = eye_env(eye_info, img)
	eye_map = ee.gen_map()
	if threshold != -1:
		eye_map[np.where(eye_map <  threshold)] = 0
		eye_map[np.where(eye_map >= threshold)] = 1
	return eye_map


if __name__ == '__main__':
	# Evangeline's Mac
	# img = cv2.imread('../../data/0/0/img.jpg')
	# r1, r2, c1, c2 = np.load('../../data/0/0/eye.npy')
	# r1, r2, c1, c2 = int(r1), int(r2), int(c1), int(c2)
	# eye_info = (r1, r2, c1, c2)
	# pos_map = img2posmap(eye_info, img)
	# cv2.imwrite('../../data/temp/pos_map.png', pos_map)

	# correct the data
	# eye_info = np.load('../data/real/10/4/eye.npy')
	# print(eye_info, eye_info.shape, type(eye_info[0]))
	# new = []
	# new.append('1061')
	# new.append('1112')
	# new.append('2778')
	# new.append('2884')
	# new = np.array(new)
	# print(new, new.shape, type(new[0]))
	# np.save('../data/real/1/3/eye.npy', new)


	# Cluster
	real_data_path = '../data/real'
	for scene_name in os.listdir(real_data_path):
		if scene_name.find('.') == 0:
			continue
		if scene_name.find('23') < 0 and scene_name.find('24') < 0:
			continue
		scene_path = os.path.join(real_data_path, scene_name)
		for face_name in os.listdir(scene_path):
			if face_name.find('.') == 0:
				continue
			if face_name.find('eye.txt') == 0:
				continue
			print("scene:", scene_name, ", face:", face_name)
			face_path = os.path.join(scene_path, face_name)
			img_path = os.path.join(face_path, 'img.jpg')
			eye_path = os.path.join(face_path, 'eye.npy')
			pos_path = os.path.join(face_path, 'pos.jpg')
			eye_env_path = os.path.join(face_path, 'eye.jpg')
			img = cv2.imread(img_path)
			try:
				r1, r2, c1, c2 = np.load(eye_path)
			except BaseException:
				continue
			r1, r2, c1, c2 = int(r1), int(r2), int(c1), int(c2)
			eye_info = (r1, r2, c1, c2)
			eye_info_class = eye_env(eye_info, img)
			eye_env_img = eye_info_class.gen_map()
			pos_map = img2posmap(eye_info = eye_info, img = img, threshold = 128)
			cv2.imwrite(pos_path, pos_map)
			cv2.imwrite(eye_env_path, eye_env_img)
