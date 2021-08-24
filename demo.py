import cv2
import math
import argparse
import numpy as np


def gauss(kernel_size, sigma):
    kernel = np.zeros([kernel_size, kernel_size])
    center = kernel_size // 2
    if sigma <= 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8

    s = sigma ** 2
    sum_val = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center

            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / 2 * s)
            sum_val += kernel[i, j]

    kernel = kernel / sum_val

    return kernel


def lid_contour(point, line):
    start, end = line[0], line[1]
    if min(start[1], end[1]) < point[1] <= max(start[1], end[1]):
        if point[0] < max(start[0], end[0]):
            k = (end[1] - start[1]) / (end[0] - start[0])
            b = end[1] - k * end[0]
            intersect_x = (point[1] - b) / k
            if point[0] < intersect_x:
                return 1
    return 0


def initialize(img, lb, rb, lc, rc):
    res = img.copy()

    rev = cv2.bitwise_not(img)
    b, g, r = cv2.split(rev)
    width = (lb[-1] - lb[-2] + rb[-1] - rb[-2]) / 4
    kernel = np.ones((int(width / 2), int(width / 2)), np.uint8)
    closing_b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel)
    closing_g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel)
    closing_r = cv2.morphologyEx(r, cv2.MORPH_CLOSE, kernel)
    bgr_closing = cv2.merge((closing_b, closing_g, closing_r))
    bgr_closing = cv2.bitwise_not(bgr_closing)

    left_point_x = [ls[0][0] for ls in lc] + [ls[1][0] for ls in lc]
    left_point_y = [ls[0][1] for ls in lc] + [ls[1][1] for ls in lc]
    left_min_x, left_max_x = min(left_point_x), max(left_point_x)
    left_min_y, left_max_y = min(left_point_y), max(left_point_y)

    right_point_x = [ls[0][0] for ls in rc] + [ls[1][0] for ls in rc]
    right_point_y = [ls[0][1] for ls in rc] + [ls[1][1] for ls in rc]
    right_min_x, right_max_x = min(right_point_x), max(right_point_x)
    right_min_y, right_max_y = min(right_point_y), max(right_point_y)

    for x in range(left_min_x, left_max_x+1):
        for y in range(left_min_y, left_max_y+1):
            tester = sum([lid_contour([x, y], line) for line in lc])
            if tester % 2 == 1:
                res[y, x, :] = bgr_closing[y, x, :]

    for x in range(right_min_x, right_max_x+1):
        for y in range(right_min_y, right_max_y+1):
            tester = sum([lid_contour([x, y], line) for line in rc])
            if tester % 2 == 1:
                res[y, x, :] = bgr_closing[y, x, :]

    return res


'''
 input: r1, r2, c1, c2 

             c1            c2
  -|--------------------------------> col
   |         |              |
 r1| - - - - + - ********   |
   |      	 | *          * |
   |      	 |*            *|
   |      	  *            *
   |           *          * 
 r2|- - - - - -  ********
   |    
   |              
   v
 row
'''


class eye_info:
    def __init__(self, eye_bbox, eye_img, eye_nc):
        self.f = 50.0  # focal length
        self.rL = 5.5  # limbus radius length in real world
        self.p, self.R = 0.75, 7.8  # parameters for ellipsoid corneal surface
        self.r1, self.r2, self.c1, self.c2 = eye_bbox  # limbus bounding box in image

        # major and minor radii of the corneal ellipse in the image plane
        self.r_min, self.r_max = (self.r2 - self.r1) / 2, (self.c2 - self.c1) / 2

        self.img = eye_img  # read portrait
        self.eye_nc = eye_nc
        self.img_row = self.img.shape[0]
        self.img_col = self.img.shape[1]
        self.pxl = math.sqrt((36 * 24) / (self.img_row * self.img_col))  # pixel size

        # eye mask for poisson blending
        self.eye_msk = np.zeros([self.img_row, self.img_col, 3]).astype('uint8')
        # source image for poisson blending
        self.eye_src = np.zeros([self.img_row, self.img_col, 3]).astype('uint8')
        # target image for poisson blending
        self.eye_tar = self.img.copy()

        self.r1_img, self.r2_img = int(self.r1), int(self.r2)
        self.c1_img, self.c2_img = int(self.c1), int(self.c2)

        self.center_img = np.array([int((self.c1_img + self.c2_img) / 2),
                                    int((self.r1_img + self.r2_img) / 2)])

        # shift the origin to the center of the image plane
        self.r1 = int(self.r1 - self.img_row / 2)
        self.r2 = int(self.r2 - self.img_row / 2)
        self.c1 = int(self.c1 - self.img_col / 2)
        self.c2 = int(self.c2 - self.img_col / 2)

        self.center = (int((self.r1 + self.r2) / 2),
                       int((self.c1 + self.c2) / 2))

        # limbus coordinate in camera system (unit in pixels)
        self.dx = self.center[1] * self.rL / (self.r_max * self.pxl)
        self.dy = self.center[0] * self.rL / (self.r_max * self.pxl)
        self.dz = (self.f / self.pxl) * self.rL / (self.r_max * self.pxl)

    def xy2tp(self, x0_, y0_):
        p = 36 * 24 / 22300000 * 10000
        dz = self.f * (self.rL * 2 / p)
        x0 = x0_ / self.r_min * self.rL
        y0 = y0_ / self.r_min * self.rL
        z0 = math.sqrt(8.028 * 8.028 - x0 * x0 - y0 * y0)

        # theta-phi mapping
        sigma1 = math.sqrt(-4 * dz * dz * x0 * x0 - 4 * dz * dz * y0 * y0 + 257.795136 * dz * dz -
                           25882.98205 * dz * z0 + 6444.8784 * x0 * x0 + 6444.8784 * y0 * y0 + 6444.8784 * z0 * z0)
        sigma2 = 2 * (dz * dz - 10 * dz * z0 + 25 * x0 * x0 + 25 * y0 * y0 + 25 * z0 * z0)

        x_1 = x0 - 5 * x0 * (-sigma1 - 2 * dz * z0 + 10 * x0 * x0 + 10 * y0 * y0 + 10 * z0 * z0) / sigma2
        x_2 = x0 - 5 * x0 * (sigma1 - 2 * dz * z0 + 10 * x0 * x0 + 10 * y0 * y0 + 10 * z0 * z0) / sigma2
        y_1 = y0 - 5 * y0 * (-sigma1 - 2 * dz * z0 + 10 * x0 * x0 + 10 * y0 * y0 + 10 * z0 * z0) / sigma2
        y_2 = y0 - 5 * y0 * (sigma1 - 2 * dz * z0 + 10 * x0 * x0 + 10 * y0 * y0 + 10 * z0 * z0) / sigma2

        temp = sigma1
        sigma1 = 10 * x0 * x0 + 10 * y0 * y0 + 10 * z0 * z0 - 2 * dz * z0 - temp
        sigma2 = 10 * x0 * x0 + 10 * y0 * y0 + 10 * z0 * z0 - 2 * dz * z0 + temp
        sigma3 = 2 * (dz * dz - 10 * dz * z0 + 25 * x0 * x0 + 25 * y0 * y0 + 25 * z0 * z0)

        z_1 = z0 + (dz * sigma1) / sigma3 - (5 * z0 * sigma1) / sigma3
        z_2 = z0 + (dz * sigma2) / sigma3 - (5 * z0 * sigma2) / sigma3

        x, y, z = x_1, y_1, z_1

        if z < 0:
            x, y, z = x_2, y_2, z_2

        y, z = z, y

        # calculate theta and phi in radian
        theta_ = math.acos(z / 8.028)

        if x != 0:
            phi_ = math.atan(y / x)
        else:
            phi_ = -1 * math.pi / 2

        if x <= 0 and y <= 0:
            phi_ = phi_ + math.pi
        elif x <= 0 and y >= 0:
            phi_ = phi_ + math.pi
        elif x >= 0 and y <= 0:
            phi_ = phi_ + 2 * math.pi

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

    def lid_contour(self, scale, point, line):
        start = scale * line[0] + (1 - scale) * self.center_img
        end = scale * line[1] + (1 - scale) * self.center_img

        if min(start[1], end[1]) < point[1] <= max(start[1], end[1]):
            if point[0] < max(start[0], end[0]):
                k = (end[1] - start[1]) / (end[0] - start[0])
                b = end[1] - k * end[0]
                intersect_x = (point[1] - b) / k
                if point[0] < intersect_x:
                    return 1
        return 0

    def render(self, row, col, irs_img, env_val):
        irs_size = irs_img.shape

        x = col - self.c1
        y = row - self.r1

        x_p = int(x / (self.r_max * 2) * irs_size[1])
        y_p = int(y / (self.r_min * 2) * irs_size[0])

        irs_val = irs_img[y_p, x_p, :]
        return 0.16 * irs_val + 0.4 * env_val

    def env2eye(self, irs_img, env_img, line_set):
        '''
        point_x = [ls[0][0] for ls in line_set] + [ls[1][0] for ls in line_set]
        point_y = [ls[0][1] for ls in line_set] + [ls[1][1] for ls in line_set]
        min_x, max_x = min(point_x), max(point_x)
        min_y, max_y = min(point_y), max(point_y)
        '''
        env_row = env_img.shape[0]
        env_col = env_img.shape[1]

        light_pos = np.array([0, 0, 0])
        light_norm = np.array([1, 0, 0])
        light_norm = light_norm / np.linalg.norm(light_norm)
        light_rad = 12000

        scale = 0.9
        col_scopes = {}
        threshold = cv2.mean(self.img[self.r1_img:self.r2_img,
                             self.c1_img:self.c2_img])
        mean_intensity = sum(threshold) / 3

        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        reverse = cv2.bitwise_not(gray)
        kernel = np.ones((int(self.r_max / 4), int(self.r_max / 4)), np.uint8)

        gray_closing = cv2.morphologyEx(reverse, cv2.MORPH_CLOSE, kernel)
        gray_closing = cv2.bitwise_not(gray_closing)
        # edges = cv2.Canny(gray_closing, 150, 255)
        ret, thresh = cv2.threshold(gray_closing, mean_intensity, 255, cv2.THRESH_BINARY)

        # for point within the ellipse
        for row in range(self.r1, self.r2):
            y0_ = row - self.center[0]
            r_img = row + int(self.img_row / 2)

            if (scale ** 2 - (y0_ / self.r_min) ** 2) <= 0:
                continue

            offset = math.sqrt((scale ** 2 - (y0_ / self.r_min) ** 2) * (self.r_max ** 2))
            offset = math.ceil(offset)
            col_scope = [self.center_img[0] - offset,
                         self.center_img[0] + offset]

            indent = 0
            for col in range(col_scope[0], col_scope[1], 1):
                tester = sum([self.lid_contour(scale, [col, r_img], line) for line in line_set])
                if tester % 2 == 1:
                    # cv2.circle(self.img, (col_scope[0], r_img), 1, [255, 0, 0])
                    while thresh[r_img, col_scope[0]]:
                        indent += 1
                        col_scope[0] += 1
                    col_scope[0] += 1
                    break
                col_scope[0] += 1

            if col_scope[0] >= col_scope[1] or indent >= self.r_max / 4:
                continue

            # cv2.circle(self.img, (col_scope[0], r_img), 1, [0, 255, 0])

            indent = 0
            for col in range(col_scope[1], col_scope[0], -1):
                tester = sum([self.lid_contour(scale, [col, r_img], line) for line in line_set])
                if tester % 2 == 1:
                    # cv2.circle(self.img, (col_scope[1], r_img), 1, [255, 0, 0])
                    while thresh[r_img, col_scope[1]]:
                        indent += 1
                        col_scope[1] -= 1
                    col_scope[0] -= 1
                    break
                col_scope[1] -= 1

            if col_scope[0] >= col_scope[1] or indent >= self.r_max / 4:
                continue

            # cv2.circle(self.img, (col_scope[1], r_img), 1, [0, 255, 0])

            temp = np.array([gray_closing[r_img, col_scope[0]:col_scope[1]] / 3]).transpose()
            self.img[r_img, col_scope[0]:col_scope[1]] = np.hstack((temp, temp, temp))

            col_scopes[r_img] = col_scope

            for col in range(col_scope[0], col_scope[1]):
                x0_ = col - self.center_img[0]

                # compute real location in camera coordinate
                theta, phi = self.xy2tp(x0_, y0_)

                t = theta / 180 * math.pi
                p = phi / 180 * math.pi
                center = np.array([self.dz, 0, 0])
                normal = np.array([math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t)])
                normal /= np.linalg.norm(normal)

                incident = center + 8.028 * normal
                emergent = incident - 2 * np.dot(incident, normal) * normal

                temp = emergent - center
                r_p = np.linalg.norm(temp)
                theta_p = math.acos(temp[2] / r_p) / math.pi * 180
                phi_p = (math.atan(temp[1] / temp[0]) + math.pi / 2) / math.pi * 180

                if theta_p < 30:
                    theta_p = 30
                if theta_p > 150:
                    theta_p = 150

                x_p = int(phi_p / 360 * env_col)
                y_p = int((theta_p - 30) / 120 * env_row)
                pixel = env_img[y_p, x_p, :] * 255

                temp1 = light_norm.dot(incident - light_pos)
                temp2 = light_norm.dot(emergent)
                t = (-1) * temp1 / temp2
                crossover = incident + t * emergent

                if np.linalg.norm(crossover - light_pos) <= light_rad:
                    pixel = np.array([255, 255, 255])

                self.eye_msk[r_img, col] = np.array([255, 255, 255])
                self.eye_src[r_img, col] = self.render(row, int(col - self.img_col / 2), irs_img, pixel)

        # cv2.imwrite('./test_result.jpg', self.img)

        dict_items = sorted(col_scopes.items(), key=lambda x: x[0])

        std1 = len(dict_items)
        std2 = sum([item[1][1] - item[1][0] for item in dict_items])
        if std1 <= 20 or std2 < 500:
            self.eye_tar = self.eye_nc.copy()
            return 0

        left_up = dict_items[0]
        right_bottom = dict_items[-1]
        center = (self.center_img[0], int((left_up[0] + right_bottom[0]) / 2))

        self.eye_tar = cv2.seamlessClone(self.eye_src, self.img, self.eye_msk, center, cv2.NORMAL_CLONE)

        '''
        for item in dict_items:
            cv2.circle(self.eye_tar, (item[1][0], item[0]), 1, (255, 0, 0))
            cv2.circle(self.eye_tar, (item[1][1], item[0]), 1, (255, 0, 0))
        '''

        # cv2.imwrite('./poisson_result.jpg', self.eye_tar)

        # self.eye_msk = cv2.Laplacian(self.eye_msk, -1)
        # self.eye_msk = cv2.GaussianBlur(self.eye_msk, (5, 5), 0)

        sigma = 0.01
        kernel_size = 5
        origin = self.eye_tar.copy()

        for item in dict_items:
            row = item[0]
            col_scope = item[1]
            for col in range(col_scope[0], col_scope[1]):
                conv = []
                for channel in range(3):
                    window = origin[row - kernel_size // 2:row + kernel_size // 2 + 1,
                             col - kernel_size // 2:col + kernel_size // 2 + 1, channel]
                    conv.append(np.sum(gauss(kernel_size, sigma) * window))
                self.eye_tar[row, col] = np.array(conv)

        '''
        r1, r2 = self.r1_img - 10, self.r2_img + 10
        c1, c2 = self.c1_img - 10, self.c2_img + 10

        itp_margin = False
        itp_start, itp_end = -1, -1
        
        for row in range(r1, r2):
            for col in range(c1, c2):
                if np.any(self.eye_msk[row, col]):
                    itp_margin = True
                    itp_start = col - 3
                if itp_margin and not np.any(self.eye_msk[row, col]):
                    itp_end = col + 3
                    for col_p in range(itp_start, itp_end):
                        width = itp_end - itp_start
                        alpha = (width - (col_p - itp_start)) / width
                        self.eye_tar[row][col_p] = alpha * origin[row][itp_start].copy() + \
                                                   (1 - alpha) * origin[row][itp_end].copy()
                    itp_margin = False
        '''

        self.eye_tar = np.clip(self.eye_tar, 0, 255).astype('uint8')
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path')
    args = parser.parse_args()

    '''
    img = cv2.imread(args.file_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)
    CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = CLAHE.apply(l)

    l_img = cv2.merge((cl, a, b))
    final = cv2.cvtColor(l_img, cv2.COLOR_LAB2BGR)
    '''

    data_path = './data.txt'
    data = open(data_path)

    left_string = data.readline()
    right_string = data.readline()
    left_bbox = left_string.split()
    right_bbox = right_string.split()
    left_bbox = [float(num) for num in left_bbox]
    right_bbox = [float(num) for num in right_bbox]

    irs_path = './iris.jpg'
    irs = cv2.imread(irs_path)
    eye_path = args.file_path
    # eye_path = '/Users/erichuang/Desktop/Eye/material/0000000000.png'
    eye = cv2.imread(eye_path)
    env_path = './env_map_0525_mid.hdr'
    env = cv2.imread(env_path, cv2.IMREAD_ANYDEPTH)
    # tone_map = cv2.createTonemap(gamma=2.2)
    # env = tone_map.process(env)
    # env = np.clip(env * 255, 0, 255).astype('uint8')

    left_contour, right_contour = [], []
    if data.readline() == 'right\n':
        while True:
            line_string = data.readline()[:-1]

            if line_string == 'left':
                break

            string_list = line_string.split(' ')
            line = []
            for string in string_list:
                num_list = string.split(',')
                point = [int(coordinate) for coordinate in num_list]
                line.append(point)
            right_contour.append(line)

        while True:
            line_string = data.readline()

            if not line_string:
                break

            line_string = line_string[:-1]
            string_list = line_string.split(' ')
            line = []
            for string in string_list:
                num_list = string.split(',')
                point = [int(coordinate) for coordinate in num_list]
                line.append(point)
            left_contour.append(line)

    left_contour = np.array(left_contour)
    right_contour = np.array(right_contour)

    init = initialize(eye, left_bbox, right_bbox, left_contour, right_contour)

    left_eye = eye_info(left_bbox, eye, init)
    left_eye.env2eye(irs, env, left_contour)

    right_eye = eye_info(right_bbox, left_eye.eye_tar, left_eye.eye_tar.copy())
    right_eye.env2eye(irs, env, right_contour)

    output_path = args.file_path.rstrip('.png') + '_tar.jpg'
    # output_path = eye_path.rstrip('.png') + '_tar.jpg'
    cv2.imwrite(output_path, right_eye.eye_tar)
