# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 22:19:42 2016

@author: junming
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from Function import makehomog, computeEssentialfromFundamental, computePfromEssential
from Function import triangulate, removeAmbiguity_P, PnPRANSAC, triangulateMulti
from Bundle_Adjustment import fun, bundle_adjustment_sparsity


def plotPoints3D(Xw):
    # 3D reconstruction from two images
    from mpl_toolkits.mplot3d import axes3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(Xw[:, 0], Xw[:, 1], Xw[:, 2], 'k.')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()  

 

'''__main__'''
'''Preparation: read images, set K, read match points, F...'''
img1 = cv2.imread('0001.jpg', 0)
img2 = cv2.imread('0002.jpg', 0)
img3 = cv2.imread('0003.jpg', 0)
img4 = cv2.imread('0004.jpg', 0)
img5 = cv2.imread('0005.jpg', 0)
img6 = cv2.imread('0006.jpg', 0)

kernel = (5, 5)
img1 = cv2.GaussianBlur(img1, kernel, 0)
img2 = cv2.GaussianBlur(img2, kernel, 0)
img3 = cv2.GaussianBlur(img3, kernel, 0)
img4 = cv2.GaussianBlur(img4, kernel, 0)
img5 = cv2.GaussianBlur(img5, kernel, 0)
img6 = cv2.GaussianBlur(img6, kernel, 0)

images = [img1, img2, img3, img4, img5, img6]
num_images = len(images)

# the calibrated matrix k, which contains information about internal parameters, we get k from camera calibration.
k = np.array([[2759.48, 0, 1520.69], [0, 2764.16, 1006.81], [0, 0, 1]])

M = np.loadtxt('M.txt', delimiter=',')
F = np.loadtxt('F.txt', delimiter=',')



'''initialization, use the first two images to triangulate'''
# read correspondent points in first two images
p1_inlier = M[:,:2][M[:,0] > 0].copy()
p2_inlier = M[:,2:4][M[:,0] > 0].copy()
p1_inlier = makehomog(p1_inlier)
p2_inlier = makehomog(p2_inlier)

# compute essential matrix from fundamental matrix
E = computeEssentialfromFundamental(F, k)
# compute the first two camera matrix P1 P2, and then use them to do triangulation.
P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
# return four possible P
P2 = computePfromEssential(E) 
# remove ambibuity of camera matrix, only one of the four possible solution is reasonable
ind, infront = removeAmbiguity_P(k, P1, P2, p1_inlier, p2_inlier)
#Xw = triangulate(k, p1_inlier[infront], p2_inlier[infront], P1, P2[ind])
p1_inlier = p1_inlier[infront]
p2_inlier = p2_inlier[infront]
P = [P1, P2[ind]]
Xw = triangulate(k, p1_inlier, p2_inlier, P1, P2[ind])
# we can also use nonelinear method to get a more accurate result at the cost of time
# Xw = triangulateNonelinear(k, P1, P2[ind], p1_inlier[infront], p2_inlier[infront])



'''visulize result of linear reconstruction of two image'''
# reprojection 
predict_points1 = k.dot(P1).dot(Xw.T).T
predict_points1 = predict_points1 / predict_points1[:, [2]]
predict_points2 = k.dot(P2[ind]).dot(Xw.T).T
predict_points2 = predict_points2 / predict_points2[:, [2]]

fig1, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(images[0], cmap='gray')
ax2.imshow(images[1], cmap='gray')
ax1.plot(p1_inlier[:,0], p1_inlier[:,1], 'bo', markersize = 20)
ax1.plot(predict_points1[:,0], predict_points1[:,1], 'ro', markersize = 15)
ax2.plot(p2_inlier[:,0], p2_inlier[:,1], 'bo', markersize = 20)
ax2.plot(predict_points2[:,0], predict_points2[:,1], 'ro', markersize = 15)
ax1.set_title('reprojection error')

# 3D reconstruction from two images
plotPoints3D(Xw)



'''Register multiple images'''
# compute camera matrix of each camera
Xw = Xw[:,:3]
num = Xw.shape[0]
for i in range(2, num_images):
    mask = M[:num,2*i]>0
    p_inlier = M[:num,2*i:2*i+2][mask]
    p_inlier = makehomog(p_inlier)
    X = Xw[mask]
    X = makehomog(X)
    print X.shape, p_inlier.shape
    p, index = PnPRANSAC(X, p_inlier, k)    
    P.append(p)

# build camera_params, nx12    
camera_params = np.zeros((num_images, 12))
for i in range(num_images):
    camera_params[i,:] = k.dot(P[i]).reshape(1,12)
    
# build 3D points 
points_3d = []
for i,item in enumerate(M):
    index = np.argwhere(item>-1)
    kp1 = item[index[0]:index[0]+2]
    kp2 = item[index[2]:index[2]+2]
    p1 = P[index[0]/2]
    p2 = P[index[2]/2]
    p3d = triangulateMulti([kp1, kp2], [p1, p2], k)
    points_3d.append(p3d)
points_3d = np.array(points_3d)
points_3d = points_3d[:,:3]

# build 2D points, nx2; camera_indices, nx1; points_indices, nx1;
points_2d = []
camera_indices = []
points_indices = []
for i,row in enumerate(M):
    for j in range(num_images):
        if row[j*2] != -1:
            points_2d.append(row[j*2:j*2+2])
            points_indices.append(i)
            camera_indices.append(j)
            
points_2d = np.array(points_2d)
camera_indices = np.array(camera_indices)
points_indices = np.array(points_indices)

# n_cameras is the number of cameras or images 
n_cameras = camera_params.shape[0]
# n_points is the number of 3D points triangulated from all images
n_points = points_3d.shape[0]



'''Bundle Adjustment'''
# initial guess
x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

import time
from scipy.optimize import least_squares
# begin bundle adjustment
A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, points_indices)
t0 = time.time()
res = least_squares(fun, x0, verbose=2, jac_sparsity=A, x_scale='jac', ftol=1e-4, method='trf',
                    args=(n_cameras, n_points, camera_indices, points_indices, points_2d))
t1 = time.time()
print("Optimization took {0:.0f} seconds".format(t1 - t0))



'''Result visulization'''
camera_params = res.x[:n_cameras * 12].reshape((n_cameras, 12))
Xw = res.x[n_cameras * 12:].reshape((n_points, 3))
Xw = makehomog(Xw)
P1 = camera_params[0,:].reshape(3,4)
P2 = camera_params[1,:].reshape(3,4)

predict_points1 = P1.dot(Xw.T).T
predict_points1 = predict_points1 / predict_points1[:, [2]]
predict_points2 = P2.dot(Xw.T).T
predict_points2 = predict_points2 / predict_points2[:, [2]]

fig1, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(images[0], cmap='gray')
ax2.imshow(images[1], cmap='gray')
ax1.plot(p1_inlier[:,0], p1_inlier[:,1], 'bo', markersize = 20)
ax1.plot(predict_points1[:,0], predict_points1[:,1], 'ro', markersize = 15)
ax2.plot(p2_inlier[:,0], p2_inlier[:,1], 'bo', markersize = 20)
ax2.plot(predict_points2[:,0], predict_points2[:,1], 'ro', markersize = 15)
ax1.set_title('reprojection error')

plotPoints3D(Xw)
