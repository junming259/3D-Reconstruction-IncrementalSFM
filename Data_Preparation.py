# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 23:23:52 2016

@author: junming
"""

import cv2
import numpy as np

def findCorrepondentPoints(images, t=0.7):
    '''This function is used to find the correspondent points of input images. 
    Input: 
            images: a list of image matrices;
            t: threshold for ratio comparing.
    Output:
            matches: a list of pair matches;
            F: a list of fundamental matrices of each pair of images.
            
    ----
    
    Examples:
    
    img1 = cv2.imread('000.jpg', 0)
    img2 = cv2.imread('001.jpg', 0)
    img3 = cv2.imread('002.jpg', 0)
    images = [img1, img2, img3]
        
    P, F = findCorrepondentPoints(images, 0.7)
    >>>P
    [[match1], [match2]]    
    
    >>>match1
    [[points_img1][points_img2]]
    
    >>>points_img1
    [(123, 23),
     (234, 56),
     ...
     ...
     (135, 64)]
           
    '''
            
    num_images = len(images)
    surf = cv2.SURF(300)
    F = []
    #  FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params) 
    
    kps = []
    desps = []
    matches = []
    for i in range(num_images):
        kp0, des0 = surf.detectAndCompute(images[i], None)
        kps.append(kp0)
        desps.append(des0)

    for i in range(num_images-1):
        kp1 = kps[i]
        kp2 = kps[i+1]
        match = flann.knnMatch(desps[i], desps[i+1], k=2)
        M = []
        for r, (m, n) in enumerate(match):
            if m.distance < t * n.distance:
                M.append(np.append(kp1[m.queryIdx].pt, kp2[m.trainIdx].pt))
        M = np.array(M)
        k0 = np.double(M[:,:2].copy())
        kt = np.double(M[:,2:].copy())
        f, mask = cv2.findFundamentalMat(k0, kt, cv2.FM_RANSAC, 3, 0.99)
        print float(np.sum(mask))/mask.shape[0]
        mask = [i for i,it in enumerate(mask) if it==1]
        M = M[mask]
        M = [[(item[0], item[1]) for item in M[:,:2]], [(item[0], item[1]) for item in M[:,2:]]]
        matches.append(M)
        F.append(f)

    return matches, F
        
      

''' __main__'''     
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

ks, F = findCorrepondentPoints(images, 0.7)

# Begin building index
M0 = np.append(np.array(ks[0][0]), np.array(ks[0][1]), axis=1)
M = np.zeros((M0.shape[0], num_images*2)) - 1
M[:,:4] = M0
for i in range(1, len(ks)):
    print 'Proccessing %dth image' %(i)
    for j,item in enumerate(ks[i][0]):
        try:
            a = M[:,2*i:2*i+2].tolist()
            a = [tuple(it) for it in a]
            index = a.index(item)
            M[index, 2*i+2:2*i+4] = np.array(ks[i][1][j])
        except ValueError:
            b = np.zeros(num_images*2) -1
            b[2*i:2*i+2] = np.array(item)
            b[2*i+2:2*i+4] = np.array(ks[i][1][j])
            M = np.append(M, b.reshape(1,num_images*2), axis=0)

# write M, F in to file
np.savetxt('M.txt', M, delimiter=',')
np.savetxt('F.txt', F[0], delimiter=',')

#a = np.loadtxt('test.txt', delimiter=',')





