import numpy as np


def makehomog(m):
    """ This function is used to convert nonehomo coordinates to homo.
     Input: m, a nx2 matrix
     Output: p, a nx3 matrix"""
    p = np.hstack((m, np.ones((m.shape[0], 1))))
    return p
	

def triangulate_point(k,x1,x2,P1,P2):
    """ Point pair triangulation from least squares solution.
    Input:
            x1: point in image1, 3x1;
            x2: correspondent point in image2, 3x1;
            P1: camera matrix of camera1, 3x4;
            P2: camera matrix of cmaera2, 3x4.
    Output:
            X: 3D point coordinate"""
#    if x1.shape[0] == 1:
#        x1 = x1.T
#    if x2.shape[0] == 1:
#        x2 = x2.T

    x1 = x1.reshape(3,1)
    x2 = x2.reshape(3,1)
    x1 = np.linalg.inv(k).dot(x1)
    x2 = np.linalg.inv(k).dot(x2)
    # M = np.zeros((6,6))
    # M[:3,:4] = P1
    # M[3:,:4] = P2
    # M[:3,4] = -x1
    # M[3:,5] = -x2
    # U,S,V = np.linalg.svd(M)
    # X = V[-1,:4]

    """Another method"""
    M = np.zeros((4, 4))
    M[0, :] = P1[0, :] - x1[0] * P1[2, :]
    M[1, :] = P1[1, :] - x1[1] * P1[2, :]
    M[2, :] = P2[0, :] - x2[0] * P2[2, :]
    M[3, :] = P2[1, :] - x2[1] * P2[2, :]
    U,S,V = np.linalg.svd(M)
    X = V[-1, :]
    return X / X[3]
	

def triangulate(k, x1, x2, P1, P2):
    """ Two-view triangulation of points in x1,x2 (nx3 homogeneous coordinates).
    Before Triangulation, you should normalize image points with k.
    Input:
            k: internal parameter matrix;
            x1, x2: two pair of points in image1 and image2, nx3;
            P1, P2: camera matrices of camera1 and camera2, 3x4.
    Ouput:
            X: point3D in homogeneous coordinate, nx4"""

    n = x1.shape[0]
    if x2.shape[0] != n:
        raise ValueError("Number of points don't match.")
#    x1n = np.linalg.inv(k).dot(x1.T).T
#    x2n = np.linalg.inv(k).dot(x2.T).T
#    X = [triangulate_point(x1n[i, :], x2n[i, :], P1, P2).T for i in range(n)]
    X = [triangulate_point(k, x1[i, :], x2[i, :], P1, P2).T for i in range(n)]
    return np.array(X)


def removeAmbiguity_P(k, P1, P2, x1, x2):
    """This function is used to remove the ambiguity when computing camera
    matrix. Using E to compute camera matrix, there are four possible solutions.
    The reasonable one is the one which make most of point3D in front of camera.
    Input:
            k: internal parameter matrix;
            P1: camera matrix p1;
            P2: camera matrices p2 (a list of four matrices);
            x1: inlier points in image1;
            x2: inlier points in image2;
    Output:
            ind:the index of reasonable matrix p2;
            infront: the index of points, which are in front of camera."""
    maxres = 0
    for i in range(4):
        # triangulate inliers and compute depth for each camera
        X = triangulate(k, x1, x2, P1, P2[i])
        d1 = np.dot(P1, X.T)[2]
        d2 = np.dot(P2[i], X.T)[2]
#        m = sum(d1 > 0) +  sum(d2 > 0)
        m = sum((d1 > 0) & (d2 > 0))
        print m
        if m > maxres:
            maxres = m
            ind = i
            infront = (d1 > 0) & (d2 > 0)
    return ind, infront
	

def computePfromEssential(E):
    """ Computes the second camera matrix (assuming P1 = [I 0]) from an essential
    matrix. Output is a list of four possible camera matrices.
    Input:
            E: essential matrix;
    Output:
            P2: a list of four possible camera matrices"""
    # make sure E is rank 2
    U,S,V = np.linalg.svd(E)
    if np.linalg.det(np.dot(U,V))<0:
        V = -V
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    # return all four solutions
    P2 = [np.vstack((np.dot(U,np.dot(W,V)).T,U[:,2])).T,
        np.vstack((np.dot(U,np.dot(W,V)).T,-U[:,2])).T,
        np.vstack((np.dot(U,np.dot(W.T,V)).T,U[:,2])).T,
        np.vstack((np.dot(U,np.dot(W.T,V)).T,-U[:,2])).T]
    return P2


def computeRTFromE(E, k, match1, match2):
    """This function is used to compute rotation and translation matrices
    from essential matrix.
    Input:
            E: essential matrix;
            k: internal parameter, 3x3;
            match1: points in image1, nx3;
            match2: points in image2, nx3.
    Output:
            R: rotation matrix;
            T: translation matrix. """
    z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    u, d, v = np.linalg.svd(E)
    t1 = u[:, [-1]]
    t2 = -t1.copy()
    q1 = u.dot(w).dot(v)
    q2 = u.dot(w.T).dot(v)
    if np.linalg.det(q1) > 0:
        r1 = q1.copy()
    else:
        r1 = -q1.copy()
    if np.linalg.det(q2) > 0:
        r2 = q2.copy()
    else:
        r2 = -q2.copy()
    rt = [np.hstack((r1, t1)), np.hstack((r2, t2)), np.hstack((r1, t2)), np.hstack((r2, t1))]
    p1 = np.zeros((3, 4))
    p1[:, 0:3] = np.eye(3)
    su = 0
    index = 0
    for i in range(4):
        p2 = rt[i]
        xw = triangulate(k, match1, match2, p1, p2)
        d1 = np.dot(p1, xw.T)[2]
        d2 = np.dot(p2, xw.T)[2]
        m = sum((d1 > 0) & (d2 > 0))
        if m > su:
            su = m
            index = i
            infront = (d1 > 0) & (d2 > 0)
    return rt[index], infront

	

def computeCameraMatrix(x, Xw):
    """This function is used to compute camera matrix from 3D points and image points
    Input:
            x: image points, nx2 or nx3(homogeneous coordinate);
            Xw: 3D points, nx4.
    Output:
            p: camera matrix, 3x4."""

    row = Xw.shape[0]
    M = np.zeros((2*row, 12))
    for i in range(row):
        M[2*i, 0:4] = Xw[i, :]
        M[2*i, 8:12] = -x[i, 0] * Xw[i, :]
        M[2*i + 1, 4:8] = Xw[i, :]
        M[2*i + 1, 8:12] = -x[i, 1] * Xw[i, :]
    _, _, vr = np.linalg.svd(M)
    pl = vr[-1, :]

    from scipy.optimize import minimize 
    def f(pm):
        p = np.array([[pm[0],pm[1],pm[2],pm[3]], [pm[4],pm[5],pm[6],pm[7]], [pm[8],pm[9],pm[10],pm[11]]])
        c = p.dot(Xw.T).T
        c = c/ c[:, [2]]
        err = c[:, :2] - x[:, :2]
        e = np.linalg.norm(err, axis=1)    
        e = sum(e)
        return e
        
#    k = minimize(f, pl)
#    print k.x.reshape(3,4)
    return pl.reshape(3,4)
    

def reprojectionErr(Xw, P, x, k):
    """This function is used to compute the cost of nonlinear estimation, this
    is for two views cases.
    Input:
            Xw: 3D points in world coordinate;
            P: camera matrix of camera1, from world coordinate to camera coordinate;
            x: correspondent points in image;
    Output:
            c1+c2: cost of nonlinear estimation."""
    P1 = k.dot(P)
    p1 = P1[0, :]
    p2 = P1[1, :]
    p3 = P1[2, :]
    c = (x[0] - p1.dot(Xw) / p3.dot(Xw))**2 + (x[1] - p2.dot(Xw) / p3.dot(Xw))**2
    return c


def nonlinearCost(Xw, P1, P2, x1, x2, k):
    """This function is used to compute the cost of nonlinear estimation, this
    is for two views cases.
    Input:
            Xw: 3D points in world coordinate;
            P1: camera matrix of camera1;
            P2: camera matrix of camera2;
            x1: correspondent points in image1;
            x2: correspondent points in image2;
    Output:
            c1+c2: cost of nonlinear estimation."""
    c1 = reprojectionErr(Xw, P1, x1, k)
    c2 = reprojectionErr(Xw, P2, x2, k)
    return c1 + c2


def triangulateNonelinear(k, P1, P2, x1, x2):
    """This function is used to triangulate 3D points with nonlinear method.
    Input:
            k: internal parameter matrix;
            P1: camera matrix of camera 1, 3x4;
            P2: camera matrix of camera 2, 3x4;
            x1: correspondent points in image1, nx2;
            x2: correspondent points in image2, nx2;
    Output:
            Xw: 3D points in world coordinate, nx3.
            """
    from scipy.optimize import minimize
    num = (x1.shape[0])
    points3D = np.zeros((num, 4))
    for i in range(num):
        initial = triangulate_point(k, x1[i, :], x2[i, :], P1, P2)
        result = minimize(nonlinearCost, initial, args=(P1, P2, x1[i, :], x2[i, :], k))
        p = result.x
        points3D[i, :] = p/p[3]
#        print nonlinearCost(initial, P1, P2, x1[i,:], x2[i,:], k), nonlinearCost(p/p[3], P1, P2, x1[i,:], x2[i,:], k)
#        print nonlinearCost(initial, P1, P2, x1[i,:], x2[i,:], k), result.fun
    return points3D

def computeEssentialfromFundamental(f, k):
    """This function is used to compute essential matrix from fundamental matrix
    Input:
            f: fundamental matrix;
            k: internal parameter matrix.
    Output:
            E: essential matrix."""
    essential = k.T.dot(f).dot(k)
    u, _, v = np.linalg.svd(essential)
    essentialMat = u.dot(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])).dot(v)
    return essentialMat

def linearPnP(Xw, x, k):
    """This function is used to compute camera matrix from 3D points and image points
    Input:
            x: points in image, nx3(homogeneous coordinate);
            Xw: 3D points, nx4.
    Output:
            p: camera matrix, 3x4."""
    row = Xw.shape[0]
    x = np.linalg.inv(k).dot(x.T).T
    x = x/x[:,[2]]
    M = np.zeros((2*row, 12))
    for i in range(row):
        M[2*i, 0:4] = Xw[i, :]
        M[2*i, 8:12] = -x[i, 0] * Xw[i, :]
        M[2*i + 1, 4:8] = Xw[i, :]
        M[2*i + 1, 8:12] = -x[i, 1] * Xw[i, :]
    _, _, vr = np.linalg.svd(M)
    pl = vr[-1, :].reshape(3,4)
    
    return pl

def PnPRANSAC(Xw, x, k, threshold=20):
    """This function is used to find the optimal camera matrix given 3D points and 
    correspondent 2D image points.
    Input:
            Xw: 3D points in world coordinate;
            x: coorespondent 3D points in image;
            k: internal parameter matrix.
    Output:
            P: estimated camera matrix."""
    num_inliers = 0
    for i in range(2000):
        N = []
        ind = np.random.choice(Xw.shape[0], 6, replace=False)
        Xw_t = Xw[ind,:]
        x_t = x[ind,:]
        p = linearPnP(Xw_t, x_t, k)
        for j in range(Xw.shape[0]):
            err = reprojectionErr(Xw[j, :], p, x[j, :], k)
            if err < threshold:
                N.append(j)
        if len(N) > num_inliers:
            num_inliers = len(N)
            index = N
            P = p
    print float(len(index))/Xw.shape[0]

    return P, index

def triangulateMulti(xps,P,k):
    """ Point pair triangulation from least squares solution.
    Input:
            x: a list of points in each image, nx2x1;
            P: a list of camera matrix of camera1, nx3x4;
    Output:
            X: 3D point coordinate"""
    n = len(xps)
    x = np.zeros((n,2))
    for i,item in enumerate(xps):
        x[i] = item
    x = makehomog(x)
    x = np.linalg.inv(k).dot(x.T).T
    M = np.zeros((2*n, 4))
    for i in range(n):
        M[2*i, :] = P[i][0, :] - x[i,0] * P[i][2, :]
        M[2*i+1, :] = P[i][1, :] - x[i,1] * P[i][2, :]
    U,S,V = np.linalg.svd(M)
    X = V[-1, :]

    return X / X[3]
