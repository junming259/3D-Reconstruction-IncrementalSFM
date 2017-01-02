from Function import makehomog
import numpy as np



def project(points, camera_params):
    """This function is used to project 3d points to 2d points
    Input:
            points: 3D points, nx3;
            camera_params: camera matrices, nx12.
    Output:
            points_proj: projected points in the images, nx2."""
    points_proj = np.zeros((points.shape[0], 2))
    points = makehomog(points)
    for i,item in enumerate(camera_params):
        p = item.reshape(3,4)
        x = p.dot(points[i].reshape(4,1))
        x = x/x[2]
        points_proj[i,0] = x[0]
        points_proj[i,1] = x[1]

    return points_proj

    
def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """This function is used to compute residuals.
    Input:
            params: camera parameters and 3D points;
            n_cameras: number of camera;
            n_points: number of 3D points;
            camera_indices: indexing of camera;
            point_indices: indexing of 3D points from 2D points;
            points_2d: 2D points in the images.
    Output:
            res: residual of prjection."""
    
    camera_params = params[:n_cameras * 12].reshape((n_cameras, 12))
    points_3d = params[n_cameras * 12:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    
    return (points_proj - points_2d).ravel()

    
from scipy.sparse import lil_matrix
def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    """To make this process time feasible we provide Jacobian sparsity structure
    (i. e. mark elements which are known to be non-zero)"""
    m = camera_indices.size * 2
    n = n_cameras * 12 + n_points * 3
    A = lil_matrix((m, n), dtype=int)
    i = np.arange(camera_indices.size)
    for s in range(12):
        A[2 * i, camera_indices * 12 + s] = 1
        A[2 * i + 1, camera_indices * 12 + s] = 1

    for s in range(3):
        print n_cameras * 12 + point_indices * 3 + s
        A[2 * i, n_cameras * 12 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 12 + point_indices * 3 + s] = 1

    return A
