import numpy as np
import cv2
from open3d import *
from matplotlib import pyplot as plt
import random
import os
import pdb




def find_clearance(map_path,visualize = 0):
    '''

    :param map_path: string : Path to depth map stored as txt
    :param visualize: save interim point clouds using open3d
    :return: None
    '''

    ## Note :  calls to open3d point cloud writes can be enabled using visualize for debug ##
    dmap = np.loadtxt(map_path)
    dmap = dmap.astype(np.float32)
    work_dir = os.path.dirname(map_path)
    suffix = "_vis"  ##work_dir and siffix are for saving point clouds, if uncommented
    ##clean up data, by removing depths outside a range #
    r = 1.3
    dr = 3
    dmap[(dmap <= r) | (dmap >= r + dr)] = 0

    ##estimate some rough intrinsic values ##
    H, W = dmap.shape

    # (width: int, height: int, fx: float, fy: float, cx: float, cy: float)
    fx, fy, cx, cy = W, W, W // 2, H // 2
    intr = PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    source_color = Image(128 * np.ones([H, W, 3], dtype=np.int8))
    source_depth = Image(dmap.squeeze())

    source_rgbd_image = create_rgbd_image_from_color_and_depth(source_color, source_depth)
    source_pcd = create_point_cloud_from_rgbd_image(source_rgbd_image, intr)
    pts = np.asarray(source_pcd.points)

    ##scale up system such that walls are 1.5 m apart ##
    scale = abs(np.max(pts[:, 0]) - np.min(pts[:, 0])) / 1.5
    pts = pts / scale
    # print("x min max:", np.min(pts[:, 0]), np.max(pts[:, 0]))
    # print("z min max", np.min(pts[:, 2]), np.max(pts[:, 2]))
    pcd = PointCloud()
    pcd.points = Vector3dVector(pts)
    if(visualize):
        write_point_cloud(work_dir + 'depth_map_cloud_' + suffix + ".ply", pcd)

    orientations = []

    def gen_plane(pts):
        '''

        :param pts: 3x3 array to fit plane to points
        :return: plane coeffs a,b,c
        '''
        A = np.ones_like(pts)
        #     pdb.set_trace()
        A[:, 0:2] = pts[:, 0:2]
        B = pts[:, 2]

        fit = np.linalg.solve(A, B)

        return fit

    def get_plane_inliers(pts, plane, thresh=0.1):
        plane = plane.reshape(3, 1)
        A = np.ones_like(pts)
        A[:, 0:2] = pts[:, 0:2]
        B = pts[:, 2]
        #     pdb.set_trace()
        dts = abs(B.reshape(-1, 1) - A.dot(plane)) / np.sqrt(plane[0] ** 2 + plane[1] ** 2 + 1)

        #     errors = B.reshape(-1,1) - A.dot(plane)
        return np.where(dts.squeeze() < thresh)[0]

    import pdb
    plane_count = 3 ##assuming 3 planes for given scenario, can be configured
    planes = []
    cur_pt_set = pts.copy()

    ransac_iters = 1000
    for i in range(plane_count):
        if (cur_pt_set.shape[0] < 4):
            # print("not enough points, breaking \n")
            break
        best, best_inliers, max_inliers = None, None, 0
        for j in range(ransac_iters):
            rng = cur_pt_set.shape[0]

            inds = random.sample(range(0, rng), 3)
            try:
                plane = gen_plane(cur_pt_set[inds])
            except:
                continue
            inliers = get_plane_inliers(cur_pt_set, plane)
            if (len(inliers) > max_inliers):
                best = plane
                max_inliers = len(inliers)
                # print("new max : ", max_inliers)
                best_inliers = inliers


        # print("new refitted inliers: ", len(best_inliers))
        wall_pts = cur_pt_set[best_inliers]
        ##determine wall orientation ##
        if (np.median(wall_pts[:, 0]) > 0):
            orientations.append('right')
        else:
            orientations.append('left')


        if(visualize):
            pcd = PointCloud()
            pcd.points = Vector3dVector(wall_pts)
            write_point_cloud(work_dir + 'wall_' + str(i) + ".ply", pcd)
        ##remove inliers from point cloud, to fit next plane
        cur_pt_set = np.delete(cur_pt_set, best_inliers, axis=0)
        planes.append(best)


    human_pts = cur_pt_set  ##
    if (visualize):
        pcd = PointCloud()
        pcd.points = Vector3dVector(human_pts)
        write_point_cloud(work_dir + 'human_' + suffix + ".ply", pcd)

    #####
    # 3 planes, find two with normals parallel to x (dot product highest)


    norms = -1 * np.ones_like(planes)
    for i in range(norms.shape[0]):
        norms[i, 0:2] = planes[i][0:2]
        norms[i, :] = norms[i, :] / np.linalg.norm(norms[i, :])
    # norms[:,0:2] = planes[:,0:2]

    ##sort planes based on dot product with x axis##
    #goal is to filter out the two main walls ##
    x = np.array([1, 0, 0])
    dots = []
    for i in range(norms.shape[0]):
        # print(abs(np.dot(norms[i], x)))
        dots.append(abs(np.dot(norms[i], x)))
    sorted_planes = np.argsort(dots)

    walls = []
    wall_ors = []
    for i in sorted_planes[1:]:
        walls.append(planes[i])
        wall_ors.append(orientations[i])

    max_clear = 0
    ##for the two walls and the "human" points, find the median distance, print the clearance as max of this ##
    for i in range(2):
        pts = human_pts
        pl = walls[i]
        A = np.ones_like(pts)
        A[:, 0:2] = pts[:, 0:2]
        B = pts[:, 2]
        #     pdb.set_trace()
        dts = abs(B.reshape(-1, 1) - A.dot(pl)) / np.sqrt(pl[0] ** 2 + pl[1] ** 2 + 1)
        med_dts = np.median(dts)
        # print("median wall to human dist:", np.median(dts))
        if (max_clear < med_dts):
            answer = [wall_ors[i], med_dts]
            max_clear = med_dts
    print (answer[0]  +" " + str(answer[1]))



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='find clearance using a depth map')
    parser.add_argument(
        'path', metavar='path', nargs=1, type=str,
        help='Path to depth map text file')


    args = parser.parse_args()

    find_clearance(args.path[0])
