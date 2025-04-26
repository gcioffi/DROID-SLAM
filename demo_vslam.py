import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse

from torch.multiprocessing import Process
from droid import Droid

import torch.nn.functional as F


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)


def image_stream(datapath, calib_fn, image_size, fisheye, stereo=False, stride=1):
    """ image generator """
    calib = np.loadtxt(calib_fn, delimiter=" ")
    fx, fy, cx, cy = calib[:4]
    d1, d2, d3, d4 = calib[4:8]
    w, h = calib[8:10].astype(int)

    K_l = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape(3,3)
    new_K = K_l.copy()
    
    if fisheye:
        d_l = np.array([[d1], [d2], [d3], [d4]])
        map_l = cv2.fisheye.initUndistortRectifyMap(K_l, d_l, np.eye(3), new_K, (w, h), cv2.CV_32F)
    else:
        d_l = np.array([d1, d2, d3, d4, 0.0])
        map_l = cv2.initUndistortRectifyMap(K_l, d_l, np.eye(3), new_K, (w, h), cv2.CV_32F)

    intrinsics_vec = [fx, fy, cx, cy]
    ht0, wd0 = [h, w]

    # read all png images in folder
    images_left = sorted(glob.glob(os.path.join(datapath, '*.png')))[::stride]

    for t, imgL in enumerate(images_left):
        tstamp = float(imgL.split('/')[-1][:-4])        
        images = [cv2.remap(cv2.imread(imgL), map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR)]
        
        images = torch.from_numpy(np.stack(images, 0))
        images = images.permute(0, 3, 1, 2).to("cuda:0", dtype=torch.float32)
        images = F.interpolate(images, image_size, mode="bilinear", align_corners=False)
        
        intrinsics = torch.as_tensor(intrinsics_vec).cuda()
        intrinsics[0] *= image_size[1] / wd0
        intrinsics[1] *= image_size[0] / ht0
        intrinsics[2] *= image_size[1] / wd0
        intrinsics[3] *= image_size[0] / ht0

        yield stride*t, images, intrinsics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", help="path to dataset", required=True)
    parser.add_argument('--scene', help="path to image folder", required=True)
    parser.add_argument('--calib', required=True)
    parser.add_argument('--fisheye', action="store_true")
    parser.add_argument("--gt", help="path to gt file")
    parser.add_argument("--weights", default="models/droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[320,512])
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--out_traj_path", help="path to saved estimated trajectory")

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--keyframe_thresh", type=float, default=3.5)
    parser.add_argument("--frontend_thresh", type=float, default=17.5)
    parser.add_argument("--frontend_window", type=int, default=20)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--backend_thresh", type=float, default=24.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=2)

    parser.add_argument("--upsample", action="store_true")

    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    imagedir = os.path.join(args.datapath, args.scene)

    # load some parameters that before where set manually as args
    calib_fn = os.path.join("calib", args.calib)

    print("Running evaluation on {}".format(imagedir))
    print(args)

    droid = Droid(args)
    time.sleep(5)

    n = 0
    for (t, image, intrinsics) in tqdm(image_stream(imagedir, calib_fn, args.image_size, args.fisheye, stereo=False, stride=1)):
        if n % 100 == 0:
            print(f"Processed {n} images")
        n += 1
        droid.track(t, image, intrinsics=intrinsics)    

    odom_traj_est = droid.return_trajectory(image_stream(imagedir, calib_fn, args.image_size, args.fisheye, stereo=False, stride=1))
    
    # This will call the global bundle adjustment
    traj_est = droid.terminate(image_stream(imagedir, calib_fn, args.image_size, args.fisheye, stereo=False, stride=1))

    if args.out_traj_path is not None:
        images_list = sorted(glob.glob(os.path.join(imagedir, '*.png')))
        tstamps = np.asarray([float(x.split('/')[-1][:-4]) for x in images_list])

        assert odom_traj_est.shape[0] == tstamps.shape[0], "Odometry Trajectory length does not match number of images"
        assert traj_est.shape[0] == tstamps.shape[0], "Trajectory length does not match number of images"

        odom_traj_out = np.zeros((tstamps.shape[0], 8))
        odom_traj_out[:, 0] = tstamps * 1e-9
        odom_traj_out[:, 1:] = odom_traj_est

        traj_out = np.zeros((tstamps.shape[0], 8))
        traj_out[:, 0] = tstamps * 1e-9
        traj_out[:, 1:] = traj_est
        
        out_odomtrajfn = args.out_traj_path + '/stamped_odomtraj_estimate.txt'
        np.savetxt(out_odomtrajfn, odom_traj_out, fmt='%.6f', header='ts x y z qx qy qz qw')
        print("Saved odometry trajectory to {}".format(out_odomtrajfn))

        out_trajfn = args.out_traj_path + '/stamped_traj_estimate.txt'
        np.savetxt(out_trajfn, traj_out, fmt='%.6f', header='ts x y z qx qy qz qw')
        print("Saved trajectory to {}".format(out_trajfn))