# Indoor forward
python demo_vslam.py --datapath=/datasets/UZH_FPV/ --scene=indoor_forward_7_snapdragon_with_gt/img/left --calib=uzhfpv_indoor_forward.txt --fisheye --buffer=2000 --out_traj_prefix=/home/giovanni/DROIDSLAM_docker/DROID_SLAM/indoor_forward_7

# Indoor 45
python demo_vslam.py --datapath=/datasets/UZH_FPV/ --scene=indoor_45_2_snapdragon_with_gt/img/left --calib=uzhfpv_indoor_45.txt --fisheye --buffer=2000 --out_traj_prefix=/home/giovanni/DROIDSLAM_docker/DROID_SLAM/indoor_45_2
python demo_vslam.py --datapath=/datasets/UZH_FPV/ --scene=indoor_45_14_snapdragon_with_gt/img/left --calib=uzhfpv_indoor_45.txt --fisheye --buffer=2000 --out_traj_prefix=/home/giovanni/DROIDSLAM_docker/DROID_SLAM/indoor_45_14

# Outdoor forward
python demo_vslam.py --datapath=/datasets/UZH_FPV/ --scene=outdoor_forward_1_snapdragon_with_gt/img/left --calib=uzhfpv_outdoor_forward.txt --fisheye --buffer=2000 --out_traj_prefix=/home/giovanni/DROIDSLAM_docker/DROID_SLAM/outdoor_forward_1
python demo_vslam.py --datapath=/datasets/UZH_FPV/ --scene=outdoor_forward_3_snapdragon_with_gt/img/left --calib=uzhfpv_outdoor_forward.txt --fisheye --buffer=2000 --out_traj_prefix=/home/giovanni/DROIDSLAM_docker/DROID_SLAM/outdoor_forward_3
python demo_vslam.py --datapath=/datasets/UZH_FPV/ --scene=outdoor_forward_5_snapdragon_with_gt/img/left --calib=uzhfpv_outdoor_forward.txt --fisheye --buffer=2000 --out_traj_prefix=/home/giovanni/DROIDSLAM_docker/DROID_SLAM/outdoor_forward_5
python demo_vslam.py --datapath=/datasets/UZH_FPV/ --scene=outdoor_forward_10_snapdragon/img/left --calib=uzhfpv_outdoor_forward.txt --fisheye --buffer=2000 --out_traj_prefix=/home/giovanni/DROIDSLAM_docker/DROID_SLAM/outdoor_forward_10

# Outdoor 45
python demo_vslam.py --datapath=/datasets/UZH_FPV/ --scene=outdoor_45_1_snapdragon_with_gt/img/left --calib=uzhfpv_outdoor_45.txt --fisheye --buffer=2000 --out_traj_prefix=/home/giovanni/DROIDSLAM_docker/DROID_SLAM/outdoor_45_1