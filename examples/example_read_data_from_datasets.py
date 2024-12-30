import h5py
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from sympy.physics.units import action

patyh= "/home/oem/datasets/orangenew-9/8-orange/210.hdf5"
# /home/oem/datasets/dataset_package_test
patyh= "/home/oem/datasets/dataset_package_test/collect_data/379.hdf5" #episode_init_406
patyh= "/home/oem/datasets/car_bat/collect_data/2.hdf5"
show_canvas = np.zeros((480, 640*3, 3), dtype=np.uint8)
with h5py.File( patyh, 'r',
               rdcc_nbytes=1024 ** 2 * 2) as root:
    print(root["observations"].keys())
    print("len: ",len(root["/observations/images/top"]))
    action1 = []
    for i in range(len(root["/observations/images/top"])):
        qpos = root["/observations/qpos"][i]
        print("step", i)
        # observation, joints angle, gripper width and images
        print("observation: left hand [J1, J2, J3, J4, J5, J6, gripper_width]:", [i for i in qpos[:7]])
        print("observation: right hand [J1, J2, J3, J4, J5, J6, gripper_width]:", [i for i in qpos[7:14]])
        # show_canvas[:, :640] = root["/observations/images/top"][i]
        # show_canvas[:, 640:640 * 2] =root["/observations/images/left_wrist"][i]
        # show_canvas[:, 640 * 2:640 * 3] = root["/observations/images/right_wrist"][i]
        show_canvas[:, :640] = np.asarray(
            cv2.imdecode(np.asarray(root["/observations/images/top"][i], dtype="uint8"), cv2.IMREAD_COLOR),
            dtype="uint8")
        show_canvas[:, 640:640 * 2] = np.asarray(
            cv2.imdecode(np.asarray(root["/observations/images/left_wrist"][i], dtype="uint8"), cv2.IMREAD_COLOR),
            dtype="uint8")
        show_canvas[:, 640 * 2:640 * 3] = np.asarray(
            cv2.imdecode(np.asarray(root["/observations/images/right_wrist"][i], dtype="uint8"), cv2.IMREAD_COLOR),
            dtype="uint8")
        cv2.imshow("0", show_canvas)

        # predict joint angle, gripper width
        action = root["action"][i]
        action1.append(qpos)
        print("predict action: left hand [J1, J2, J3, J4, J5, J6, gripper_width]:", [i for i in action[:7]])
        print("predict action: right hand [J1, J2, J3, J4, J5, J6, gripper_width]:", [i for i in action[7:14]])
        print()
        cv2.waitKey(0)
    plt.plot(action1)
    plt.show()


