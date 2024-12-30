import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import cv2
import time
from dataclasses import dataclass
import numpy as np
import tyro
import threading
from dobot_control.agents.agent import BimanualAgent
from scripts.format_obs import save_frame
from dobot_control.env import RobotEnv
from dobot_control.robots.robot_node import ZMQClientRobot
from scripts.function_util import mismatch_data_write, wait_period, log_write, mk_dir
from scripts.manipulate_utils import robot_pose_init, pose_check, dynamic_approach, obs_action_check, servo_action_check, load_ini_data_hands, set_light, load_ini_data_camera
from dobot_control.agents.dobot_agent import DobotAgent
from dobot_control.cameras.realsense_camera import RealSenseCamera
import datetime
from pathlib import Path
import requests

is_sensor_version = False
# _, hands_dict = load_ini_data_hands()
# print("left hand ids: ", hands_dict["HAND_LEFT"].joint_ids)
# print("right hand ids: ", hands_dict["HAND_RIGHT"].joint_ids)
# left_agent = DobotAgent(which_hand="HAND_LEFT", dobot_config=hands_dict["HAND_LEFT"])
# right_agent = DobotAgent(which_hand="HAND_RIGHT", dobot_config=hands_dict["HAND_RIGHT"])
#
# # set torque
# right_agent.set_torque(True)
# left_agent.set_torque(True)

# while 1:
#     print(left_agent.act({}), right_agent.act({}))  # joints
#     print(left_agent.get_keys(), right_agent.get_keys())  # button status

def run_thread_cam(rs_cam, which_cam):
    global image_left, image_right, image_top, thread_run
    if which_cam==0:
        while thread_run:
            image_left, _ = rs_cam.read()
            image_left = image_left[:, :, ::-1]
    elif which_cam==1:
        while thread_run:
            image_right, _ = rs_cam.read()
            image_right = image_right[:, :, ::-1]
    elif which_cam==2:
        while thread_run:
            image_top, _ = rs_cam.read()
            image_top = image_top[:, :, ::-1]
    else:
        print("Camera index error! ")


@dataclass
class Args:
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    show_img: bool = False
    save_data_path = str(Path(__file__).parent.parent.parent)+"/datasets/"
    project_name = "dataset_package_test"
def get_robot_type(robot_ip):
    response = requests.post("http://"+robot_ip+":22000/properties/controllerType")
    if response.status_code == 200:
        print("The type of robot is:", eval(response.text)["name"])
        return eval(response.text)["name"]
    else:
        print("Failed to obtain the type of robot")
        return None

def get_firmware_version_satisfied(robot_ip):
    try:
        response = requests.post("http://"+robot_ip+":22000/settings/version")
        if response.status_code == 200:
            rt_version = response.text.split("{")[1].split("\n\t")[3].split(":")[1].split("\"")[1].split("-")[0].split(".")
            rt_num = "".join(rt_version)
            return 1, int(rt_num)
        else:
            print("Failed to retrieve the version webpage")
            return 0, 0
    except Exception as e:
        print(e)
        return 0, 0
def check_firmware_version():
    left_version = get_firmware_version_satisfied("192.168.5.1")
    right_version = get_firmware_version_satisfied("192.168.5.2")
    if left_version[1] < 3581 or left_version[1]>=4000:
        print("[ERROR]Left hand error[192.168.5.1]:firmware version requires V3 and must >=3.5.8.1 (found: %s),please check and update"%left_version[1])
        return False
    if right_version[1] < 3581 or right_version[1]>=4000:
        print("[ERROR]Right hand error[192.168.5.1]:firmware version requires V3 and must >=3.5.8.1 (found: {%s}),please check and update"% right_version[1])
        return False
    return True
def main(args):
    # pose init
    #机械臂初始化
    print("Waiting to connect the robot...")
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    print("If the robot fails to initialize successfully after 5 seconds,please check that the robot network is connected correctly and make sure TCP/IP mode is turned!")
    if check_firmware_version()==False:
        return
    robot_type = get_robot_type("192.168.5.1")
    print("robot env init ....")
    env = RobotEnv(robot_client)
    env.set_do_status([1, 0])
    env.set_do_status([2, 0])
    env.set_do_status([3, 0])
    print("robot env init success....")
    robot_pose_init(env)
    start_servo = False
    curr_light = "dark"
    print("robot init success....")

if __name__ == "__main__":
    main(tyro.cli(Args))