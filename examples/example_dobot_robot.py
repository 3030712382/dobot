from dobot_control.robots.dobot import DobotRobot
from scripts.manipulate_utils import robot_pose_init, pose_check, dynamic_approach, obs_action_check, servo_action_check, load_ini_data_hands, set_light

dobot = DobotRobot("192.168.5.2", no_gripper=False)
dobot1 = DobotRobot("192.168.5.1", no_gripper=False)
err = dobot.r_inter.ClearError()#清除错误
err = dobot1.r_inter.ClearError()#清除错误 PowerOn
dobot.r_inter.PowerOn()#上电
dobot1.r_inter.PowerOn()#上电
dobot.r_inter.DisableRobot()#下使能
dobot1.r_inter.DisableRobot()#下使能err = dobot.r_inter
# dobot.r_inter.StartDrag()
# print(err)
set_light(dobot, "green", 1)
while 1:
    print("0 warn: ",dobot.robot_status.get_error())
    print("1 warn: ",dobot1.robot_status.get_error())
