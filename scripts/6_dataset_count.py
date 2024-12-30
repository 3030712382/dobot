import os
import glob
import numpy as np
from scripts.function_util import mk_dir
from pathlib import Path
import tyro
from dataclasses import dataclass


# @dataclass
class Args:
    dataset_name: str = "dataset_package_test"
    # dataset_name: str = "dataset_package_test"
    task_name: str = "orange"
    MIRROR_STATE_MULTIPLY: list = (1, 1, 1, 1, 1, 1, 1)
    MIRROR_BASE_MULTIPLY: tuple = (1, 1)

def main(args):
    root_dir = str(Path(__file__).parent.parent.parent / "datasets/")
    dataset_dir = root_dir + "/" + args.dataset_name + "/collect_data/"
    mk_dir(dataset_dir)
    print(dataset_dir)
    output_video_dir = root_dir + "/" + args.dataset_name + "/output_videos/"
    mk_dir(output_video_dir)
    output_train_data = root_dir + "/" + args.dataset_name + "/train_data/"
    mk_dir(output_train_data)

    all_data_dir = os.listdir(dataset_dir)
    # print(all_data_dir)
    all_data_dir.sort(key=lambda x: int(x))
    # print(all_data_dir)
    MIRROR_STATE_MULTIPLY = np.array(args.MIRROR_STATE_MULTIPLY)
    max_step = 0
    for idx in range(len(all_data_dir)):
        one_data_dir = dataset_dir+all_data_dir[idx]+"/"
        data_pose_list = glob.glob(one_data_dir + 'observation/*.pkl')
        if len(data_pose_list) > max_step:
            max_step = len(data_pose_list)
    print(max_step)


if __name__ == "__main__":
    main(tyro.cli(Args))
