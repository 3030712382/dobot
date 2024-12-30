# -*-encoding:utf-8-*-
import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py
import IPython
import cv2
import queue

e = IPython.embed


class hdf5data:
    def __init__(self, dataset_dir="/home/hp-t4/data/berthing", episode_idx=50, camera_names=["front"],hdf5_file = None):
        """
        For each timestep:
        - action                  (14,)         'float64'   [throttle rudder] 油门舵角
        - targetstate               (14,)         'float64'   [u  yaw] 期望航速航向
        observations:
        - images
            - each_cam_name     (480, 640, 3) 'uint8'
        - qpos                    (14,)         'float64' [x y z pitch roll yaw] 无人艇状态：三轴位置及角度
        - qvel                      (14,)         'float64' [u v r dpitch droll dyaw] 无人艇速度
        # - motion                  (2,)         'float64'   [thrust rudder] 动力系统：转速舵角
        """
        self.dataset_dir = dataset_dir
        self.hd_dir = "episode_1"
        self.episode_idx = episode_idx
        self.camera_names = camera_names
        self.data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        self.COMPRESS = True
        # self.compressed_len = 0
        self.image_dict = dict()
        for cam_name in self.camera_names:
            self.data_dict[f'/observations/images/{cam_name}'] = []
            self.image_dict[cam_name] = []
        self.padded_size = 0
        self.steps = 0
        if hdf5_file != None:
            self.hdf5_file = hdf5_file
            self.data_queue = queue.Queue()  # 用于生产者和消费者之间的数据传递
    def update_que(self,pos, vel, action, cameras):
        data = {
            'qpos': pos,
            'qvel': vel,
            'action': action,
            'cameras': {cam: cameras[cam] for cam in self.camera_names},
        }
        self.data_queue.put(data)
    def _consumer_worker(self):
        """消费者线程，从队列中取数据并保存到 HDF5 文件"""
        with h5py.File(self.hdf5_file, 'a', rdcc_nbytes=1024 ** 2 * 2) as f:
            while not self.stop_event.is_set() or not self.data_queue.empty():
                try:
                    data = self.data_queue.get(timeout=1)  # 等待数据
                except queue.Empty:
                    continue

                # 保存数据到 HDF5 文件
                self._save_to_hdf5(f, data)

    def _save_to_hdf5(self, hdf5_file, data):
        """保存数据到 HDF5 文件"""
        if '/observations/qpos' not in hdf5_file:
            # 创建数据集
            hdf5_file.create_dataset('/observations/qpos', data=np.array([data['qpos']]), maxshape=(None, len(data['qpos'])))
            hdf5_file.create_dataset('/observations/qvel', data=np.array([data['qvel']]), maxshape=(None, len(data['qvel'])))
            hdf5_file.create_dataset('/action', data=np.array([data['action']]), maxshape=(None, len(data['action'])))
            for cam_name in self.camera_names:
                hdf5_file.create_dataset(f'/observations/images/{cam_name}', data=np.array([data['cameras'][cam_name]]), maxshape=(None, *data['cameras'][cam_name].shape))
        else:
            # 追加数据
            hdf5_file['/observations/qpos'].resize(hdf5_file['/observations/qpos'].shape[0] + 1, axis=0)
            hdf5_file['/observations/qpos'][-1] = data['qpos']

            hdf5_file['/observations/qvel'].resize(hdf5_file['/observations/qvel'].shape[0] + 1, axis=0)
            hdf5_file['/observations/qvel'][-1] = data['qvel']

            hdf5_file['/action'].resize(hdf5_file['/action'].shape[0] + 1, axis=0)
            hdf5_file['/action'][-1] = data['action']

            for cam_name in self.camera_names:
                hdf5_file[f'/observations/images/{cam_name}'].resize(hdf5_file[f'/observations/images/{cam_name}'].shape[0] + 1, axis=0)
                hdf5_file[f'/observations/images/{cam_name}'][-1] = data['cameras'][cam_name]

    def update_data(self, pos, vel, action, cameras):
        self.data_dict['/observations/qpos'].append(pos)
        self.data_dict['/observations/qvel'].append(vel)
        self.data_dict['/action'].append(action)
        for cam_name in self.camera_names:
            self.data_dict[f'/observations/images/{cam_name}'].append(cameras[cam_name])
        # cv2.imshow("0",
        #            np.asarray((self.data_dict[f'/observations/images/right_wrist'][-1]), dtype="uint8"))  # right_wrist
        # cv2.imshow("0", np.asarray((self.data_dict[f'/observations/images/{cam_name}'][-1]), dtype="uint8")) #right_wrist
        # np.asarray(camerasall["right_wrist"], dtype="uint8")
        # cv2.waitKey(1)
        # cv2.imwrite("/home/oem/datasets/dataset_package_test/collect_data/image/{}".format(self.steps)+".png",np.asarray((self.data_dict[f'/observations/images/right_wrist'][-1]), dtype="uint8"))
        # self.steps+=1
        print("len action: ",len(self.data_dict['/action']))
        print("len right_wrist: ", len(self.data_dict[f'/observations/images/right_wrist']))
        # if self.COMPRESS:
        #     for cam_name in self.camera_names:
        #         self.image_dict[cam_name].append(cameras[cam_name])
        # else:
        #     for cam_name in self.camera_names:
        #         self.data_dict[f'/observations/images/{cam_name}'].append(cameras[cam_name])

    def save_data(self,dataset_path):
        # HDF5
        t0 = time.time()
        max_timesteps = len(self.data_dict['/action'])
        # iop = 0
        # for img1 in self.data_dict[f'/observations/images/top']:
        #     # np.asarray(, dtype="uint8")
        #     # cv2.imshow("0", np.asarray(img1, dtype="uint8"))
        #     cv2.imwrite("/home/oem/datasets/dataset_package_test/collect_data/image/{}".format(iop)+".png",np.asarray(img1, dtype="uint8"))
        #     iop+=1
            # np.asarray(camerasall["right_wrist"], dtype="uint8")
            # cv2.waitKey(10)
        # for cam_name in self.image_dict.keys():
        #     self.data_dict[f'/observations/images/{cam_name}'] = self.image_dict[cam_name]
        # dataset_path = os.path.join(self.dataset_dir, self.hd_dir)
        # dataset_path = os.path.join(self.dataset_dir, f'episode_{self.episode_idx}')#保存在指定路径下 第episode_idx轮数据
        # with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        import pdb
        # pdb.set_trace()
        with h5py.File(dataset_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = False
            root.attrs['compress'] = self.COMPRESS
            obs = root.create_group('observations')
            imagedata = obs.create_group('images')

            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))
            if self.COMPRESS:
                # JPEG compression
                t0 = time.time()
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # tried as low as 20, seems fine
                compressed_len = []
                for cam_name in self.camera_names:
                    image_list = self.data_dict[f'/observations/images/{cam_name}']
                    compressed_list = []
                    compressed_len.append([])
                    for image in image_list:
                        # if cam_name == "top":
                        #     cv2.imshow("0", np.asarray(image, dtype="uint8"))
                        #     # np.asarray(camerasall["right_wrist"], dtype="uint8")
                        #     cv2.waitKey(1)
                            # pdb.set_trace()
                        result, encoded_image = cv2.imencode('.jpg', image,
                                                             encode_param)  # 0.02 sec # cv2.imdecode(encoded_image, 1)
                        compressed_list.append(encoded_image)
                        compressed_len[-1].append(len(encoded_image))
                    self.data_dict[f'/observations/images/{cam_name}'] = compressed_list
                print(f'compression: {time.time() - t0:.2f}s')

                # pad so it has same length
                t0 = time.time()
                compressed_len = np.array(compressed_len)
                self.padded_size = compressed_len.max()
                for cam_name in self.camera_names:
                    compressed_image_list = self.data_dict[f'/observations/images/{cam_name}']
                    padded_compressed_image_list = []
                    for compressed_image in compressed_image_list:
                        padded_compressed_image = np.zeros(self.padded_size, dtype='uint8')
                        image_len = len(compressed_image)
                        padded_compressed_image[:image_len] = compressed_image
                        padded_compressed_image_list.append(padded_compressed_image)
                    self.data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list
                print(f'padding: {time.time() - t0:.2f}s')

            if self.COMPRESS:
                _ = root.create_dataset('compress_len', (len(self.camera_names), max_timesteps))
                root['/compress_len'][...] = compressed_len
            for cam_name in self.camera_names:
                if self.COMPRESS:
                    _ = imagedata.create_dataset(cam_name, (max_timesteps, self.padded_size), dtype='uint8',
                                             chunks=(1, self.padded_size), )
                else:
                    _ = imagedata.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                             chunks=(1, 480, 640, 3), )
            for name, array in self.data_dict.items():
                print(name)
                root[name][...] = array
                self.data_dict[name] = []
        print(f'Saved to {self.dataset_dir}')
        # self.data_dict.clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--onscreen_render', action='store_true')

    # main(vars(parser.parse_args()))
