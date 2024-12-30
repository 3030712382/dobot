import os
import h5py

# 定义源文件路径和目标文件夹路径
source_file_path = "/home/robot/dobot_xtrainer_diffusion_v1.0.0/datasets/orange/train_data/episode_init_5.hdf5"  # 另一个.hdf5文件，获取新的 'left' 值
target_folder_path = "/home/robot/datasets/dataset_package_test/collect_data"  # 目标文件夹，包含所有要修改的 .hdf5 文件
# source_file_path1 = "/home/robot/datasets/dataset_package_test/collect_data/22.hdf5"
# 读取HDF5文件中的所有数据集
source_file_path1 = "/home/oem/datasets/orangenew-9/8-orange/209.hdf5"
"/home/oem/datasets/dataset_package_test/collect_data/1.hdf5"
source_file_path1 = "/home/oem/datasets/dataset_package_test/collect_data/1.hdf5"
#查看.hdf5文件的属性与详细信息
def traverse_datasets(hdf_file):
    import h5py

    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            # print("key: ",key)
            path = '{}/{}'.format(prefix, key)
            if isinstance(item, h5py.Dataset): # test for dataset
                yield (path, item)
                # print("item: ", item)
            elif isinstance(item, h5py.Group): # test for group (go down)
                yield from h5py_dataset_iterator(item, path)
                # print("item Group:  ", item)

    with h5py.File(hdf_file, 'r') as f:
        import pdb
        # pdb.set_trace()
        for (path, dset) in h5py_dataset_iterator(f):
            print(path, dset)
            # if "action" in path:
            #     print(" ")

    return None

# 传入路径即可
# traverse_datasets('datasets/train_catvnoncat.h5')

def showfile(file_path):
    # 打开HDF5文件
    with h5py.File(file_path, 'r') as file:
        # 打印文件内容
        print(list(file.keys()))

        # 假设我们有一个名为'dataset'的数据集
        dataset = file['action']

        # 打印数据集的属性
        for i in range(len(file["/observations/images/top"])):
            print(file['action'][i])
        # for key, value in dataset.attrs.items():
        #     print(f"Attribute '{key}': {value}")
# 读取源文件中的 'left' 值
def get_left_value_from_source(source_file_path):
    with h5py.File(source_file_path, 'r') as f:
        # 假设 'action' 存储在文件的根目录下
        action = f['action'][0]
        # left_value = action['left'][()]  # 获取 'left' 的数据
    return action

# 修改目标文件中的 'left' 值
def modify_left_in_file(target_file_path, new_left_value):
    with h5py.File(target_file_path, 'r+') as f:
        # 假设 'action' 存储在文件的根目录下
        # 打印数据集的属性
        for i in range(len(f["/observations/images/top"])):
            print(f['action'][i])
            action = f['action'][i]
            print("action now:",action,"last value: ",new_left_value)
            new_left_value[8:] = f['action'][i][8:]
            f['action'][i] = new_left_value
            print("action changed:", f['action'][i], "last value: ", new_left_value)
        # action['left'][...] = new_left_value  # 修改 'left' 的值为新的
# 修改目标文件中的 'left' camera值
def modify_cameraname_in_file(target_file_path):
    with h5py.File(target_file_path, 'r+') as hdf_file:
        try:
            # 检查是否存在需要重命名的数据集
            if '/observations/images/left' in hdf_file:
                hdf_file.move('/observations/images/left', '/observations/images/left_wrist')
                print("已重命名: /observations/images/left -> /observations/images/left_wrist")
            if '/observations/images/right' in hdf_file:
                hdf_file.move('/observations/images/right', '/observations/images/right_wrist')
                print("已重命名: /observations/images/right -> /observations/images/right_wrist")
        except Exception as e:
            print(f"文件 {target_file_path} 修改时出错: {e}")
        # action['left'][...] = new_left_value  # 修改 'left' 的值为新的
#修改路径下的0 - N 的.hdf5文件的名称
def change_hdf5_name(depath,N=73):
    # 遍历文件夹中所有文件
    for i in range(N):  # 从0.hdf5到72.hdf5
        old_filename = os.path.join(depath, f"{i}.hdf5")
        new_filename = os.path.join(depath, f"{377 + i}.hdf5")

        # 检查文件是否存在
        if os.path.exists(old_filename):
            # 重命名文件
            os.rename(old_filename, new_filename)
            print(f"Renamed {old_filename} to {new_filename}")
        else:
            print(f"File {old_filename} does not exist.")
# 获取源文件中的 'left' 值
# new_left_value = get_left_value_from_source(source_file_path)
# print("action back:",new_left_value)
# traverse_datasets(source_file_path)
print("       \n")
# traverse_datasets(source_file_path1)
# modify_left_in_file(source_file_path)
#修改名称
change_hdf5_name("/home/oem/datasets/dataset_package_test/collect_data",N=73)
# 遍历目标文件夹中的所有 .hdf5 文件，并修改
# 'left' 值

# for filename in os.listdir(target_folder_path):
#     if filename.endswith(".hdf5"):
#         target_file_path = os.path.join(target_folder_path, filename)
#         print(f"Modifying {target_file_path}...")
#         modify_cameraname_in_file(target_file_path)

print("Modification completed.")
