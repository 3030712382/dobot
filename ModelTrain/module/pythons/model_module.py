# uncompyle6 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.20 (default, Oct  3 2024, 15:24:27) 
# [GCC 11.2.0]
# Embedded file name: model_inference.py
# Compiled at: 2024-05-09 16:05:43
# Size of source mod 2**32: 15254 bytes
import torch, numpy as np, os, pickle
from einops import rearrange
import matplotlib.pyplot as plt
import time
from torchvision import transforms
from module.policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from detr.models.latent_model import Latent_Model_Transformer
from ModelTrain.model_train import arg_config

def set_config():
    args = arg_config()
    ckpt_dir = args["ckpt_dir"]
    policy_class = "ACT"
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]
    num_steps = args["num_steps"]
    eval_every = args["eval_every"]
    validate_every = args["validate_every"]
    save_every = args["save_every"]
    resume_ckpt_path = args["resume_ckpt_path"]
    is_sim = task_name[None[:4]] == "sim_"
    if is_sim or task_name == "all":
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config["dataset_dir"]
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]
    stats_dir = task_config.get("stats_dir", None)
    sample_weights = task_config.get("sample_weights", None)
    train_ratio = task_config.get("train_ratio", 0.99)
    name_filter = task_config.get("name_filter", lambda n: True)
    state_dim = 14
    lr_backbone = 1e-05
    backbone = "resnet18"
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr':args["lr"],  'num_queries':args["chunk_size"], 
         'kl_weight':args["kl_weight"], 
         'hidden_dim':args["hidden_dim"], 
         'dim_feedforward':args["dim_feedforward"], 
         'lr_backbone':lr_backbone, 
         'backbone':backbone, 
         'enc_layers':enc_layers, 
         'dec_layers':dec_layers, 
         'nheads':nheads, 
         'camera_names':camera_names, 
         'vq':False, 
         'vq_class':None, 
         'vq_dim':None, 
         'action_dim':16, 
         'no_encoder':args["no_encoder"]}
    else:
        if policy_class == "Diffusion":
            policy_config = {'lr':args["lr"],  'camera_names':camera_names, 
             'action_dim':16, 
             'observation_horizon':1, 
             'action_horizon':8, 
             'prediction_horizon':args["chunk_size"], 
             'num_queries':args["chunk_size"], 
             'num_inference_timesteps':10, 
             'ema_power':0.75, 
             'vq':False}
        else:
            if policy_class == "CNNMLP":
                policy_config = {'lr':args["lr"], 
                 'lr_backbone':lr_backbone,  'backbone':backbone,  'num_queries':1,  'camera_names':camera_names}
            else:
                raise NotImplementedError
    config = {'num_steps':num_steps,  'eval_every':eval_every, 
     'validate_every':validate_every, 
     'save_every':save_every, 
     'ckpt_dir':ckpt_dir, 
     'resume_ckpt_path':resume_ckpt_path, 
     'episode_len':episode_len, 
     'state_dim':state_dim, 
     'lr':args["lr"], 
     'policy_class':policy_class, 
     'policy_config':policy_config, 
     'task_name':task_name, 
     'seed':args["seed"], 
     'temporal_agg':args["temporal_agg"], 
     'camera_names':camera_names, 
     'real_robot':not is_sim, 
     'load_pretrain':args["load_pretrain"]}
    return config


class Imitate_Model:

    def __init__(self, ckpt_dir=None, ckpt_name='policy_last.ckpt'):
        config = set_config()
        self.ckpt_name = ckpt_name
        if ckpt_dir == None:
            self.ckpt_dir = config["ckpt_dir"]
            print(self.ckpt_dir)
        else:
            self.ckpt_dir = ckpt_dir
        self.state_dim = config["state_dim"]
        self.policy_class = config["policy_class"]
        self.policy_config = config["policy_config"]
        self.camera_names = config["camera_names"]
        self.max_timesteps = config["episode_len"]
        self.temporal_agg = config["temporal_agg"]
        self.vq = config["policy_config"]["vq"]
        self.t = 0

    def __make_policy(self):
        if self.policy_class == "ACT":
            policy = ACTPolicy(self.policy_config)
        else:
            if self.policy_class == "CNNMLP":
                policy = CNNMLPPolicy(self.policy_config)
            else:
                if self.policy_class == "Diffusion":
                    policy = DiffusionPolicy(self.policy_config)
                else:
                    raise NotImplementedError
        return policy

    def __image_process(self, observation, camera_names, rand_crop_resize=False):
        curr_images = []
        for cam_name in camera_names:
            curr_image = rearrange(observation["images"][cam_name], "h w c -> c h w")
            curr_images.append(curr_image)
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        if rand_crop_resize:
            print("rand crop resize is used!")
            original_size = curr_image.shape[(-2)[:None]]
            ratio = 0.95
            curr_image = curr_image[(...,
             int(original_size[0] * (1 - ratio) / 2)[:int(original_size[0] * (1 + ratio) / 2)],
             int(original_size[1] * (1 - ratio) / 2)[:int(original_size[1] * (1 + ratio) / 2)])]
            curr_image = curr_image.squeeze(0)
            resize_transform = transforms.Resize(original_size, antialias=True)
            curr_image = resize_transform(curr_image)
            curr_image = curr_image.unsqueeze(0)
        return curr_image

    def __get_auto_index(self, dataset_dir):
        max_idx = 1000
        for i in range(max_idx + 1):
            if not os.path.isfile(os.path.join(dataset_dir, f"qpos_{i}.npy")):
                return i
        raise Exception(f"Error getting auto index, or more than {max_idx} episodes")

    def loadModel(self):
        cur_path = os.path.dirname(os.path.abspath(__file__))
        dir_path = os.path.dirname(cur_path)
        ckpt_path = os.path.join(self.ckpt_dir, self.ckpt_name)
        ckpt_path = dir_path + ckpt_path[1[:None]]
        self.policy = self._Imitate_Model__make_policy()
        loading_status = self.policy.deserialize(torch.load(ckpt_path))
        print(loading_status)
        self.policy.cuda()
        self.policy.eval()
        if self.vq:
            vq_dim = self.config["policy_config"]["vq_dim"]
            vq_class = self.config["policy_config"]["vq_class"]
            latent_model = Latent_Model_Transformer(vq_dim, vq_dim, vq_class)
            latent_model_ckpt_path = os.path.join(self.ckpt_dir, "latent_model_last.ckpt")
            latent_model.deserialize(torch.load(latent_model_ckpt_path))
            latent_model.eval()
            latent_model.cuda()
            print(f"Loaded policy from: {ckpt_path}, latent model from: {latent_model_ckpt_path}")
        else:
            print(f"Loaded: {ckpt_path}")
        stats_path = os.path.join(dir_path + self.ckpt_dir[1[:None]], "dataset_stats.pkl")
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        self.pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
        if self.policy_class == "Diffusion":
            self.post_process = lambda a: (a + 1) / 2 * (stats["action_max"] - stats["action_min"]) + stats["action_min"]
        else:
            self.post_process = lambda a: a * stats["action_std"] + stats["action_mean"]
        self.query_frequency = self.policy_config["num_queries"]
        if self.temporal_agg:
            self.query_frequency = 1
            self.num_queries = self.policy_config["num_queries"]
        self.max_timesteps = int(self.max_timesteps * 1)
        self.episode_returns = []
        self.highest_rewards = []
        if self.temporal_agg:
            self.all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps + self.num_queries, 16]).cuda()
        self.qpos_history_raw = np.zeros((self.max_timesteps, self.state_dim))
        self.image_list = []
        self.qpos_list = []
        self.target_qpos_list = []
        self.rewards = []
        self.all_actions = []

    def predict(self, observation, t, save_qpos_history=False):
        with torch.inference_mode():
            qpos_numpy = np.array(observation["qpos"])
            self.qpos_history_raw[t] = qpos_numpy
            qpos = self.pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            if t % self.query_frequency == 0:
                curr_image = self._Imitate_Model__image_process(observation, (self.camera_names), rand_crop_resize=(self.policy_class == "Diffusion"))
            elif t == 0:
                for _ in range(10):
                    self.policy(qpos, curr_image)
                print("network warm up done")
                time1 = time.time()

            elif self.policy_class == "ACT":
                if t % self.query_frequency == 0:
                    if self.vq:
                        self.vq_sample = self.latent_model.generate(1, temperature=1, x=None)
                        self.all_actions = self.policy(qpos, curr_image, vq_sample=(self.vq_sample))
                    else:
                        self.all_actions = self.policy(qpos, curr_image)
                elif self.temporal_agg:
                    self.all_time_actions[([t], t[:t + self.num_queries])] = self.all_actions
                    actions_for_curr_step = self.all_time_actions[(None[:None], t)]
                    actions_populated = torch.all((actions_for_curr_step != 0), axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = self.all_actions[(None[:None], t % self.query_frequency)]
            else:
                if self.config["policy_class"] == "Diffusion":
                    if t % self.query_frequency == 0:
                        self.all_actions = self.policy(qpos, curr_image)
                    raw_action = self.all_actions[(None[:None], t % self.query_frequency)]
                else:
                    if self.config["policy_class"] == "CNNMLP":
                        raw_action = self.policy(qpos, curr_image)
                        self.all_actions = raw_action.unsqueeze(0)
                    else:
                        raise NotImplementedError
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = self.post_process(raw_action)
            target_qpos = action[None[:-2]]
            base_action = action[(-2)[:None]]
            self.qpos_list.append(qpos_numpy)
            self.target_qpos_list.append(target_qpos)
            if save_qpos_history:
                log_id = self._Imitate_Model__get_auto_index(self.ckpt_dir)
                np.save(os.path.join(self.ckpt_dir, f"qpos_{log_id}.npy"), self.qpos_history_raw)
                plt.figure(figsize=(10, 20))
                for i in range(self.state_dim):
                    plt.subplot(self.state_dim, 1, i + 1)
                    plt.plot(self.qpos_history_raw[(None[:None], i)])
                    if i != self.state_dim - 1:
                        plt.xticks([])
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.ckpt_dir, f"qpos_{log_id}.png"))
                    plt.close()

        return target_qpos
