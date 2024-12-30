# uncompyle6 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.20 (default, Oct  3 2024, 15:24:27) 
# [GCC 11.2.0]
# Embedded file name: train_module.py
# Compiled at: 2024-05-09 16:09:15
# Size of source mod 2**32: 16468 bytes
import os, torch, numpy as np, pickle, argparse
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt
import time
from torchvision import transforms
from ModelTrain.module.utils import load_data
from ModelTrain.module.utils import compute_dict_mean, set_seed, detach_dict, calibrate_linear_vel, postprocess_base_action
from ModelTrain.module.policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
import IPython
e = IPython.embed

def get_auto_index(dataset_dir):
    max_idx = 1000
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'qpos_{i}.npy')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def train(args):
    set_seed(1)
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
    from ModelTrain.constants import TASK_CONFIGS
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
     'load_pretrain':args["load_pretrain"]}
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    config_path = os.path.join(ckpt_dir, "config.pkl")
    expr_name = ckpt_dir.split("/")[-1]
    with open(config_path, "wb") as f:
        pickle.dump(config, f)
    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, name_filter, camera_names, batch_size_train, batch_size_val, (args["chunk_size"]), (args["skip_mirrored_data"]), (config["load_pretrain"]), policy_class, stats_dir_l=stats_dir, sample_weights=sample_weights, train_ratio=train_ratio)
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_step, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}")


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    else:
        if policy_class == "CNNMLP":
            policy = CNNMLPPolicy(policy_config)
        else:
            if policy_class == "Diffusion":
                policy = DiffusionPolicy(policy_config)
            else:
                raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == "ACT":
        optimizer = policy.configure_optimizers()
    else:
        if policy_class == "CNNMLP":
            optimizer = policy.configure_optimizers()
        else:
            if policy_class == "Diffusion":
                optimizer = policy.configure_optimizers()
            else:
                raise NotImplementedError
    return optimizer


def get_image(ts, camera_names, rand_crop_resize=False):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    if rand_crop_resize:
        print("rand crop resize is used!")
        original_size = curr_image.shape[(-2)[:None]]
        ratio = 0.95
        curr_image = curr_image[(..., int(original_size[0] * (1 - ratio) / 2)[:int(original_size[0] * (1 + ratio) / 2)],
         int(original_size[1] * (1 - ratio) / 2)[:int(original_size[1] * (1 + ratio) / 2)])]
        curr_image = curr_image.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        curr_image = resize_transform(curr_image)
        curr_image = curr_image.unsqueeze(0)
    return curr_image


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = (
     image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda())
    return policy(qpos_data, image_data, action_data, is_pad)


def train_bc(train_dataloader, val_dataloader, config):
    num_steps = config["num_steps"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    eval_every = config["eval_every"]
    validate_every = config["validate_every"]
    save_every = config["save_every"]
    set_seed(seed)
    policy = make_policy(policy_class, policy_config)
    if config["load_pretrain"]: #deserialize :反序列化
        loading_status = policy.deserialize(torch.load(os.path.join("/home/interbotix_ws/src/act/ckpts/pretrain_all", "policy_step_50000_seed_0.ckpt")))
        print(f"loaded! {loading_status}")
    if config["resume_ckpt_path"] is not None:
        loading_status = policy.deserialize(torch.load(config["resume_ckpt_path"]))
        print(f'Resume policy from: {config["resume_ckpt_path"]}, Status: {loading_status}')
    optimizer = make_optimizer(policy_class, policy)
    policy.cuda()
    min_val_loss = np.inf
    best_ckpt_info = None
    train_dataloader = repeater(train_dataloader)
    train_loss = []
    val_loss = []
    last_time = time.time()
    start_time = last_time
    for step in tqdm(range(num_steps + 1)):
        if step % validate_every == 0:
            print("validating")
            with torch.inference_mode():
                policy.eval()
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)
                    validation_dicts.append(forward_dict)
                    if batch_idx > 50:
                        break
                validation_summary = compute_dict_mean(validation_dicts)
                epoch_val_loss = validation_summary["loss"].mean()
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (step, min_val_loss, deepcopy(policy.serialize()))

            for k in list(validation_summary.keys()):
                validation_summary[f"val_{k}"] = validation_summary.pop(k)

            print(f"Val loss:   {epoch_val_loss:.5f}")
            val_loss.append(float(epoch_val_loss.item()))
            summary_string = ""
            for k, v in validation_summary.items():
                summary_string += f"{k}: {v.mean().item():.3f} "
            print(summary_string)
        # evaluation
        if step > 0:
            if step % eval_every == 0:
                ckpt_name = f"policy_step_{step}_seed_{seed}.ckpt"
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        policy.train()
        optimizer.zero_grad()
        data = next(train_dataloader)
        forward_dict = forward_pass(data, policy)
        loss = forward_dict["loss"]
        loss.mean().backward()
        optimizer.step()
        train_loss.append(float(loss.mean().item()))

        if step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_step_{step}_seed_{seed}.ckpt")
            torch.save(policy.serialize(), ckpt_path)
    cur_time = time.time()
    last_time = cur_time
    # else:
    print("train all time:", cur_time - start_time)
    ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
    torch.save(policy.serialize(), ckpt_path)
    best_step, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f"policy_step_{best_step}_seed_{seed}.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at step {best_step}")
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Training Loss")
    plt.title("Training Loss Over Steps")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(ckpt_dir + "/train_loss.png")
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(val_loss, label="val Loss")
    plt.title("val Loss Over Steps")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(ckpt_dir + "/val_loss.png")
    plt.show()
    return best_ckpt_info


def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f'Epoch {epoch} done')
        epoch += 1
