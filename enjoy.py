import argparse
import os
import types
import matplotlib.pyplot as plt

import numpy as np
import torch
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

from envs import make_env

from Xlib import display, X
from PIL import Image  # PIL
import time
import pandas as pd
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=2,
                    help='random seed (default: 1)')
parser.add_argument('--num-stack', type=int, default=1,
                    help='number of frames to stack (default: 4)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='Hopper-v2',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-dir', default='./trained_models/ppo/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
args = parser.parse_args()


def generateData(filename="3-20180801-170454-gaurav-msi-64-2-", data_dir="data", N= 1000, loading=True, save_rate=1):
    env = make_env(args.env_name, args.seed, 0, None, args.add_timestep)
    env = DummyVecEnv([env])

    actor_critic, ob_rms = \
                torch.load(os.path.join(os.path.join(args.load_dir, args.env_name), filename + ".pt"))


    if len(env.observation_space.shape) == 1:
        env = VecNormalize(env, ret=False)
        env.ob_rms = ob_rms

        # An ugly hack to remove updates
        def _obfilt(self, obs):
            if self.ob_rms:
                obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
                return obs
            else:
                return obs
        env._obfilt = types.MethodType(_obfilt, env)
        render_func = env.venv.envs[0].render
    else:
        render_func = env.envs[0].render

    obs_shape = env.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
    current_obs = torch.zeros(1, *obs_shape)
    states = torch.zeros(1, actor_critic.state_size)
    masks = torch.zeros(1, 1)


    def update_current_obs(obs):
        shape_dim0 = env.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if args.num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs

    if not loading:
        render_func('human')
        obs = env.reset()
        update_current_obs(obs)

        time.sleep(5)

        if args.env_name.find('Bullet') > -1:
            import pybullet as p

            torsoId = -1
            for i in range(p.getNumBodies()):
                if (p.getBodyInfo(i)[0].decode() == "torso"):
                    torsoId = i

        Xs = torch.empty(N, current_obs.shape[1])
        i = 0

        while True:
            with torch.no_grad():
                value, action, _, states = actor_critic.act(current_obs,
                                                            states,
                                                            masks,
                                                            deterministic=True)

            Xs[i] = current_obs
            i = i+1

            if i % N == 0:
                break

            cpu_actions = action.squeeze(1).cpu().numpy()
            # Obser reward and next obs
            obs, reward, done, _ = env.step(cpu_actions)

            masks.fill_(0.0 if done else 1.0)

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks
            update_current_obs(obs)

            if args.env_name.find('Bullet') > -1:
                if torsoId > -1:
                    distance = 5
                    yaw = 0
                    humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
                    p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

            render_func('human')

            if i % save_rate == 0:
                W, H = 500, 700
                dsp = display.Display()
                root = dsp.screen().root
                raw = root.get_image(500, 150, W, H, X.ZPixmap, 0xffffffff)
                image = Image.frombytes("RGB", (W, H), raw.data, "raw", "BGRX")
                image.save("{}/images/img{}.png".format(data_dir, i), "PNG")

        np.savetxt("{}/X.csv".format(data_dir), Xs.numpy())

    else:
        Xs = torch.Tensor(np.loadtxt("{}/X.csv".format(data_dir)))

    return Xs, obs_shape[0], N


def tsne(X,N, data_dir="data"):
    feat_cols = [ 'feature'+str(i) for i in range(X.shape[1]) ]

    df = pd.DataFrame(X,columns=feat_cols)
    X= None

    rndperm = np.random.permutation(df.shape[0])


    n_sne = N

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    df_tsne = df.loc[rndperm[:n_sne],:].copy()
    df_tsne['x-tsne'] = tsne_results[:,0]
    df_tsne['y-tsne'] = tsne_results[:,1]
    df_tsne['x-bin'] = tsne_results[:,0].astype(int)
    df_tsne['y-bin'] = tsne_results[:, 1].astype(int)
    df_tsne = df_tsne.sort_values(by=['x-bin', 'y-bin']).sort_index()

    df_tsne.to_csv("{}/tsne.csv".format(data_dir))
    return df_tsne

def plot(df_tsne):
    # Create the figure
    fig = plt.figure( figsize=(8,8) )
    ax = fig.add_subplot(1, 1, 1, title='TSNE' )
    # Create the scatter
    colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
    ]

    ax.scatter(
        x=df_tsne['x-tsne'],
        y=df_tsne['y-tsne'],
        c=colors[0],
        cmap=plt.cm.get_cmap('Paired'),
        alpha=0.15)

    plt.show()


data_dir="data"
X, shape, N = generateData(filename="20180801-203140-gaurav-4144-8-2-", data_dir=data_dir, loading=False, N=1000, save_rate=20000)
X = X.view(N, shape).numpy()
get_tsne = True
if get_tsne:
    df_tsne = tsne(X,N, data_dir=data_dir)

df_tsne = pd.read_csv("{}/tsne.csv".format(data_dir), index_col=0)

plot(df_tsne)