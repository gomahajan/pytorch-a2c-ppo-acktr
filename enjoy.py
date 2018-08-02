import argparse
import os
import types

import numpy as np
import torch
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

from envs import make_env


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

filename = "3-20180801-170454-gaurav-msi-64-2-"


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


render_func('human')
obs = env.reset()
update_current_obs(obs)

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

N = 1000
print(current_obs.shape)
print(current_obs.shape[1])
X = torch.empty(N, current_obs.shape[1])
y = torch.empty(N, 1)
i = 0

while True:
    with torch.no_grad():
        value, action, choice, _, choice_log_probs, states = actor_critic.act(current_obs,
                                                    states,
                                                    masks,
                                                    deterministic=True)

    X[i] = current_obs
    y[i] = choice
    i = i+1

    if i % N == 0:
        break;

    cpu_actions = action.squeeze(1).cpu().numpy()
    #print("Actions: {} for choice {}".format(action, choice))
    print("Choice {} with probability {}".format(choice, torch.exp(choice_log_probs)))
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

import pandas as pd

X = X.view(N, current_obs.shape[1]).numpy()
y = y.view(N).numpy()
print(X)
print(y)

feat_cols = [ 'feature'+str(i) for i in range(X.shape[1]) ]

df = pd.DataFrame(X,columns=feat_cols)
df['label'] = y
df['label'] = df['label'].apply(lambda i: str(i))

X, y = None, None

rndperm = np.random.permutation(df.shape[0])


import time

from sklearn.manifold import TSNE

n_sne = N

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_tsne = df.loc[rndperm[:n_sne],:].copy()
df_tsne['x-tsne'] = tsne_results[:,0]
df_tsne['y-tsne'] = tsne_results[:,1]

print("Index")
print(df.index)
print(df['label'])
print(df['label'] == str(float(1)))
import matplotlib.pyplot as plt
# Create the figure
fig = plt.figure( figsize=(8,8) )
ax = fig.add_subplot(1, 1, 1, title='TSNE' )
# Create the scatter
colors = ['red', 'green', 'blue']
print()
for i in range(0,3):
    ax.scatter(
        x=df_tsne.loc[df['label'] == str(float(i))]['x-tsne'],
        y=df_tsne.loc[df['label'] == str(float(i))]['y-tsne'],
        c=colors[i],
        cmap=plt.cm.get_cmap('Paired'),
        alpha=0.15,
        label="Actor {}".format(i))

plt.legend()
plt.show()