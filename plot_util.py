# Copied from https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/visualize_atari.py
# and https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/load.py
# Thanks to the author and OpenAI team!

import glob
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from compare import generateData

matplotlib.rcParams.update({'font.size': 8})


def smooth_reward_curve(x, y):
    # Halfwidth of our smoothing convolution
    halfwidth = min(31, int(np.ceil(len(x) / 30)))
    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
            np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]


def fix_point(x, y, interval):
    np.insert(x, 0, 0)
    np.insert(y, 0, 0)

    fx, fy = [], []
    pointer = 0

    ninterval = int(max(x) / interval + 1)

    for i in range(ninterval):
        tmpx = interval * i

        while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
            pointer += 1

        if pointer + 1 < len(x):
            alpha = (y[pointer + 1] - y[pointer]) / \
                    (x[pointer + 1] - x[pointer])
            tmpy = y[pointer] + alpha * (tmpx - x[pointer])
            fx.append(tmpx)
            fy.append(tmpy)

    return fx, fy


def load_data(indir, smooth, bin_size):
    datas = []
    infiles = glob.glob(indir)

    for inf in infiles:
        with open(inf, 'r') as f:
            f.readline()
            f.readline()
            for line in f:
                tmp = line.split(',')
                t_time = float(tmp[2])
                tmp = [t_time, int(tmp[1]), float(tmp[0])]
                datas.append(tmp)

    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = []
    timesteps = 0
    for i in range(len(datas)):
        result.append([timesteps, datas[i][-1]])
        timesteps += datas[i][1]

    if len(result) < bin_size:
        bin_size = len(result)

    x, y = np.array(result)[:, 0], np.array(result)[:, 1]

    if smooth == 1:
        x, y = smooth_reward_curve(x, y)

    if smooth == 2:
        y = medfilt(y, kernel_size=9)

    x, y = fix_point(x, y, bin_size)
    return [x, y]


color_defaults = [
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


def plot(files, bin_size=2, smooth=1, split=True):
    tys = []
    otx = None
    min_v = 100000000
    for file in files:
        tx, ty = load_data(file, smooth, bin_size)
        if tx is None or ty is None:
            continue
        if len(tx) <= min_v:
            otx = tx
            min_v = len(tx)
        tys.append(ty)

    l = len(min(tys, key=lambda x: len(x)))
    y = np.zeros([len(tys), l])
    for i, j in enumerate(tys):
        y[i][0:l] = j[0:l]

    if split:
        min_v = min_v // 2
        y = np.delete(y, [i for i in range(min_v, y.shape[1])], axis=1)
        otx = otx[:][0:min_v]

    ymax = np.amax(y)
    mean = np.mean(y, axis=0)
    ylast = mean[-1]
    sd = 0.1*np.std(y, axis=0)
    cis = (mean - sd, mean + sd)

    return otx, mean, cis, ymax, ylast


def lfcn(game, file):
    infiles = glob.glob('{}*.monitor.csv'.format(file))
    if len(infiles) > 0:
        tx, mean, cis, m, l = plot(infiles, smooth=1, split=False)
        return m,l

if __name__ == "__main__":
    scna = np.array([2624/2885, 649/2885, 292/2885])*100
    scns = np.array([1980/2885, 423/2885, 159/2885])*100
    lfcna = np.array([2446/2585, 1051/2585,456/2585])*100
    lfcns = np.array([2075/2585, 403/2585, 208/2585])*100

    fig1 = plt.figure(1)
    plt.plot(scna, label="SCN action noise", color="r", marker="v")
    plt.plot(scns, label="SCN state noise", color="r", marker="o")
    plt.plot(lfcna, label="l-FCN action noise", color="b", marker="v")
    plt.plot(lfcns, label="l-FCN state noise", color="b", marker="o")

    print(np.arange(0, 2, step=0.2))
    plt.xticks([0,1, 2], ("0.1", "0.5", "1.0"))
    plt.ylim(0, 101)
    plt.legend()
    plt.xlabel('Injected noise STD')
    plt.ylabel('Resulting performance (%)')
    plt.show()

#games = ["HalfCheetah-v2", "Hopper-v2", "Humanoid-v2", "InvertedDoublePendulum-v2", "InvertedPendulum-v2","Swimmer-v2","Walker2d-v2"]
    #for game in games:
    #    algo = "ppo"

    #    data_dir = "data"
    #    tl = 0
    #    for seed in range(1,6):
    #        file = 'test/{}-{}-'.format(game, seed)
    #        generateData(loading=False, N=5000,save_rate=1000000000, fileSave=file, env_name=game, seed=seed)
    #        m,l = lfcn(game, file)
    #        tl = tl + l
    #    print("{}: mean {}".format(game,tl/5))