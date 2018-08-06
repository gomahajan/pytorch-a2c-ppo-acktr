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
        return [None, None]

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


def plot(files, bin_size=100, smooth=1, split=True):
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
    sd = 0.1*np.std(y, axis=0)
    cis = (mean - sd, mean + sd)

    return otx, mean, cis, ymax


def linear(algo, game, plt, c):
    infiles = glob.glob('/home/gaurav/PycharmProjects/Atari35/linear/graphs/{}/{}/'.format(algo,game)+'*0.monitor.csv')
    if len(infiles) > 0:
        tx, mean, cis, m = plot(infiles, smooth=1, split=False)
        plt.fill_between(tx, cis[0], cis[1], alpha=0.5, color=c)
        plt.plot(tx, mean, label="Linear", color=c)
        return m

def scn(algo, game, plt, c):
    infiles = glob.glob('/home/gaurav/PycharmProjects/Atari35/scn/graphs/{}/{}/'.format(algo,game)+'*0.monitor.csv')
    if len(infiles) > 0:
        tx, mean, cis, m = plot(infiles, smooth=1, split=False)
        plt.fill_between(tx, cis[0], cis[1], alpha=0.5, color=c)
        plt.plot(tx, mean, label="SCN-64", color=c)
        return m

def scn16(algo, game, plt,c):
    infiles = glob.glob('/home/gaurav/PycharmProjects/Atari35/scn16/graphs/{}/{}/'.format(algo,game)+'*0.monitor.csv')
    tx, mean, cis, m = plot(infiles, smooth=1, split=False)
    plt.fill_between(tx, cis[0], cis[1], alpha=0.5, color=c)
    plt.plot(tx, mean, label="SCN-16", color=c)
    return m


def nlfcn64(pcn, algo, game, plt,c):
    infiles = glob.glob('/home/gaurav/PycharmProjects/Atari35/fcn/graphs/{}/{}/'.format(algo,game)+'6*{}*64*{}-0.monitor.csv'.format(pcn, rseeds))
    if len(infiles) > 0:
        tx, mean, cis, m = plot(infiles, smooth=1, split=False)
        plt.fill_between(tx, cis[0], cis[1], alpha=0.5, color=c)
        plt.plot(tx, mean, label="nl-FCN with 6 actors", color=c)
        return m

def lfcn(algo, game, plt,c):
    infiles = glob.glob('/home/gaurav/PycharmProjects/Atari35/lfcn/graphs/{}/{}/'.format(algo,game)+'*.monitor.csv')
    if len(infiles) > 0:
        tx, mean, cis, m = plot(infiles, smooth=1, split=False)
        plt.fill_between(tx, cis[0], cis[1], alpha=0.5, color=c)
        plt.plot(tx, mean, label="l-FCN with 3 actors", color=c)
        return m

def nlfcn16(algo, game, plt,c):
    infiles = glob.glob('/home/gaurav/GitRepo/results/fcn/graphs/{}/{}/'.format(algo,game)+'*{}*16*{}-0.monitor.csv'.format(pc, rseeds))
    if len(infiles) > 0:
        tx, mean, cis, m = plot(infiles, smooth=1, split=False)
        plt.fill_between(tx, cis[0], cis[1], alpha=0.5, color=c)
        plt.plot(tx, mean, label="nlfcn policy using 6 actors 16 hidden units", color=c)
        return m

def nlfcn32(algo, game, plt,c):
    infiles = glob.glob('/home/gaurav/GitRepo/results/fcn/graphs/{}/{}/'.format(algo,game)+'*{}*32*{}-0.monitor.csv'.format(pc, rseeds))
    if len(infiles) > 0:
        tx, mean, cis, m = plot(infiles, smooth=1, split=False)
        plt.fill_between(tx, cis[0], cis[1], alpha=0.5, color=c)
        plt.plot(tx, mean, label="nlfcn policy using 6 actors 32 hidden units", color=c)
        return m


def nlfcn8(algo, game, plt,c):
    infiles = glob.glob('/home/gaurav/GitRepo/results/fcn/graphs/{}/{}/'.format(algo,game)+'*{}*8*{}-0.monitor.csv'.format(pc, rseeds))
    if len(infiles) > 0:
        tx, mean, cis, m = plot(infiles, smooth=1, split=False)
        plt.fill_between(tx, cis[0], cis[1], alpha=0.5, color=c)
        plt.plot(tx, mean, label="nlfcn policy using 6 actors 8 hidden units", color=c)
        return m

def nlinear16(algo, game, plt,c):
    infiles = glob.glob('/home/gaurav/PycharmProjects/Atari35/nlinear/graphs/{}/{}/'.format(algo,game)+'*{}*16*{}-0.monitor.csv'.format(pc, rseeds))
    tx, mean, cis, m = plot(infiles, smooth=1, split=False)
    plt.fill_between(tx, cis[0], cis[1], alpha=0.5, color=c)
    plt.plot(tx, mean, label="MLP-16", color=c)
    return m

def nlinear64(algo, game, plt,c):
    infiles = glob.glob('/home/gaurav/PycharmProjects/Atari35/nlinear/graphs/{}/{}/'.format(algo,game)+'*{}*64*{}-0.monitor.csv'.format(pc, rseeds))
    tx, mean, cis, m = plot(infiles, smooth=1, split=False)
    plt.fill_between(tx, cis[0], cis[1], alpha=0.5, color=c )
    plt.plot(tx, mean, label="MLP-64", color=c)
    return m


def nlfcn_baseline(algo, game, plt,c):
    infiles = glob.glob('/home/gaurav/GitRepo/fpn/nlfpn/graphs/{}/{}/'.format(algo,game)+'baseline-[12]-0.monitor.csv')
    if len(infiles) > 0:
        tx, mean, cis, m = plot(infiles, smooth=1, split=False)
        plt.fill_between(tx, cis[0], cis[1], alpha=0.5)
        plt.plot(tx, mean, label="baseline nlfcn policy using 6 actors 64 hidden units", color=c)
        return m

if __name__ == "__main__":
    games = ["HalfCheetah-v2", "Hopper-v2", "Humanoid-v2", "InvertedDoublePendulum-v2","Swimmer-v2","Walker2d-v2"]
    for game in games:
        algo = "ppo"

        num_steps = 2000000
        f1 = plt.figure()
        tick_fractions = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ticks = tick_fractions * num_steps
        tick_names = ["{:.1e}".format(tick) for tick in ticks]

        pc = "*"
        rseeds = "*"
        print("Starting printing for {}".format(game))
        #nlfcn_baseline(algo, game, plt)
        #linear(algo,game,plt,color_defaults[0])
        m = lfcn(algo, game, plt, color_defaults[1])
        print("LFCN: {}".format(m))
        m = nlfcn64("msi",algo, game, plt,color_defaults[2])
        print("nl-fcn: {}".format(m))
        m = scn16(algo,game,plt,color_defaults[3])
        print("scn16: {}".format(m))
        m = scn(algo,game,plt,color_defaults[4])
        print("scn64: {}".format(m))
        #nlinear16(algo,game,plt,color_defaults[5])
        #nlinear64(algo, game, plt,color_defaults[6])

        plt.xticks(ticks, tick_names)
        plt.xlim(0, num_steps * 1.01)
        plt.xlabel('Number of Timesteps (M)')
        plt.ylabel('Rewards')
        plt.title(game)
        plt.legend(loc=4)
        #plt.savefig("images/{}.pdf".format(game))
        print("Done printing for {}".format(game))