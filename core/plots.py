import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_confidence(labels, X):
        un_labls = np.unique(labels)
        mean = np.array([ np.mean(X[labels==u]) for u in un_labls])
        std = np.array([ np.std(X[labels==u]) for u in un_labls])
        p = plt.plot(un_labls, mean)
        plt.fill_between(un_labls, mean+std, mean-std, color=p[0].get_color(),alpha=0.1)

def plot_minmax(labels, X):
        un_labls = np.unique(labels)
        mean = np.array([ np.mean(X[labels==u]) for u in un_labls])
        mn = np.array([ np.min(X[labels==u]) for u in un_labls])
        mx = np.array([ np.max(X[labels==u]) for u in un_labls])
        p = plt.plot(un_labls, mean)
        plt.fill_between(un_labls, mx, mn, color=p[0].get_color(),alpha=0.1)

def plot_3d(R,size, size2=None):
    A = np.arange(size)
    if size2 is None:
        B = np.arange(size)
    else:
        B = np.arange(size2)
    A, B = np.meshgrid(A, B)
    ax = plt.axes(projection='3d')
    ax.plot_surface(A, B, R, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

def plot_3d_vect(X,size,figure_n=None):
    #plt.ion()
    R = X
    if figure_n is not None: plt.figure(figure_n)
    plt.clf()
    plot_3d(R.reshape(size,size),size)
    #plt.draw()
    #plt.show(block=False)
    plt.pause(0.001)


def plot_3d_vect_2(X,Y,grid_size,figure_n):
    A = np.arange(grid_size)
    A, B = np.meshgrid(A, A)

    fig =plt.figure(figure_n,figsize=plt.figaspect(0.5))
    fig.clf()
    ax = fig.add_subplot(1,2,1,projection='3d')
    ax.plot_surface(A, B, X.reshape(grid_size,grid_size), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax = fig.add_subplot(1,2,2,projection='3d')
    ax.plot_surface(A, B, Y.reshape(grid_size,grid_size), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    #plt.draw()
    #plt.show(block=False)
    plt.pause(0.001)

def plot(X,Y,figure_n=None):
    #plt.ion()
    if figure_n is not None: plt.figure(figure_n)
    plt.clf()
    plt.plot(X,Y)
    #plt.draw()
    #plt.show(block=False)
    plt.pause(0.001)

def matshow(X,figure_n=None):
    #plt.ion()
    if figure_n is not None: plt.figure(figure_n)
    plt.clf()
    if figure_n is not None:
        plt.matshow(X,fignum=figure_n)
    else:
        plt.matshow(X)
    plt.pause(0.001)
    #plt.draw()
    #plt.show(block=False)