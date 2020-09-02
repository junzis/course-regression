import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.pyplot import cm
from matplotlib.patches import Rectangle

train_acs = "A320,A343,A359,A388,B737,B744,B748,B752,B763,B773,B789,C550,E145,E190"
train_acs = train_acs.split(",")


def plot_data(
    ax,
    df_train,
    df_test,
    xcol,
    ycol,
    show_train=True,
    show_test=False,
    label=True,
    show_type=False,
):
    x_train = df_train[xcol].values
    y_train = df_train[ycol].values

    x_test = df_test[xcol].values
    y_test = df_test[ycol].values

    type_train = df_train.index.values
    type_test = df_test.index.values

    if show_train:
        ax.scatter(
            x_train,
            y_train,
            color="k",
            s=50,
            lw=2,
            label="training set",
            facecolor="w",
            zorder=10,
        )

    if show_test:
        ax.scatter(
            x_test,
            y_test,
            color="r",
            s=50,
            lw=2,
            label="testing set",
            facecolor="w",
            zorder=10,
        )

    if label:
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)

    if show_type:
        for x, y, t in zip(x_train, y_train, type_train):
            # move the text label around
            x += max(x_train) / 50
            ax.text(x, y, t, ha="left", va="center", fontsize=8)

    if show_type and show_test:
        for x, y, t in zip(x_test, y_test, type_test):
            x -= max(x_train) / 25
            ax.text(x, y, t, ha="left", va="center", fontsize=8, color="r")

    ax.legend()


def plot_linear_model(ax, b0, b1, x, y, color="k", err=False, rss=False):

    x_ = np.linspace(min(x), max(x), 10)

    y_ = b0 + b1 * x_

    ax.plot(x_, y_, label="$\\beta_0$:{} \t $\\beta_1$:{}".format(b0, b1), color=color)

    if err:
        for x, y in zip(x, y):
            ax.plot([x, x], [y, b0 + b1 * x], color=color, ls=":")

    if rss:
        for x, y in zip(x, y):
            eps = y - (b0 + b1 * x)
            rect = Rectangle((x, y), -eps, -eps, alpha=0.2, color=color)
            ax.add_patch(rect)
        ax.set_aspect(1)

        ax.set_xlim([-400, 800])
        ax.set_ylim([-500, 2000])
    ax.legend(loc="upper left")


def print_model(coef):
    print("-" * 70)
    print(np.poly1d(coef[::-1]))
    print("-" * 70)


def plot_poly(ax, x, coef, color):
    x_ = np.linspace(min(x) - 2, max(x) + 2, 200)
    y_ = np.zeros(len(x_))
    for i, c in enumerate(coef):
        y_ += c * x_ ** i
    ax.plot(x_, y_, color=color, lw=2)