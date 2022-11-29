
""" Utils functions """

import numpy as np
from matplotlib import pyplot as plt


def save_frame_seq(state: np.array, path: str) -> None:
    """ Save a frame from a stacked environement in png.

    Args:
        state : a frame of a stacked environment
        path : path of the file
    """
    plt.figure(figsize=(20, 16))
    for idx in range(state.shape[3]):
        plt.subplot(1,4, idx+1)
        plt.imshow(state[0][:,:,idx])
    plt.savefig(path)

def save_frame(state: np.array, path:str) -> None:
    """ Save a frame from an environement in png.

    Args:
        state : a frame of an environment
        path : path of the file
    """
    plt.figure()
    plt.imshow(state)
    plt.savefig(path)
    plt.close()
