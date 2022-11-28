from matplotlib import pyplot as plt

def save_frame_seq(state):
    plt.figure(figsize=(20, 16))
    for idx in range(state.shape[3]):
        plt.subplot(1,4, idx+1)
        plt.imshow(state[0][:,:,idx])
    plt.savefig(f'mario_frame_seq.png')

def save_frame(state, path:str):
    plt.figure()
    plt.imshow(state)
    plt.savefig(path) 
    plt.close()