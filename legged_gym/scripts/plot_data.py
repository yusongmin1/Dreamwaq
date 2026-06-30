import numpy as np
import matplotlib.pyplot as plt

def plot_data(data1, data2=None):
        nb_rows = 4
        nb_cols = 4
        fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(15, 10))
        axs = axs.flatten()
           
        data1 = np.array(data1)
        data2 = np.array(data2)
        print(data1.shape)
        for i in range(16):
            axs[i].plot(data1[:, i], label='gazebo')
            if data2 is not None:
                axs[i].plot(data2[:, i], label='gym')
        plt.legend()
        plt.show()

data_gazebo = np.loadtxt('actions.txt')
data_gym = np.loadtxt('actions_gym.txt')

print(data_gazebo.shape, data_gym.shape)

plot_data(data_gazebo, data_gym)