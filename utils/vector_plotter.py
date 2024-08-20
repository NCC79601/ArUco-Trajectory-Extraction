import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

class VectorPlotter:
    def __init__(self, vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], origin = [0, 0, 0], show=True, save=False, save_dir='./', elev=30, azim=-60):
        self.vector_list = []
        for i in range(len(vectors)):
            self.vector_list.append(vectors[i])
        # self.vector = np.array(vector)
        self.origin = np.array(origin)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(elev=elev, azim=azim)
        # self.draw_vectors()

        self.frame_cnt = 0

        self.show = show
        self.save = save
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if self.show:
            self.draw_vectors()
            plt.ion()
            plt.show()

    def draw_vectors(self, axis_lim=[-3, 3], scale=1.0):
        plt.clf()  # clear the current figure
        fig = plt.gcf()  # get current figure
        # check if there is an existing axis, and get current view
        if hasattr(self, 'ax'):
            elev, azim = self.ax.elev, self.ax.azim
        else:
            elev, azim = None, None
        self.ax = fig.add_subplot(111, projection='3d')  # create a new axis
        # if there is a view, set it
        if elev is not None and azim is not None:
            self.ax.view_init(elev=elev, azim=azim)
        color_list = ['r', 'g', 'b', 'c', 'm', 'y']
        for i in range(len(self.vector_list)):
            vector = np.array(self.vector_list[i]) * scale
            self.ax.quiver(
                self.origin[0],
                self.origin[1],
                self.origin[2],
                vector[0],
                vector[1],
                vector[2],
                color=color_list[i % 6]
            )  # draw vectors
        self.ax.set_xlim(axis_lim)
        self.ax.set_ylim(axis_lim)
        self.ax.set_zlim(axis_lim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        # plt.ioff()

        if self.save:
            plt.savefig(os.path.join(
                self.save_dir,
                f'{self.frame_cnt:06d}.png'
            ))
            self.frame_cnt += 1

    def update_vectors(self, new_vectors, new_origin = None, axis_lim=[-3, 3], scale=1.0):
        if new_origin is not None:
            self.origin = np.array(new_origin)
        self.vector_list = new_vectors
        self.draw_vectors(axis_lim, scale)


if __name__ == '__main__':
    # Example usage
    vector = [0.1, 0.1, 0.1]
    dir = [1, 1, 1]

    plotter = VectorPlotter([vector])
    
    while True:
        vector = np.array(vector) + np.array(dir) * 0.01
        plotter.update_vectors([vector])
        plt.pause(0.2)