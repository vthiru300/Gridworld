import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os, shutil

from gridworld import *


def image_maker(index, robots, interestpoints, dim):
    for robot in robots:
        text = 'Robot Position: {}'.format(robot.position)
        #print(text)

    image_folder = 'images'

    filename = 'snapshot'+f'{index:13}'+'.png'
    figure, axes = plt.subplots()

    axes.set_xticks(np.arange(0, 1, 1/(dim*2)))
    axes.set_yticks(np.arange(0, 1, 1/(dim*2)))
    axes.xaxis.set_ticklabels([])
    axes.yaxis.set_ticklabels([])
    plt.grid(linestyle = '--', linewidth = 0.3)

    delta = np.array([1/(dim*2), 1/(dim*2)])

    circle_patches = [plt.Circle(robot.position / dim + delta, 0.01) for robot in robots]
    circle_coll = matplotlib.collections.PatchCollection(circle_patches, facecolors='blue')
    axes.add_collection(circle_coll)

    triangle_coordinate1 = np.array([0.5, 0.5])
    sidelength = 0.025
    triangle_coordinate2 = np.array([0.5 + sidelength, 0.5])
    segmentmidpoint = (triangle_coordinate1 + triangle_coordinate2) / 2
    opoint = (triangle_coordinate1 - segmentmidpoint) * (3 ** 0.5)
    t = np.array([[0, -1], [1, 0]])
    triangle_coordinate3 = segmentmidpoint + opoint @ t

    triangle_patches =[]
    for interestpoint in interestpoints:
        displacement1 = np.array([-sidelength / 2, -1*(triangle_coordinate3[1]-0.5) / 2]) - np.array([0.5, 0.5]);
        displacement2 = interestpoint.position / dim
        displacement = displacement1 + displacement2 + delta
        triangle_coordinates = np.array([triangle_coordinate1+displacement, triangle_coordinate2+displacement, triangle_coordinate3 + displacement]);
        triangle_patches.append(plt.Polygon(triangle_coordinates))

    triangle_coll = matplotlib.collections.PatchCollection(triangle_patches, facecolors='red');
    axes.add_collection(triangle_coll)

    axes.set_aspect(1)
    plt.savefig(os.path.join(image_folder, filename))
    plt.close()

def movie_maker():
    image_folder = 'images'
    video_name = 'simulation.avi'
    images = [img for img in sorted(os.listdir(image_folder))]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 5, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
