import numpy as np 
from imageio import imread
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Load terrain
terrain1 = imread('terrain.tif')

print(terrain1.shape)

"""
plt.figure()
plt.title('Terrain over Schio')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
"""