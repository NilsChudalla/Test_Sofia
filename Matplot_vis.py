import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
block = np.load("").reshape(50, 50, 50)
grid = np.load("solutions_grid.npy", allow_pickle=True).reshape(1,)[0].values
grid_df = pd.DataFrame(grid, columns=['x', 'y', 'z'])

xmin, xmax, ymin, ymax, zmin, zmax = [6,25,3,35,-5.5,0.5]
resolution = np.array([50.0, 50.0, 50.0])

d_grid = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
cell_width = d_grid / resolution
grid_min = np.array([xmin, ymin, zmin]) + cell_width/0.5
grid_max = np.array([xmax, ymax, zmax]) - cell_width/0.5

x_array = np.linspace(grid_min[0], grid_max[0], resolution[0].astype(int))
y_array = np.linspace(grid_min[1], grid_max[1], resolution[1].astype(int))
z_array = np.linspace(grid_min[2], grid_max[2], resolution[2].astype(int))
XX, YY = np.meshgrid(x_array, y_array)
#%%
depth = -1000
depth_index = (np.round(dz/depth)).astype(int)

hslice_coords = block[-1,:,:]


plt.contourf(XX, YY, hslice_coords, cmap='magma')
plt.axis('equal')
plt.show()
#%%