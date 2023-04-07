import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# create the figure
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(np.random.random((50,50)))
plt.show(block=False)

# draw some data in loop
for i in range(10):
    # wait for a second
    time.sleep(1)
    # replace the image contents
    im.set_array(np.random.random((50,50)))
    # redraw the figure
    fig.canvas.draw()

#%%
# Based on code from https://stackoverflow.com/a/43971236/190597 (umutto) and
# https://stackoverflow.com/q/23042482/190597 (Raphael Kleindienst)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll

fig, ax = plt.subplots()
cmap = mcolors.ListedColormap(['white', 'black'])
bounds = [-0.5, 0.5, 1.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)
data = np.random.rand(10, 10) * 2 - 0.5
im = ax.imshow(data, cmap=cmap, norm=norm)

grid = np.arange(-0.5, 11, 1)
xmin, xmax, ymin, ymax = -0.5, 10.5, -0.5, 10.5
lines = ([[(x, y) for y in (ymin, ymax)] for x in grid]
         + [[(x, y) for x in (xmin, xmax)] for y in grid])
grid = mcoll.LineCollection(lines, linestyles='solid', linewidths=2,
                            color='teal')
ax.add_collection(grid)

def animate(i):
    data = np.random.rand(10, 10) * 2 - 0.5
    im.set_data(data)
    # return a list of the artists that need to be redrawn
    return [im, grid]

anim = animation.FuncAnimation(
    fig, animate, frames=200, interval=0, blit=True, repeat=True)
plt.show()

#%%
import numpy
%matplotlib notebook
import matplotlib.pyplot as plt

x = [1, 2, 3]
fig, ax = plt.subplots()

plt.ion()
plt.show()
for loop in range(0,3): 
    y = numpy.dot(x, loop)
    line,=ax.plot(x,y)  # plot the figure
    plt.gcf().canvas.draw()
    line.remove()
    del line
    _ = input("Press [enter] to continue.") # wait for input from the 
#%%
import numpy as np
import matplotlib.pyplot as plt

plt.axis([0, 10, 0, 1])

for i in range(10):
    y = np.random.random()
    plt.scatter(i, y)
    plt.pause(0.05)

plt.show()
#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def getsumofneighbors(matrix, i, j):
    region = matrix[max(0, i-1) : i+2,
                    max(0, j-1) : j+2]
    return np.sum(region) - matrix[i, j] # Sum the region and subtract center

data = np.array([[1,0,0],[1,1,0],[0,1,0]])
#plt.scatter(data)

plt.matshow(data)
#%%
def update_matrix(matrix):
    
#%% [1,1,1,2,2,2,1,1,1,1,2,2,2,2,1]
x = [1,0,0,1,1,0,1,0,0,1,0,1,1,0,1]
x1 = [0,0] + x + [0]
y = np.cumsum(x1)
print(y[3:] - y[:-3])
#%%
vect_v = np.vectorize(update_vector)
data_test = vect_v(data)

#%% works for any rectangular matrix
data = np.array([[1,0,0,1,0],[1,0,1,0,1],[0,1,1,0,1],[0,1,1,1,0]])
data1 = []
for i in data:
    data1.append(update_vector(i))
data1 = np.array(data1)
data2 = np.transpose(data1)
data3 = []
for i in data2:
    data3.append(update_vector(i))
data3 = np.transpose(data3)
data4 = data3 - data
result = vec_gol(data4,data)
#%%

def update_vector(v):
    #v1 = [0,0] + v + [0]
    v1 = np.concatenate([[0,0],v,[0]])
    y = np.cumsum(v1)
    y1 = y[3:] - y[:-3]
    return y1

def game_of_life(x_sum,x):
    if x == 1 and x_sum == 2:
        return 1
    elif x_sum == 3:
        return 1
    else:
        return 0
vec_gol = np.vectorize(game_of_life) 
def full_process(x):
    data1 = []
    for i in data:
        data1.append(update_vector(i))
    data1 = np.array(data1)
    data2 = np.transpose(data1)
    data3 = []
    for i in data2:
        data3.append(update_vector(i))
    data3 = np.transpose(data3)
    data4 = data3 - data
    result = vec_gol(data4,data)
    return result

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#data = np.array([[1,0,0,1,0],[1,0,1,0,1],[0,1,1,0,1],[0,1,1,1,0]])
#data_result = full_process(data)

data = np.random.randint(2,size = (10,10))
plt.matshow(data)

for i in range(50):
    data = full_process(data)
    plt.matshow(data)
    #%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True)
plt.show()

#%% FULL CODE WITH DYNAMIC SIZE, RULES, INITIAL CONDITION
# Based on code from https://stackoverflow.com/a/43971236/190597 (umutto) and
# https://stackoverflow.com/q/23042482/190597 (Raphael Kleindienst)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll

def update_vector(v):
    #v1 = [0,0] + v + [0]
    v1 = np.concatenate([[0,0],v,[0]])
    y = np.cumsum(v1)
    y1 = y[3:] - y[:-3]
    return y1

def game_of_life(x_sum,x):
    if x == 1 and (x_sum == 2 or x_sum == 3):
        return 1
    elif x == 0 and (x_sum == 3 or x_sum == 6):
        return 1
    else:
        return 0

vec_gol = np.vectorize(game_of_life) 

def full_process(x):
    data1 = []
    for i in x:
        data1.append(update_vector(i))
    data1 = np.array(data1)
    data2 = np.transpose(data1)
    data3 = []
    for i in data2:
        data3.append(update_vector(i))
    data3 = np.transpose(data3)
    data4 = data3 - x
    result = vec_gol(data4,x)
    return result

fig, ax = plt.subplots()
cmap = mcolors.ListedColormap(['white', 'black'])
bounds = [-0.5, 0.5, 1.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

size_of_field = 105

data = np.random.randint(2, size = (size_of_field,size_of_field))
#data = np.zeros(shape=(size_of_field,size_of_field))
#data[50,50] = 1
#data[49,50] = 1
#data[50,49] = 1
#data[50,51] = 1
#data[51,51] = 1
#data = np.random.randint(2,size = (10,10))
im = ax.imshow(data, cmap=cmap)

#grid = np.arange(-0.5, 11, 1)
#xmin, xmax, ymin, ymax = -0.5, 10.5, -0.5, 10.5
#lines = ([[(x, y) for y in (ymin, ymax)] for x in grid]
#         + [[(x, y) for x in (xmin, xmax)] for y in grid])
#grid = mcoll.LineCollection(lines, linestyles='solid', linewidths=2,
#                            color='teal')
#ax.add_collection(grid)
#temp = np.copy(data)

def animate(i):
    #data = np.random.randint(2, size = (10,10))
    data_temp = im._A.data
    new = full_process(data_temp)
    im.set_data(new)
    # return a list of the artists that need to be redrawn
    return [im, grid]

anim = animation.FuncAnimation(
    fig, animate, frames=200, interval=0, blit=True, repeat=True)
plt.show()
#%%
#%% AS ABVOE WITH GRID
# Based on code from https://stackoverflow.com/a/43971236/190597 (umutto) and
# https://stackoverflow.com/q/23042482/190597 (Raphael Kleindienst)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll

def update_vector(v):
    #v1 = [0,0] + v + [0]
    v1 = np.concatenate([[0,0],v,[0]])
    y = np.cumsum(v1)
    y1 = y[3:] - y[:-3]
    return y1

def game_of_life(x_sum,x):
    if x == 1 and x_sum == 2:
        return 1
    elif x_sum == 3:
        return 1
    else:
        return 0

vec_gol = np.vectorize(game_of_life) 

def full_process(x):
    data1 = []
    for i in x:
        data1.append(update_vector(i))
    data1 = np.array(data1)
    data2 = np.transpose(data1)
    data3 = []
    for i in data2:
        data3.append(update_vector(i))
    data3 = np.transpose(data3)
    data4 = data3 - x
    result = vec_gol(data4,x)
    return result

fig, ax = plt.subplots()
cmap = mcolors.ListedColormap(['white', 'black'])
bounds = [-0.5, 0.5, 1.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

size_of_field = 50

data = np.random.randint(2, size = (size_of_field,size_of_field))
#data = np.random.randint(2,size = (10,10))
im = ax.imshow(data, cmap=cmap)

grid = np.arange(-0.5, size_of_field + 1, 1)
xmin, xmax, ymin, ymax = -0.5, size_of_field + 0.5, -0.5, size_of_field + 0.5
lines = ([[(x, y) for y in (ymin, ymax)] for x in grid]
         + [[(x, y) for x in (xmin, xmax)] for y in grid])
grid = mcoll.LineCollection(lines, linestyles='solid', linewidths=2,
                            color='teal')
ax.add_collection(grid)


def animate(i):
    #data = np.random.randint(2, size = (10,10))
    data_temp = im._A.data
    new = full_process(data_temp)
    im.set_data(new)
    # return a list of the artists that need to be redrawn
    return [im, grid]

anim = animation.FuncAnimation(
    fig, animate, frames=200, interval=0, blit=True, repeat=True)
plt.show()