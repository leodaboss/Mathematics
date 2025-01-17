import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def generate_maze_graph(array):
    n, m = array.shape
    G = nx.Graph()

    for i in range(n):
        for j in range(m):
            if array[i, j]:
                if i > 0 and array[i-1, j]:  # Connect to the cell above
                    G.add_edge((i, j), (i-1, j))
                if j > 0 and array[i, j-1]:  # Connect to the cell to the left
                    G.add_edge((i, j), (i, j-1))
                if i < n-1 and array[i+1, j]:  # Connect to the cell below
                    G.add_edge((i, j), (i+1, j))
                if j < m-1 and array[i, j+1]:  # Connect to the cell to the right
                    G.add_edge((i, j), (i, j+1))

    pos = {(i, j): (j, -i) for i in range(n) for j in range(m)}
    nx.draw(G, pos, with_labels=True, node_size=50, node_color='skyblue', font_size=8, font_color='black')
    plt.show()
p=0.9
n=10
def __main__():
    grid=np.random.rand(n,n)<p
    generate_maze_graph(grid)
    print(grid)
__main__()