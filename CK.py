#%%
import random
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Hàm thực hiện giải thuật Simulated Annealing
def Simulated_Annealing(current, temps):
    current = current[:]  # Tạo một bản sao của đường đi hiện tại
    for i in range(10000):  # Lặp qua 10000 lần để cố gắng tối ưu hóa đường đi
        nextnode = neighbor_function(current)  # Tìm một đường đi lân cận
        deltaE = value(current) - value(nextnode)  # Tính sự thay đổi của giá trị
        if deltaE > 0:
            current = nextnode  # Chấp nhận đường đi mới nếu giá trị tăng
        else:
            acceptance_probability = math.exp(deltaE / temps)  # Xác suất chấp nhận đường đi mới
            if random.random() < acceptance_probability:
                current = nextnode
        temps = cooling_schedule(temps, i)  # Cập nhật nhiệt độ
    print("TSP solution found:", current, "\nPath cost:", value(current))
    return current

# Hàm cập nhật nhiệt độ theo hàm giảm dần
def cooling_schedule(temperature, iteration, k=20, lam=0.005, limit=1000):
    return k * np.exp(-lam * iteration) if iteration < limit else temperature

# Hàm tạo đường đi lân cận bằng cách hoán đổi hai điểm ngẫu nhiên
def neighbor_function(current):
    neighbor = current[:]
    i = random.randint(0, len(current) - 1)
    j = random.randint(0, len(current) - 1)
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor

# Hàm tính giá trị của đường đi dựa trên ma trận trọng số
def value(x):
    y = a[x[0]][x[1]] + a[x[1]][x[2]] + a[x[2]][x[3]] + a[x[3]][x[4]]
    return y

# Ma trận trọng số
a = [
    [0, 2, 4, 6, 8],
    [2, 0, 5, 7, 9],
    [4, 5, 0, 8, 3],
    [6, 7, 8, 0, 1],
    [8, 9, 3, 1, 0]
]

# Tạo đồ thị từ ma trận trọng số
G = nx.Graph()
num_nodes = len(a)

for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        if a[i][j] != 0:
            G.add_edge(i, j, weight=a[i][j])

# Vẽ đồ thị vô hướng có trọng số
pos = nx.spring_layout(G)
edge_labels = {(i, j): w['weight'] for i, j, w in G.edges(data=True)}

plt.subplot(1, 2, 1)
nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color="skyblue", font_size=10)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
plt.title('Original Graph')

# Áp dụng Simulated Annealing để tìm đường đi tối ưu
temps = 100
initial_solution = list(range(num_nodes))
random.shuffle(initial_solution)
current = Simulated_Annealing(initial_solution, temps)

# Tạo đồ thị kết quả
G_result = nx.DiGraph()
for i in range(len(current) - 1):
    G_result.add_edge(current[i], current[i + 1])
G_result.add_edge(current[-1], current[0])

# Vẽ đồ thị kết quả
plt.subplot(1, 2, 2)
nx.draw(G_result, with_labels=True, font_weight='bold', arrows=True)
plt.title('Result after Simulated Annealing')

plt.show()

# %%

