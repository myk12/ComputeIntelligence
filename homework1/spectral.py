# spectral cluster

############## Algorithm #############
#
# 1. 构造相似性矩阵 W 和度矩阵 D
# 2. 计算拉普拉斯矩阵 L
# 3. 构建标准化之后的拉普拉斯矩阵 D^(-1/2)LD^(-1/2)
# 4. 计算 D^(-1/2)LD^(-1/2)最小的k1个特征值所各自对应的特征向量f
# 5. 将各自对应的特征向量f组成的矩阵按行标准化，最终组成 n*k1 维的特征矩阵F
# 6. 对F中的每一行作为一个k1维的样本，共n个样本，用K-means方法聚类
# 7. 得到簇划分 C( c1, c2, ... ck2)

########### IMPLEMENTATION ###########
from sklearn.cluster import KMeans
import collections
import numpy as np
float_formatter = lambda x: "%.7f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})

# predefine variables
filename_data = "./iris.dat"
filename_log  = "./spectral.log"
cluster_num   = 3
max_iter      = 100

# prepare data
dataset = np.loadtxt(filename_data)
samples = dataset[:, 0:4].astype(np.float64)
labels  = dataset[:, 4].astype(np.int32)

# -------------- start cluster --------------
# 计算样本间距离
def calculate_w_ij(a, b, sigma=1):
    w_ab = np.exp(-np.sum((a-b)**2)/(2*sigma**2))
    return 

# 计算邻接矩阵
def construct_matrix_W(data, k=5, sigma=1):
    rows = len(data)
    W = np.zeros((rows, rows))
    for i in range(rows):
        for j in range(rows):
            if (i!=j):
                W[i][j] = np.exp(-np.sum((data[i] - data[j])**2/(2*sigma**2)))

        t = np.argsort(W[i, :])
        for x in range(rows - k):
            W[i][t[x]] = 0
    W = (W + W.T) / 2

    return W

# 计算标准化拉普拉斯矩阵
def calculate_matrix_laplace(W):
    degree_matrix = np.sum(W, axis=1)

    L = np.diag(degree_matrix) - W

    # 拉普拉斯矩阵标准化，就是选择Ncut切图
    sqrt_degree_matrix = np.diag(1.0 / (degree_matrix**0.5)) # D^(-1/2)
    return np.dot(np.dot(sqrt_degree_matrix, L), sqrt_degree_matrix) # D^(-1/2)*L*D^(-1/2)

# 按行归一化
def normalization(mat):
    sum = np.sqrt(np.sum(mat**2, axis=1, keepdims=True))
    return mat / sum

# 计算分类错误率
def calculate_error_rate(label):
    true_label = 0

    c1 = collections.Counter(label[0:50])
    true_label += c1.most_common(1)[0][1]

    c2 = collections.Counter(label[50:100])
    true_label += c2.most_common(1)[0][1]

    c3 = collections.Counter(label[100:150])
    true_label += c3.most_common(1)[0][1]
    
    #print(true_label)
    return 1 - true_label / len(label)

if __name__ == '__main__':
    # 计算邻接矩阵
    W = construct_matrix_W(samples)
    #print(W)
    # 计算邻接矩阵W的标准化拉普拉斯矩阵
    L_sym = calculate_matrix_laplace(W)

    # 特征值分解
    lam, H = np.linalg.eig(L_sym)

    # 特征值排序
    t = np.argsort(lam)

    H = np.c_[H[:, t[0]], H[:, t[1]], H[:, t[2]]]
    
    # 归一化处理
    H = normalization(H)

    # 使用KMeans对H矩阵聚类
    model = KMeans(n_clusters=3)
    model.fit(H)

    labels = model.labels_

    res = np.c_[samples, labels]

    print(len(labels))
    print(type(labels))
    print(labels.shape)

    error_rate = calculate_error_rate(labels)
    print("error_rate: %f" %error_rate)