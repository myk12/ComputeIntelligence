# Kmeans cluster

import numpy as np
import random
import utils

# predefine  variables
filename_data = "./iris.dat"
filename_log  = "./kmeans.log"
cluster_num = 3
max_iter = 100

# --------------------- load data --------------------
dataset = np.loadtxt(filename_data)
samples = dataset[:, 0:4].astype(np.float64)
labels  = dataset[:, 4].astype(np.int32)
samples_num = samples.shape[0]
with open(filename_log, "w") as logfd:
    logfd.write("------- KMEANS log -------\n")

# -------------------- start cluster -----------------

# random chose centers 初始化聚类中心
choice = np.random.choice(samples_num, cluster_num)
centers = samples[choice]

utils.logger_init_center(centers, filename_log)

c_label = np.zeros(labels.shape, dtype=np.int32)
clusters = [set() for _ in range(cluster_num)]

it = 0
while it < max_iter:
    print("iteartion : %d" %it)
    clusters_ = [set() for _ in range(cluster_num)]
    
    # start cluster 开始聚类
    for i in range(samples_num):
        sample = samples[i]
        parallel = np.tile(sample, (cluster_num, 1))

        # It's no need to calculate the sqrt
        distances = np.sum(np.square(parallel - centers), axis=1)
        minidx = distances.argmin()

        # Put into box and update label
        clusters_[minidx].add(i)
        c_label[i] = minidx

    # recalculate centers 更新类中心点
    static = True
    for i in range(cluster_num):
        c, c_ = clusters[i], clusters_[i]
        if c == c_:
            continue
        else:
            static = False
            # renew center
            samples_c = samples[list(c_)]
            centers[i] = np.average(samples_c, axis=0)
    
    clusters = clusters_
    if static:
        break
    it += 1

# -------------------- result -----------------
utils.logger_iter_times(it, filename_log)
utils.logger_cluster_result(clusters, centers, filename_log)
utils.calculate_error_rate(labels, clusters, filename_log)





