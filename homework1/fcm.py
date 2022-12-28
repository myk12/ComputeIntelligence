import errno
import numpy as np
import matplotlib.pyplot as plt

def FCM_dist(X, centers):
    N, D = np.shape(X)
    C, D = np.shape(centers)

    tile_x = np.tile(np.expand_dims(X, 1), [1, C, 1])
    tile_centers = np.tile(np.expand_dims(centers, axis=0), [N, 1, 1])

    dist = np.sum((tile_x - tile_centers) ** 2, axis=-1)

    return np.sqrt(dist)

def FCM_get_centers(U, X, m):
    N, D = np.shape(X)
    N, C = np.shape(U)

    um = U ** m

    tile_X  = np.tile(np.expand_dims(X, 1), [1, C, 1])
    tile_um = np.tile(np.expand_dims(um, -1), [1, 1, D])
    temp = tile_X * tile_um

    new_C = np.sum(temp, axis=0) / np.expand_dims(np.sum(um, axis=0), axis=-1)

    return new_C

def FCM_get_U(X, centers, m):
    N, D = np.shape(X)
    C, D = np.shape(centers)

    temp = FCM_dist(X, centers) ** float(2 / (m-1))

    tile_temp = np.tile(np.expand_dims(temp, 1), [1, C, 1])
    
    denominator_ = np.expand_dims(temp, -1) / tile_temp

    return 1 / np.sum(denominator_, axis=-1)

def save_nparray(opend_fd, nparray):
    for i in range(len(nparray)):
        opend_fd.write(str(nparray[i]) + '\n')

def FCM_train(X, n_centers, m, max_iter=100, theta=1e-5, seed=0, res_log=None):

    rng = np.random.RandomState(seed)
    N, D = np.shape(X)

    # randomly init relationship matrix U
    U = rng.uniform(size=(N, n_centers))

    U = U / np.sum(U, axis=1, keepdims=True)

    # start iteration
    for i in range(max_iter):
        print("+++++++ iteration %d +++++++" %i)
        U_old = U.copy()
        centers = FCM_get_centers(U, X, m)
        if i == 0:
            # record init class center
            res_log.write("1.初始类中心点：\n")
            save_nparray(res_log, centers)
        U = FCM_get_U(X, centers, m)

        # the difference between two adjacent training process
        # is slight, end training
        if np.linalg.norm(U - U_old) < theta:
            res_log.write("2.迭代次数\n%d\n" %(i+1))
            break
    
    return centers, U

def FCM_get_class(U):
    return np.argmax(U, axis=-1)

def FCM_partition_coefficient(U):
    return np.mean(U ** 2)

def FCM_partition_entropy_coefficient(U):
    return -np.mean(U * np.log2(U))


def main():
    N = 3000

    datafile = "./iris.dat"
    resfile  = "./res.txt"
    raw_data = np.loadtxt(datafile, dtype='float32')
    
    #print(raw_data)
    #print(raw_data.shape)

    X = raw_data[:,:-1]
    y = raw_data[:,-1]

    #print(X)
    #print(y)
    #print(X.shape)
    #print(y.shape)

    # devide into three class
    n_centers = 3
    m = 2
    res_fd = open(resfile, "w+", encoding='utf-8')
    centers, U = FCM_train(X, \
                        n_centers, \
                        m, \
                        max_iter = 100, \
                        theta=1e-5, \
                        seed=0, \
                        res_log=res_fd)

    labels = FCM_get_class(U)
    res_fd.write("3.聚类结果：\n")
    for i in range(3):
        cluster_i = np.argwhere(labels == i)
        res_fd.write("3.%d 类%d:\n" %(i+1, i+1))
        res_fd.write("样本中心点：\n")
        res_fd.write(str(centers[i]) + '\n')
        #save_nparray(res_fd, centers[i].reshape(centers[i].shape[1], centers[i].shape[0]))
        #np.savetxt(res_fd, centers[i], delimiter=',')
        res_fd.write("样本点：\n")
        save_nparray(res_fd, X[cluster_i])
        #np.savetxt(res_fd, cluster_i, delimiter=',')
        #res_fd.write(cluster_i)

    idx_1 = (labels == 2)
    idx_2 = (labels == 1)
    labels[idx_1] = 1
    labels[idx_2] = 2

    error_num = sum((labels - y) != 0)
    res_fd.write("4.错误率：\n%.4f %%" %(error_num / len(X) * 100))
    res_fd.close()

if __name__ == "__main__":
    main()