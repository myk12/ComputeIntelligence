import numpy as np

line_split  = "\n+-+-+-+-+-+- %s +-+-+-+-+-+\n"
line_end    = "\n---------------------------\n"

def logger_init_center(centers, filename):
    with open(filename, 'a+') as logfd:
        logfd.write(line_split %"init_center")
        
        for center in centers:
            logfd.write(str(center) + '\n')

        logfd.write(line_end)

    return True

def logger_iter_times(times, filename):
    if not times:
        return False
    with open(filename, 'a+') as logfd:
        logfd.write("iteration times: %d" %times)
    
    return True

def logger_cluster_result(clusters, centers, filename):
    with open(filename, 'a+') as logfd:
        logfd.write(line_split %"cluster result")
        for cluster, center in zip(clusters, centers):
            logfd.write(line_split %"cluster")
            logfd.write("center  : %s \n" %str(center))
            logfd.write("element : %s \n" %str(cluster))
    return True

def calculate_error_rate(labels, clusters, filename):
    total_num   = labels.shape[0]
    true_num    = 0
    for cluster in clusters:
        label_dict = {}
        for it in cluster:
            try:
                label_dict[labels[it]] += 1
            except KeyError:
                label_dict[labels[it]] = 1
        true_num += max(label_dict.values())
    
    with open(filename, 'a+') as logfd:
        logfd.write("error_rate : %f" %(1 - true_num/total_num))

    return True