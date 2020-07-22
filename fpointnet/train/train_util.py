''' Util functions for training and evaluation.

Author: Charles R. Qi
Date: September 2017
'''

import numpy as np

def get_batch_by_imgindex(dataset, img_index, object_index, num_point, num_channel = 4 ,bsize = 32,  mode = "num") :
    '''

    :param dataset: dataset对象
    :param img_index:  根据 mode 不同 , mode 为 num 就是序号，为 name是文件名   int
    :param object_index:  这张图片的第几个 index  numpy tensor
    :param mode:  name or num
    :return:
    '''

    assert mode in ["name","num"]
    if mode == "num" :
        base_index = dataset.cum_num_index[img_index]
    else :
        base_index = dataset.img_index_map[img_index]


    batch_data = np.zeros((bsize, num_point, num_channel))
    batch_label = np.zeros((bsize, num_point), dtype=np.int32)
    batch_center = np.zeros((bsize, 3))
    batch_heading_class = np.zeros((bsize,), dtype=np.int32)
    batch_heading_residual = np.zeros((bsize,))
    batch_size_class = np.zeros((bsize,), dtype=np.int32)
    batch_size_residual = np.zeros((bsize, 3))
    batch_rot_angle = np.zeros((bsize,))

    obj = object_index.cpu().data.numpy()
    obj = obj.ravel()        # 将数组平铺 返回视图
    assert (obj.shape[0] == bsize)
    if dataset.one_hot:
        batch_one_hot_vec = np.zeros((bsize, 3))  # for car,ped,cyc
    for i in range(bsize):
        idx = base_index + obj[i]
        if dataset.one_hot:
            ps, seg, center, hclass, hres, sclass, sres, rotangle, onehotvec = \
                dataset[idx]
            batch_one_hot_vec[i] = onehotvec
        else:
            ps, seg, center, hclass, hres, sclass, sres, rotangle = \
                dataset[idx]
        batch_data[i, ...] = ps[:, 0:num_channel]
        batch_label[i, :] = seg
        batch_center[i, :] = center
        batch_heading_class[i] = hclass
        batch_heading_residual[i] = hres
        batch_size_class[i] = sclass
        batch_size_residual[i] = sres
        batch_rot_angle[i] = rotangle
    if dataset.one_hot:
        return batch_data, batch_label, batch_center, \
               batch_heading_class, batch_heading_residual, \
               batch_size_class, batch_size_residual, \
               batch_rot_angle, batch_one_hot_vec
    else:
        return batch_data, batch_label, batch_center, \
               batch_heading_class, batch_heading_residual, \
               batch_size_class, batch_size_residual, batch_rot_angle

def get_batch(dataset, idxs, start_idx, end_idx,
              num_point, num_channel,
              from_rgb_detection=False):
    ''' Prepare batch data for training/evaluation.
    batch size is determined by start_idx-end_idx

    Input:
        dataset: an instance of FrustumDataset class
        idxs: a list of data element indices
        start_idx: int scalar, start position in idxs
        end_idx: int scalar, end position in idxs
        num_point: int scalar
        num_channel: int scalar
        from_rgb_detection: bool
    Output:
        batched data and label
    '''
    if from_rgb_detection:
        return get_batch_from_rgb_detection(dataset, idxs, start_idx, end_idx,
            num_point, num_channel)

    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, num_point, num_channel))
    batch_label = np.zeros((bsize, num_point), dtype=np.int32)
    batch_center = np.zeros((bsize, 3))
    batch_heading_class = np.zeros((bsize,), dtype=np.int32)
    batch_heading_residual = np.zeros((bsize,))
    batch_size_class = np.zeros((bsize,), dtype=np.int32)
    batch_size_residual = np.zeros((bsize, 3))
    batch_rot_angle = np.zeros((bsize,))
    if dataset.one_hot:
        batch_one_hot_vec = np.zeros((bsize,3)) # for car,ped,cyc
    for i in range(bsize):
        if dataset.one_hot:
            ps,seg,center,hclass,hres,sclass,sres,rotangle,onehotvec = \
                dataset[idxs[i+start_idx]]
            batch_one_hot_vec[i] = onehotvec
        else:
            ps,seg,center,hclass,hres,sclass,sres,rotangle = \
                dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps[:,0:num_channel]
        batch_label[i,:] = seg
        batch_center[i,:] = center
        batch_heading_class[i] = hclass
        batch_heading_residual[i] = hres
        batch_size_class[i] = sclass
        batch_size_residual[i] = sres
        batch_rot_angle[i] = rotangle
    if dataset.one_hot:
        return batch_data, batch_label, batch_center, \
            batch_heading_class, batch_heading_residual, \
            batch_size_class, batch_size_residual, \
            batch_rot_angle, batch_one_hot_vec
    else:
        return batch_data, batch_label, batch_center, \
            batch_heading_class, batch_heading_residual, \
            batch_size_class, batch_size_residual, batch_rot_angle

def get_batch_from_rgb_detection(dataset, idxs, start_idx, end_idx,
                                 num_point, num_channel):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, num_point, num_channel))
    batch_rot_angle = np.zeros((bsize,))
    batch_prob = np.zeros((bsize,))
    if dataset.one_hot:
        batch_one_hot_vec = np.zeros((bsize,3)) # for car,ped,cyc
        # 建立车、行人、自行车one_hot编码初始
    for i in range(bsize):
        if dataset.one_hot:
            # ？？？？？看不懂啊
            ps,rotangle,prob,onehotvec = dataset[idxs[i+start_idx]]
            batch_one_hot_vec[i] = onehotvec
        else:
            ps,rotangle,prob = dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps[:,0:num_channel]
        batch_rot_angle[i] = rotangle
        batch_prob[i] = prob
    if dataset.one_hot:
        return batch_data, batch_rot_angle, batch_prob, batch_one_hot_vec
    else:
        return batch_data, batch_rot_angle, batch_prob


