import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
import h5py
import math
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import pc_util


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--label', type=int, default=0, help='the query label [default: 0]')
parser.add_argument('--model', default='asg1_pointcloud_model', help='Model name: asg1_pointcloud_model [default: asg1_pointcloud_model]')
parser.add_argument('--batch_size', type=int, default=40, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/asg1_model.ckpt', help='model checkpoint file path [default: log/asg1_model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
FLAGS = parser.parse_args()

LABEL = FLAGS.label
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'asg1_log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'asg1_data/shape_names.txt'))]

HOSTNAME = socket.gethostname()

# get data files
QUERY_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'asg1_data/query.txt'))
DATABASE = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'asg1_data/test.txt'))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def setup_for_computation(num_votes, label):
    is_training = False


    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)

        # allocate space for global features
        gf_pl = tf.placeholder(tf.float32)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model

        pred, end_points, gf = MODEL.get_model(pointclouds_pl, is_training_pl)
        tf.get_variable_scope().reuse_variables()
        loss = MODEL.get_loss(pred, labels_pl, end_points)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)

        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        log_string("Model restored.")

        ops = {'pointclouds_pl': pointclouds_pl,
            'labels_pl': labels_pl,
            'is_training_pl': is_training_pl,
            'pred': pred,
            'loss': loss,
            'gf': gf}

    return sess, ops

def compute_gf(sess, ops, num_votes, data_file, label, topk=1):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    return_val = tf.placeholder(tf.float32, shape = (40, 128))
    fout = open(os.path.join(DUMP_DIR, 'asg1_pred_label.txt'), 'w')
    for fn in range(len(data_file)):
        log_string('----'+str(fn)+'----')
        unicode(data_file[fn])
        #current_data, current_label = provider.loadDataFile(data_file[fn])
        f = h5py.File(data_file[fn])
        current_data = f['data'][:]
        current_label = f['label'][:]
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        print(current_data.shape) # (40, 1024, 3)

        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        print(file_size)

        for batch_idx in range(num_batches): # fetch data of size BATCH_SIZE
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            cur_batch_size = end_idx - start_idx

            # Aggregating BEG
            batch_loss_sum = 0 # sum of losses for the batch
            batch_pred_sum = np.zeros((cur_batch_size, NUM_CLASSES)) # score for classes
            batch_pred_classes = np.zeros((cur_batch_size, NUM_CLASSES)) # 0/1 for classes
            for vote_idx in range(num_votes):
                rotated_data = provider.rotate_point_cloud_by_angle(current_data[start_idx:end_idx,: , :],
                                                  vote_idx/float(num_votes) * np.pi * 2)
                feed_dict = {ops['pointclouds_pl']: rotated_data,
                        ops['labels_pl']: current_label[start_idx:end_idx],
                        ops['is_training_pl']: is_training}

                loss_val, pred_val, gf_val = sess.run([ops['loss'], ops['pred'], ops['gf']],
                                          feed_dict=feed_dict)
                batch_pred_sum += pred_val
                batch_pred_val = np.argmax(pred_val, 1)
                for el_idx in range(cur_batch_size):
                    batch_pred_classes[el_idx, batch_pred_val[el_idx]] += 1
                batch_loss_sum += (loss_val * cur_batch_size / float(num_votes))
            # pred_val_topk = np.argsort(batch_pred_sum, axis=-1)[:,-1*np.array(range(topk))-1]
            # pred_val = np.argmax(batch_pred_classes, 1)
            pred_val = np.argmax(batch_pred_sum, 1)
            #put all the global features into one tensor
            if (batch_idx == 0):
                return_val = gf_val
            elif(batch_idx != 0):
                return_val = tf.concat([return_val, gf_val], 0)

            if (BATCH_SIZE == 800):
                return_val.eval() # convert the tensor back to an np.array
            # Aggregating END

    return return_val

def calculate_distance(coor1, coor2):
    #coor1 = (40, 128), coor2 = (800, 128)
    distances_gf = np.zeros((40, 800))

    # len1 supposedly equals to len2
    for i in range(40): # for each label
        for j in range(800): # for each datum in database
            for x in range(128):
                distances_gf[i][j] += math.pow((coor1[i, x] - coor2[j, x]), 2.0)
            distances_gf[i][j] = math.sqrt(distances_gf[i][j])

    return distances_gf # should be [40, 800]

def retrieve(sess, ops, model, label, database, k): #input network, query data and database
    # load query data and label, for each query
    num_votes = 1
    for fn in range(len(QUERY_FILES)):
        # compute feature of query pointcloud using our NN
        query_gf = compute_gf(sess, ops, num_votes, QUERY_FILES, LABEL) # [40, 128]

        # a dict to store all distances
        distances = {}
        #read from database for later retrieval
        db_data, db_label = provider.loadDataFile(DATABASE[fn])
        #compute global feature
        BATCH_SIZE = 800 # change BATCH_SIZE only for completing the computation in one loop
        db_item_gf = compute_gf(sess, ops, num_votes, DATABASE, LABEL) # should be [800, 128]

        with sess.as_default():
            db_item_gf = db_item_gf.eval() # convert it back to np ndarray

        # calculate feature distance with query data
        ret = calculate_distance(query_gf, db_item_gf) # ret should be [40, 800] of type np array
        # put the return value into the dict
        distances = ret  # should be [40, 800]
        # copy everything in the dict into a list
        distances_list = [[0 for x in range(800)] for y in range(40)]

        for i in range(len(distances)): # 40
            for j in range(len(distances_list[0])): # 800
                distances_list[i][j] = distances[i][j]

        # sort the list
        for idx in range(len(distances)): # 40
            distances_list[idx].sort() # sort in ascending order, with indexes 0-4 of smallest distacnes

        count = 0
        ix = 0
        jx = 0
        debugger = open(os.path.join(DUMP_DIR, 'debug.txt'), 'w') # for bookkeeping and debugging
        #retrieve the 5 objects with smallest distance
        while ix < len(distances): # 40
            jx = 0
            while jx < len(distances[0]): # 800
                if distances[ix][jx] == distances_list[ix][count]:
                    point_cloud = np.savetxt('pointcloud_%d_%d.xyz' % (ix, count), db_data[jx, :, :])
                    count += 1
                if count >= 5:
                    count = 0
                    break
                jx += 1
                if jx == len(distances[0]) and count < 5: # some are not yet found
                    jx = 0
            ix += 1
        print("Hooray! Program ends without errors!")

if __name__=='__main__':
    K = 5
    num_votes = 1
    with tf.Graph().as_default():
        sess_val, ops_val = setup_for_computation(num_votes, LABEL)
        retrieve(sess_val, ops_val, MODEL, LABEL, DATABASE, K)
    LOG_FOUT.close()
