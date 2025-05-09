# https://github.com/openai/improved-gan/blob/master/inception_score/model.py
# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import math
import glob
import tarfile
import scipy.misc
import numpy as np
from six.moves import urllib
import tensorflow as tf
import argparse
import imageio

MODEL_DIR = './imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

parser = argparse.ArgumentParser()
parser.add_argument('--metric', type=str, default='is', help='[is | fid | ndb | jsd]')
parser.add_argument('--pred_list', type=str, help='predict file list path')
parser.add_argument('--gt_list', type=str, help='real file list path')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--gpu_id', type=str, default='0', help='default is 0th GPU')
parser.add_argument('--resize', type=int, default=128, help='128 for NDB and JSD; 299 for FID and IS')
parser.add_argument('--num_bins', default=100, help='used in NDB and JSD')
parser.add_argument('--dataset', type=str, help='dataset to be tested')
args = parser.parse_args()

def print_eval_log(opt):
    message = ''
    message += '----------------- Eval ------------------\n'
    for k, v in sorted(opt.items()):
        message += '{:>20}: {:<10}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    print(message)


# Call this function with list of images. Each of elements should be a 
# numpy array with values ranging from 0 to 255.
def get_inception_score(images, splits=10):
  assert(type(images) == list)
  assert(type(images[0]) == np.ndarray)
  assert(len(images[0].shape) == 3)
  assert(np.max(images[0]) > 10)
  assert(np.min(images[0]) >= 0.0)

  # list of preprocessed images
  inps = []

  # convert images to float32
  for img in images:
    img = img.astype(np.float32)
    inps.append(np.expand_dims(img, 0))
  
  # batch size
  bs = 16
  # Perform prediction with Inception model, using batches of 16 images
  with tf.Session() as sess:
    preds = []
    n_batches = int(math.ceil(float(len(inps)) / float(bs)))
    for i in range(n_batches):
        print("Batch %d of %d" % (i + 1, n_batches))
        # get the batch of images 
        inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, axis=0)

        # run the Inception model and get the p(y|x) predictions
        pred = sess.run(softmax, {'InputTensor:0': inp})
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
      part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

# This function is called automatically.
def _init_inception():
  global softmax
  if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(MODEL_DIR, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
  # Sostituzione di tf.gfile.FastGFile con tf.io.gfile.GFile
  with tf.io.gfile.GFile(os.path.join(
      MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    # Import model with a modification in the input tensor to accept arbitrary
    # batch size.
    input_tensor = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 3],
                                  name='InputTensor')
    _ = tf.import_graph_def(graph_def, name='',
                            input_map={'ExpandDims:0':input_tensor})
  with tf.Session() as sess:
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            o.set_shape(tf.TensorShape(new_shape))
    w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
    softmax = tf.nn.softmax(logits)

if softmax is None:
  _init_inception()


if __name__ == '__main__':
    use_cuda = args.gpu_id != ''
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    batch_size = args.batch_size
    metric_mode = args.metric
    
    pred_list, gt_list = [], []
    if os.path.isdir(args.pred_list):
        # Se pred_list è una directory, raccogli tutti i file immagine
        pred_list = glob.glob(os.path.join(args.pred_list, '*'))
        pred_list = [f for f in pred_list if os.path.isfile(f)]  # Filtra solo i file
    else:
        with open(args.pred_list, 'r') as fin_pred:
            pred_list = [line.strip() for line in fin_pred]

    if metric_mode in ['fid', 'ndb', 'jsd']:
        with open(args.gt_list, 'r') as fin_gt:
            gt_list = [line.strip() for line in fin_gt]

    final_score = 0.0
    if metric_mode == 'is':
        import tensorflow as tf
        from scores.inception_score_tf import get_inception_score
        images = [imageio.imread(ll.strip()) for ll in pred_list]
        with tf.device('/device:GPU:{}'.format(args.gpu_id)):
             final_score, stddev = get_inception_score(images)
        print(final_score, stddev)
    else:
        print('Unknown metric mode.')

    logs = {'num_of_files': len(pred_list),
            'metric_mode': metric_mode,
            'final_score': final_score}
    print_eval_log(logs)