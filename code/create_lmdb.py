import os
import random

import cv2
import numpy as np
import pandas as pd

import caffe
from caffe.proto import caffe_pb2
import lmdb

from params import *

def make_datum(img, label):
	return caffe_pb2.Datum(
		channels=3,
		width=img.shape[0],
		height=img.shape[1],
		label=label,
		data=np.rollaxis(img, 2).tostring())

train_lmdb = '../input/amv/train_lmdb'
validation_lmdb = '../input/amv/validation_lmdb'

os.system('rm -rf ' + train_lmdb)
os.system('rm -rf ' + validation_lmdb)

df = pd.read_csv('../input/amv/train/train.csv', delimiter=',')
train_data = [tuple(x) for x in df.values]

df = pd.read_csv('../input/amv/test/test.csv', delimiter=',')
test_data = [tuple(x) for x in df.values]

# shuffle train_data
random.shuffle(train_data)

print "Creating train_lmdb"

in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
	for in_idx, img_tuple in enumerate(train_data):
		if in_idx % 6 == 0:
			continue
		img_path = '../input/amv/train/%s.png' % img_tuple[0]
		img = cv2.imread(img_path, cv2.IMREAD_COLOR)
		label = int(img_tuple[1])
		datum = make_datum(img, label)
		in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
		print '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()

print '\nCreating validation_lmdb'

in_db = lmdb.open(validation_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
	for in_idx, img_tuple in enumerate(train_data):
		if in_idx % 6 != 0:
			continue
		img_path = '../input/amv/train/%s.png' % img_tuple[0]
		img = cv2.imread(img_path, cv2.IMREAD_COLOR)
		label = int(img_tuple[1])
		datum = make_datum(img, label)
		in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
		print '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()

print '\nFinished processing all images'