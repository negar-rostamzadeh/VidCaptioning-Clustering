# Script is written by Li Yao to read the already extracted Dvs (M-vad) feature
import tables, numpy
import cPickle

def load_pkl(path):
    f = open(path, 'rb')
    try:
        rval = cPickle.load(f)
    finally:
        f.close()
    return rval

def dvs():
    path = '/data/lisatmp3/yaoli/datasets/lisadvd/'
    train_cap = numpy.load(path+'captions_train.npy')
    valid_cap = numpy.load(path+'captions_valid.npy')
    test_cap = numpy.load(path+'captions_test.npy')
    feature = tables.open_file(path+'googlenet_features.h5', 'r').root.feature
    feature_mapping = load_pkl(
            path+'key_videoID_value_StartEnd_googlenet_feature.pkl')
    train_ids = numpy.load(path+'videoIDs_train.npy')
    valid_ids = numpy.load(path+'videoIDs_valid.npy')
    test_ids = numpy.load(path+'videoIDs_test.npy')
    for i, vidID in enumerate(train_ids):
        start, end = feature_mapping[vidID]
        feat = feature[start:end]
        print 'load %d frames of %s'%(len(feat), vidID)

if __name__ == '__main__':
    dvs()


    
