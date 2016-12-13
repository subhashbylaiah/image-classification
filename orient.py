import sys
import nnet
import time
import warnings

#   nnet.py implements Neural Network algorithm
# sys.argv[1:3] = ('train-data.txt', 'test-data.txt', 'nnet')

c = time.time()
if __name__ == "__main__":
    train_file, test_file, algo = sys.argv[1:4]

    if algo == 'nnet':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            nnet.predict_with_NN_using_CV(train_file, test_file)

print "Time taken:", (time.time() - c)/60
