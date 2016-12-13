import numpy as np
import sys
import itertools

#   This module implements the Neural Networks training algorithm for classification
#       with one hidden layer, and number of nodes that can be configured
#   uses sigmoid activation function for the hidden layer
#   uses softmax transfer function for the output prediction, for the multiclass prediction
#     Getting accuracy on an average about 70-75% for different configurations of the nodes in
#     the hiddlen layer.
#

class NeuralNet():
    # Class implements the Neural Network training algorithm for classification
    #   using sigmoid activation function for the hidden layer
    #   using softmax transfer function for the output prediction
    #
    #
    #

    def __init__(self):
        self.wi = None
        self.wo = None
        self.params = {
            'nh' : 16,      # param for hidden nodes
            'epochs': 20,   # param used for Stochastic gradient descent training
            'stepsize': 0.1 # learning rate/stepsize
        }

    def reset(self, new_parameters):
        # function to reset the classifier parameters with a new set
        self.wi = None
        self.wo = None
        for param in self.params:
            if param in new_parameters:
                self.params[param] = new_parameters[param]


    def sigmoid(self, z):
        '''
        Implements the standard sigmoid function
        :param z: a numpy array of any dimension
        :return: a numpy array with the element-wise sigmoid functions applied on the input array
        '''
        return 1.0 / (1.0 + np.exp(np.negative(z)))


    def dsigmoid(self, z):
        '''
        This function implements the gradient of the sigmoid
        :param z:
        :return:
        '''
        sigmoid_z = self.sigmoid(z)
        return sigmoid_z * (1.0 - sigmoid_z)


    def softmax(self, z):
        # softmax activation function for the output layer
        softmax_transfer = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        return softmax_transfer


    def get_mini_batch(self, X, y, batchSize):
        # function creates a batch of the input exemplars
        #   this is used for the SGD training where each iteration is processed for a batch of the examples
        for i in np.arange(0, X.shape[0], batchSize):
            yield (X[i:i + batchSize], y[i:i + batchSize])


    def learn(self, Xtrain, ytrain):

        # convert single column of values in y to encoded form with 4 columns,
        #   each having one col for each orientation

        encode = lambda val: [1 if val == 0 else 0] + [1 if val == 90 else 0] + [1 if val == 180 else 0] + [
            1 if val == 270 else 0]
        ytrain = np.apply_along_axis(encode, 1, ytrain)


        # ytrain = ytrain.reshape((len(ytrain), 1))
        self.wi = np.random.randn(Xtrain.shape[1], self.params['nh'])
        self.wo = np.random.randn(self.params['nh'] + 1, ytrain.shape[1])

        for epoch in range(self.params['epochs']):
            i = 0
            # print 'Training for epoch:', epoch
            for (batchX, batchY) in self.get_mini_batch(Xtrain, ytrain, 64):
                h = self.sigmoid(np.dot(batchX, self.wi))
                h = np.hstack((h, np.ones((h.shape[0], 1))))
                y_hat = self.softmax(np.dot(h, self.wo))

                # calculate the errors
                error = batchY - y_hat
                cost = sum(sum(np.nan_to_num(-batchY * np.log(y_hat) - (1 - batchY) * np.log(1 - y_hat))))

                # implement back propagation
                delta_o = - (error)
                gradient_wo = np.dot(h.T, delta_o)

                delta_i = np.multiply(np.dot(delta_o, self.wo[:-1].T), self.dsigmoid(np.dot(batchX, self.wi)))
                gradient_wi = np.dot(batchX.T, delta_i)

                self.wo = self.wo - self.params['stepsize'] * gradient_wo
                self.wi = self.wi - self.params['stepsize'] * gradient_wi
                i += 1
                # print "Cost for iteration: ", cost
        pass

    def predict(self, Xtest):

        h = self.sigmoid(np.dot(Xtest, self.wi))
        h = np.hstack((h, np.ones((h.shape[0], 1))))
        y_hat = self.softmax(np.dot(h, self.wo))

        y_hat = np.argmax(y_hat, 1)
        decode = lambda val: 0 if val == 0 else 90 if val == 1 else 180 if val == 2 else 270 if val == 3 else -1
        y_hat = np.apply_along_axis(decode, 1, y_hat.reshape(len(y_hat),1))

        return y_hat


def load_dataset(file_path):
    X = []
    y = []
    file_names = []
    with open(file_path, 'r') as f:
        for line in f:
            row = line.split(" ")
            X.append(row[2:])
            y.append(row[1])
            file_names.append(row[0])
    # dataset = np.genfromtxt(file_path)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape((len(y),1))
    file_names = np.asarray(file_names).reshape((len(y),1))

    # dataset = np.genfromtxt(file_path)
    # file_names = dataset[:, [0]]
    # X = dataset[:, 2:]
    # y = dataset[:, [1]]

    # Normalize the input features to range from 0 to 1
    for ii in range(X.shape[1]):
        maxval = np.max(np.abs(X[:,ii]))
        if maxval > 0:
            X[:,ii] = np.divide(X[:,ii], maxval)

    return (file_names, X, y)


def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct / float(len(ytest))) * 100.0


def calculatePerformance(actual_values, test_predictions):

    # classification_Accuracy = metrics.accuracy_score(actual_values, test_predictions)
    confusion_matrix = np.zeros((4,4), dtype=int)


    for i in range(len(test_predictions)):
        actual_index = int(actual_values[i]/90)
        predicted_index = int(test_predictions[i]/90)
        confusion_matrix[actual_index, predicted_index] = confusion_matrix[actual_index, predicted_index] + 1

    print "\n############-Model Report-################"

    print "Confusion Matrix:: Actuals along rows, Predicted along columns, \n" \
          "indices represent orientations: 0, 90, 180, 270 in that order"
    print confusion_matrix

    classification_Accuracy = sum(confusion_matrix[i,i] for i in range(4))/ float(np.sum(confusion_matrix))
    #
    #
    # tpr = float(true_positives) / (true_positives + false_negatives)
    # fpr = float(false_positives) / (false_positives + true_negatives)

    # print metrics.classification_report(y_true=list(actual_values), y_pred=test_predictions)
    print "Accuracy  : %s" % "{0:.3%}".format(classification_Accuracy)
    # print "True Positive Rate: %f, False Positive Rate: %f" % (tpr, fpr)




def predict_with_NN(train_datafile, test_datafile, num_hidden_nodes):
    # function invokes a NN algorithm and trains the NN with the training data
    #   with the specified number of nodes in the hidden layer
    #   makes predictions using the trained model
    #   computes accuracy and reports the same

    # read dataset
    filenames_train, Xtrain, ytrain = load_dataset(train_datafile)
    filenames_test, Xtest, ytest = load_dataset(test_datafile)

    clf = NeuralNet()
    clf.reset({'nh': num_hidden_nodes})
    clf.learn(Xtrain, ytrain)
    predictions = clf.predict(Xtest)
    accuracy = getaccuracy(ytest, predictions)
    print 'Accuracy on test set is: ', accuracy
    ytest = ytest.reshape(len(ytest), 1)
    predictions = predictions.reshape(len(predictions), 1)
    calculatePerformance(ytest, predictions)

    output_file = 'nnet_output.txt'
    with open(output_file, 'w+') as f:
        f.writelines(map(lambda row: str(row[0]) + " " + str(row[1]) + "\n", zip(filenames_test[:, 0], predictions[:,0])))
    pass



# #########################################################################
# Additional code for analysis Qn 4
# #########################################################################

def predict_with_NN_using_CV(train_datafile, test_datafile):
    # function invokes a NN algorithm and trains the NN with the training data
    #   IMplements 5-Fold cross-validation to identify the best parameters to train on
    #   The model is then built using the best prediction
    #   makes predictions using the trained model
    #   computes accuracy and reports the same

    # read dataset
    files_train, Xtrain, ytrain = load_dataset(train_datafile)
    files_test, Xtest, ytest = load_dataset(test_datafile)

    clf = NeuralNet()
    params = {
                  'nh': [8, 16, 32, 64, 128, 256]
              }

    param_lists = list(dict(itertools.izip(params, x)) for x in itertools.product(*params.itervalues()))

    n_folds = 5
    split_size = Xtrain.shape[0]/n_folds


    best_accuracy = -sys.maxint
    best_params = None

    for param_values in param_lists:
        clf.reset(param_values)
        print "Running {0} fold cross-validation with params {1}".format(n_folds, param_values)
        cumulative_accuracy = 0
        # run N-fold cross validation
        for r in range(n_folds):
            (split_training_set, validation_set) = get_n_splits(Xtrain, ytrain, split_size, r)
            clf.learn(split_training_set[0], split_training_set[1])
            predictions = clf.predict(validation_set[0])
            accuracy = getaccuracy(validation_set[1], predictions)
            print 'Accuracy on Validation set {' + str(r) + '}: ', accuracy
            cumulative_accuracy += accuracy

        avg_accuracy = cumulative_accuracy/float(5)
        print "Accuracy for CV run with params {0} is {1}".format(param_values, avg_accuracy)
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_params = param_values


    print "Best param setting is {}".format(best_params)

    # retrain with the best params setting
    # best_params = params
    clf.reset(best_params)
    clf.learn(Xtrain, ytrain)
    predictions = clf.predict(Xtest)
    accuracy = getaccuracy(ytest, predictions)
    print 'Accuracy on test set is: ', accuracy
    ytest = ytest.reshape(len(ytest),1)
    predictions = predictions.reshape(len(predictions),1)

    print ytest.shape
    print predictions.shape
    calculatePerformance(ytest, predictions)
    # test against the test set

    pass

def get_n_splits(X, y, splitsize, fold_num):
    # function to generate the splits for N-Fold cross validation

    start_index_for_validation = int(fold_num*splitsize)
    end_index_for_validation = int(fold_num*splitsize + splitsize)
    num_rows = int(X.shape[0])
    y = np.reshape(y, (len(y), 1))
    train_set = (np.vstack((X[0:start_index_for_validation, :],X[end_index_for_validation:num_rows, :])),
                np.vstack((y[0:start_index_for_validation, :], y[end_index_for_validation:num_rows, :])))

    validation_set = (X[start_index_for_validation:end_index_for_validation, :],
                      y[start_index_for_validation:end_index_for_validation, :])

    return (train_set, validation_set)



if __name__ == "__main__":
    predict_with_NN_using_CV('train-data.txt', 'test-data.txt')

