
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import random
from math import *
######
DEBUG = False
MAX_LOOPS = 1000
######

### HYPERS IMPORTED FROM HW.py FIEPOCHS = None
EPOCHS = None
BATCH = None
RATE = None
MOMENTUM = None
RC = None
INPUT_NORM_MODE = 'feature' # or 'z_feature'

RESOURCES = {
    "iris_train": "resources/iris-train.txt",
    "iris_test": "resources/iris-test.txt",
    "cifar_train": "resources/cifar-10-batches-py/data_batch_1",
    "cifar_test": "resources/cifar-10-batches-py/test_batch",
    "msd_train": "resources/YearPredictionMSD.txt",
    "msd_test": "resources/YearPredictionMSD.txt",
}

####
## HELPER FUNCTIONS
####
def update_hypers(dict):
    global EPOCHS
    EPOCHS = dict['EPOCHS']
    global BATCH
    BATCH = dict['BATCH']
    global RATE
    RATE = dict['RATE']
    global MOMENTUM
    MOMENTUM = dict['MOMENTUM']
    global RC
    RC = dict['RC']

def randomize_helper(*sets):
    """
    be sure and call function with a * if needing different sets
    an arbitrary set of indexed sets must be same length
    returns same sets randomized, but with same order of sets
    """
    print(len(sets))
    count = len(sets[0])
    for s in sets:
        assert count == len(s), "sets not same length"
    rand = [[] for s in sets]  # new sets
    indices = [i for i in range(count)]
    while indices:
        i = indices.pop(random.randint(0, len(indices) - 1))
        for k, s in enumerate(sets):
            rand[k].append(s[i])
    rand = [np.array(each) for each in rand]
    return rand


def hot_helper(Y, labels_override=None):
    """
    input normal Y array of labels or targets

    """
    # temp patch for sample test
    labels = set(range(labels_override)) if labels_override else set(Y)
    ##
    # labels = set(Y)
    Y_hot = []
    _labels = [label for label in labels]  # index 0 = 1, 1 = 2, etc
    for y in Y:
        # y_hot = [0 for i in len(hot)]
        y_hot = [0 for i in _labels]
        for n, m in enumerate(_labels):
            y_hot[n] = 1 if m == y else 0
        Y_hot.append(y_hot)

    return np.array(Y_hot), _labels


# changing from binary to python dictionary of (batch, label, data, filename)
def unpickle_helper(file):
    """input file path, output dictionary object, of labels, data, filenames """
    with open(file, 'br') as f:
        d = pickle.load(f, encoding='bytes')  # need to specify encoding

    return d

# counting function for total elements in arr2d)
def countel_helper(collection, s=0):
    """
    counts total first dimension elements in a multidimensional array
    """
    for nested in collection:
        if type(nested) == list:
            s += len(nested)
            countel_helper(nested)

    return s


####
## LEARNING FUNCTIONS, FORWARDS AND BACKWARDS
####
class Function(object):
    def __init__(self):
        pass

    def apply_forward(self, *args):
        pass
        ## apply forward....save outputs?


class Poisson_Regression(Function):
    """
    Log-Linear (generalized linear function)...not totally sure what that means.
    """
    @staticmethod
    def forward(logit, dim = -1):
        assert isinstance(logit, np.ndarray)
        y_hat = np.exp(logit)
        return y_hat

    @staticmethod
    def backward(self, *args):
        pass


class Poisson_Loss(Function):
    @staticmethod
    def forward(y, y_hat, dim=-1, reg_fn=None):
        y = y.astype(int)
        print(f'Loss Forward')if DEBUG else None
        assert isinstance(y_hat, np.ndarray)
        assert isinstance(dim, int)
        assert y.shape == y_hat.shape, f"y.shape: {y.shape}, yhat.shape{y_hat.shape}"

        if reg_fn:
            assert issubclass(reg_fn, Function)
        loss = np.exp(y_hat) - y * y_hat

        # _reg functions are just for adding the regularizer component if desired
        def _reg(lam, p, w):
            return loss + reg_fn.forward(lam, p, w) # returning log_loss

        return _reg if reg_fn else loss # returning log_loss

    @staticmethod
    def backward(y, x, w, y_hat_fn, logit_fn, reg_fn=None):
        print(f'LOSS BACKWARD') if DEBUG else None
        assert issubclass(y_hat_fn, Function)
        assert issubclass(logit_fn, Function)
        assert w.shape[-1] == 1, print(w.shape) # could be a x,1...ok

        ### Essentially:
        # dJ/dW = X_transpose @ (exp(X @ w) - Y)
        A = logit_fn.backward(x, w)[0]  # MatMul Backwards, just X.transpose
        y_hat = y_hat_fn.forward(logit_fn.forward(x, w))
        # the deriv of loss wrt poisson output
        # chain ruling for dl/dw
        d_loss = logit_fn.forward(A, y_hat) - logit_fn.forward(A, y)  # matmulling again by the transpose of X,

        # _reg functions are just for adding the regularizer component if desired
        def _reg(lam, p, w):
            return d_loss + reg_fn.backward(lam, p, w)

        return _reg if reg_fn else d_loss


# SEE MATMUL
class Ridge_Regression(Function):
    @staticmethod
    def forward(self, *args):
        pass # see MatMul
    @staticmethod
    def backward(self, *args):
        pass  # see MatMul


class Mean_Squared_Error(Function):
    @staticmethod
    def forward(y, y_hat, dim=-1, reg_fn=None):
        """
        """
        print(f'Loss Forward') if DEBUG else None
        assert isinstance(y_hat, np.ndarray)
        assert isinstance(dim, int)
        assert y.shape == y_hat.shape, f"y.shape: {y.shape}, yhat.shape{y_hat.shape}"
        if reg_fn:
            assert issubclass(reg_fn, Function)

        y_hat = np.round(y_hat,decimals=0) # round values
        loss = (y-y_hat)**2 # Elementwise squared difference
        assert loss.shape[-1] == 1, print(loss.shape)
        loss = loss.mean() # Means of N losses
        # _reg functions are just for adding the regularizer component if desired
        def _reg(lam, p, w):
            return loss + reg_fn.forward(lam, p, w)

        return _reg if reg_fn else loss


    @staticmethod
    def backward(y, x, w, y_hat_fn, logit_fn, reg_fn=None):
        print(f'LOSS BACKWARD') if DEBUG else None
        assert issubclass(y_hat_fn, Function)
        assert issubclass(logit_fn, Function)
        assert w.shape[-1] == 1, print(w.shape) # could be a x,1...ok
        assert y.shape[-1] == 1

        ### Essentially:
        # Logit Function is the MatMul
        # dJ/dW = X_transpose @ (exp(X @ w) - Y)
        A = logit_fn.backward(x, w)[0]  # MatMul Backwards, just X.transpose
        B = logit_fn.forward(x, w) - y
        d_loss = (2 / y.shape[0]) * logit_fn.forward(A, B) # Loss Backwards
        # chain ruling for dl/dw

        def _reg(lam, p, w):
            return d_loss + reg_fn.backward(lam, p, w)

        return _reg if reg_fn else d_loss


class Tikhonov(Function):
    @staticmethod
    def forward(X, Y):
        return np.linalg.inv(X.T @ X + RC * np.eye(X.shape[1])) @ X.T@ Y

class Softmax_Regression(Function):
    """
    """

    @staticmethod
    def forward(logit, dim = -1):
        """
        dim is dim along which...typically [-1]
        :param logit:
        :param dim:
        :return:
        """
        assert isinstance(logit,np.ndarray)
        assert isinstance(dim, int), print(type(dim))
        print(f'logit:{logit}') if DEBUG else None
        e = np.exp(logit - np.max(logit))
        e_sum = e.sum(dim)
        e_sum = e_sum.reshape(e_sum.shape[-2],e_sum.shape[-1],1)
        print(f'softmax denom: {e_sum}') if DEBUG else None
        soft = e / e_sum
        return soft

    @staticmethod
    def backward(l_fn):
        """"
        not needed....also really hard
        """
        pass

# RIDGE REGRESSION
class MatMul(Function):
    @staticmethod
    def forward(a, b):
        print(f'input: {a}') if DEBUG else None
        print(f'weights: {b}')if DEBUG else None
        # print(a,b)
        return np.matmul(a,b)

    @staticmethod
    def backward(a, b):
        """
        assuming ab or XW ordering, dy/dw
        multiply by outer in the actual outer function.
        """
        assert len(a.shape) == 3, f"ashape: {a.shape}"
        return a.transpose(0,2,1), b.transpose(0,2,1)
        # at = dw if XW, ignore b
        # bt = dw if WX, ignore a


class Cross_Entropy(Function):
    @staticmethod
    def forward(y, y_hat, dim=-1, reg_fn=None):
        """
        curry in lam, p, w
        only used for softmax classification....
        input final actvation, in this case softmax
        dim is dim for loss summation, typically -1,

        this is the element wise function, it must be averaged over the batch elsewhere
        """
        print(f'Loss Forward')if DEBUG else None
        assert isinstance(y_hat, np.ndarray)
        assert isinstance(dim, int)
        assert y.shape == y_hat.shape, f"y.shape: {y.shape}, yhat.shape{y_hat.shape}"
        if reg_fn:
            assert issubclass(reg_fn, Function)
        assert 0 not in y_hat, print(y_hat)

        loss = (y * np.log(y_hat)).sum(dim) * -1
        #summed over classes, multi-d tensors have extra step

        # _reg functionsa are just regularizer passed in
        def _reg(lam, p, w):
            return loss + reg_fn.forward(lam, p, w)

        return _reg if reg_fn else loss


    @staticmethod
    def backward(y, x, w, y_hat_fn, logit_fn, reg_fn=None):
        """
        y is y, x is input, yhatfn is output fn or softmax, logitfn is matmul, reg_fn is regul...
        curry in lam, p, w for regularization
        only used with softmax at moment, can just use softmax forward for y_hat (will have
        access in network...will also have access to X, W.

        arguments for other loss backwards should be same for other functions, but w/o fn inputs
        """
        print(f'LOSS BACKWARD')if DEBUG else None
        assert issubclass(y_hat_fn, Function)
        assert issubclass(logit_fn, Function)

        ### Essentially:
        # dJ/dW = X_transpose @ (Soft(X @ W) - Y)

        A = logit_fn.backward(x,w)[0] # MatMul Backwards, just X.transpose
        B = (y_hat_fn.forward(logit_fn.forward(x,w)) - y) # the deriv of loss wrt softmax output
        d_loss = logit_fn.forward(A,B) # Chain Rule: @ again by the transpose of X,
        assert d_loss.shape == (BATCH, w.shape[-2], w.shape[-1]), \
            f"ooops check loss deriv, it's {d_loss.shape}"

        def _reg(lam, p, w):
            return d_loss + reg_fn.backward(lam, p, w)

        return _reg if reg_fn else d_loss


class Regularize(Function):
    @staticmethod
    def forward(lam, p, w):
        """
        Note that This returns the full expressions (lam/p)||W||^p
        p is the degree of norm...ie 1 or 2 or
        can only be 1 or 2 for now.
        lam is regularization constant
        w is weights
        """
        assert p == 1 or 2
        norm = ((abs(w) ** p).sum()) ** (1 / p)
        reg = lam / p * norm ** p
        return reg

    @staticmethod
    def backward(lam, p, w):
        assert p == 1 or 2
        if p == 1:
            return np.piecewise(w, [w < 0, w > 0,w == 0], [-1, 1, 0])
        if p == 2:
            return lam * w


####
## ALL TRAINING AND TESTING FUNCTIONS, NETWORK INITIALIZATION
####
class Single_Layer_Network(object):
    """"
    """
    def __init__(
            self,
            inputs,
            targets,
            test_inputs,
            test_targets,
            loss_fn,
            output_fn,
            activation_fn,  #possibly same from output
            feat_d=1,
            n_classes=1,
            bias = 'embedded',
            K=1, # weight multiplier
            normalizing = False
    ):
        """
        inputs, weights should be arrays,
        classes are number of classes, 1 for regression
        forward is the forward function being called....can be lambda if combining, or just the
        loss fn is the loss function CLASS
        logit/activatoin fn is the activation fn if seperate from the output
        """
        # FUNCTIONAL ARGS:
        self.reg_fn = Regularize
        self.loss_fn = loss_fn
        self.activation_fn = activation_fn # sometimes same as output
        self.output_fn = output_fn # only different in case of softmax regression

        # DATA AND DIMENSIONAL ARGS:
        self.X = inputs
        self.Y = targets
        self.C = n_classes
        self.X_t = test_inputs
        self.Y_t = test_targets
        self.last_y_hat = 0

        # WEIGHTS init, BIAS EMBEDDED, randomize to gaussian around 0...not (0,1)
        feat_w = self.X.shape[-1]
        feat_w = feat_w + 1 if bias == 'embedded' else feat_w # feat width (includes bias)
        self.W = np.random.normal(0.0, 1.0, feat_d * feat_w * self.C)
        self.W = K * self.W / np.sqrt(feat_d*feat_w*self.C) # scaling Weights.
        self.W = self.W.reshape(feat_d,feat_w,self.C)
        # self.bias is embedded in inputs (X) as column of 1s, downstream in batching

        # PRE-BATCHING SHAPE ASSERTIONS, NORMALIZATION:
        # disabling testing during traing for msd classification
        # assert self.Y_t.shape[-1] == self.C, print(self.Y.shape, self.Y_t.shape, self.C)
        assert self.Y.shape[-1] == self.C, print(self.Y.shape, self.C)
        assert self.W.shape[-1] == self.C, print(self.W.shape)

        # NORMALIZATION
        # print(self.W)
        # print(self.X.shape)
        if normalizing:
            for i in range(self.X.shape[-1]): #feat width maybe already including bias,, no good here
                feat_column = self.X[:,i]
                feat_column_t = self.X_t[:,i]
                assert feat_column.shape[0] == self.X.shape[0], print(self.X.shape[0], feat_column.shape)
                l2_norm = np.linalg.norm(feat_column)
                std_dev = feat_column.std()
                avg = feat_column.mean() # use train stats for both testing and training inputs
                # feat_column = feat_column / l2_norm # for unitizing length
                # feat_column_t = feat_column_t / l2_norm # for unitizing length
                feat_column = (feat_column - avg) / std_dev # for z-score
                feat_column_t = (feat_column_t - avg) / std_dev # for z-score normalize
                np.testing.assert_almost_equal(feat_column.mean(), 0, decimal=5) # for z-score
                self.X[:, i] = feat_column
                self.X_t[:, i] = feat_column_t


        # UDPATING, LOGGING, EVALUATION
        # self.eval_array is an array data structure to track accuracy and generate accuracy
        # visualization like confusino matrix and mean accuracy curves. if only one class (
        # regression), then array is (epochs x 2 x 1), otherwise (epochs x classes x classes)
        if self.C == 1:
            self.training_eval_array = np.zeros((EPOCHS, 2, 1)) # not needed
            self.testing_eval_array = np.zeros((EPOCHS, 2, 1))  # not needed
        else:
            self.training_eval_array = np.zeros((EPOCHS, self.C, self.C))
            self.testing_eval_array = np.zeros((EPOCHS, self.C, self.C))
        self.velocity = 0 # inititalized at 0
        self.training_losses = []
        self.testing_losses = []

        # NOTES ON SHAPE:
        # if dtype == 'input':
        #     shape(batches, batch_size, feat_h, feat_w)
        # if dtype == 'weights':
        #     shape(batches, feat_d, feat_w, output)
        # not actually different weights obj per batch, but just helpful to think in that way in
        # terms of shape
        # if dtype == 'output':
        #     shape(batches, batch_size, feat_h, output)

        # TYPE ASSERTIONS
        assert isinstance(self.X, np.ndarray)
        assert isinstance(self.W, np.ndarray)
        # assert isinstance(self.C, np.ndarray)
        assert isinstance(self.Y, np.ndarray)
        if self.activation_fn:
            assert issubclass(self.activation_fn, Function)
        assert issubclass(self.output_fn, Function)
        assert issubclass(self.loss_fn, Function)

    def forward(self):  # argument must be curried
        if self.output_fn == MatMul:
            ret = lambda x, w: self.output_fn.forward(x, w)
        else:
            ret = lambda x,w: self.output_fn.forward(self.activation_fn.forward(x, w))
        return ret

    def update(self,d_loss):
        """
        input: d_loss
        output: none
        (changes weights only)

        updating weights....
        new velocity = (mu)(old velocity) - (a/BATCH)(new gradient.sum(0)
        new weights = old weights + new velocity

        the d_loss is a 3d matrix of Batch x Weights basically. So the idea is to take the
        average along the first dimension. That's really the key insight, is the shape of the
        gradient is extra dimensional...makse sense tho since each ouput has a gradienet!

        so here, the idea is to take average of weights
        """
        # NOTE: There is an extra sum step here, because using 3d tensors, not 2d matrices
        d_loss_sum = d_loss.sum(0)
        assert d_loss_sum.shape[-2] == self.W.shape[-2]
        assert d_loss_sum.shape[-1] == self.W.shape[-1]

        self.velocity = MOMENTUM * self.velocity - RATE * d_loss_sum / BATCH
        self.W = self.W + self.velocity
        print(f"weight updated:{self.W}") if DEBUG else None

    def evaluate(self,mode,epoch,Y,Y_hat):
        """
        needs to be called per batch per epoch. Builds the accuracy matrix.
        """
        assert mode == "training" or "testing"
        eval_array = self.training_eval_array if mode == "training" else self.testing_eval_array
        Y_hat_argsmax = np.argmax(Y_hat, axis=-1)
        Y_argsmax = np.argmax(Y, axis=-1)
        for j,k in zip(Y_hat_argsmax,Y_argsmax):
            i = epoch
            eval_array[i,j,k] += 1 #adding count to eval array

    def testing(self,epoch,p):
        '''
        called per epoch
        '''
        batches_complete = 0
        loops = 0
        epoch_loss = 0

        for X, Y in DataImport.batch_gen(self.X_t, self.Y_t, output=self.C):
            loops += 1
            if loops < 32: # cutting testing short at 32
                batches_complete += 1
                W = self.W
                assert len(X.shape) == 3
                assert len(Y.shape) == 3
                assert len(W.shape) == 3
                assert X.shape[-1] == self.W.shape[-2]
                assert X.shape[0] == BATCH
                Y_hat = self.forward()(X, W)  # currying to forward_fn
                if not p:
                    loss = self.loss_fn.forward(Y, Y_hat)
                elif p:
                    loss = self.loss_fn.forward(Y, Y_hat, reg_fn=self.reg_fn)(RC, p, W)
                epoch_loss += loss.sum()
                self.evaluate("testing",epoch,Y,Y_hat)
                print(f'Test Batch: {batches_complete}')
        print(f'Epoch Test Loss: {epoch_loss}')
        self.testing_losses.append(epoch_loss)

    def train(self, p):

        """
        p is degree of regularizing
        """
        for epoch in range(EPOCHS):
            count = 0
            epoch_loss = 0
            Y_hat = 0
            print(f'epoch:{epoch}')
            # stuff per epoch
            for X, Y in DataImport.batch_gen(self.X, self.Y, output=self.C):
                count +=1
                W = self.W
                # print(f'training batch: {count}')
                # per batch shape assertions
                assert len(X.shape) == 3
                assert len(Y.shape) == 3
                assert len(W.shape) == 3
                assert X.shape[0] == BATCH
                assert X.shape[-1] == W.shape[-2]

                Y_hat = self.forward()(X, W)  # currying to forward_fn

                if not p:
                    loss = self.loss_fn.forward(Y, Y_hat)
                elif p:
                    loss = self.loss_fn.forward(Y, Y_hat, reg_fn=self.reg_fn)(RC, p, W)
                epoch_loss += loss.sum()
                if not p:
                    d_loss = self.loss_fn.backward(
                        Y,
                        X,
                        W,
                        self.output_fn,
                        self.activation_fn,
                    )
                elif p:
                    d_loss = self.loss_fn.backward(
                        Y,
                        X,
                        W,
                        self.output_fn,
                        self.activation_fn,
                        self.reg_fn)(RC, p, W)
                self.update(d_loss)
                self.evaluate("training",epoch,Y,Y_hat)
            self.testing(epoch,p) # DISABLE FOR MSD CLASSIFICATION (DIFFERENT NUMBER OF CLASSES)
            print(f'Epoch Training Loss: {epoch_loss}')
            self.training_losses.append(epoch_loss)
            self.last_y_hat = Y_hat


####
## ALL DATA IMPORT FUNCTIONS
####
class DataImport(object):
    '''
    '''

    @staticmethod
    def delimited(path, delim, randomize=False, mode="classification", interval=None,y_norm_fn =
    None):
        """
        takes text file, reads line by line, and outputs array of inputs, outputs, and hot-vector
        outputs
        delim must be string...iris is space, msd is comma....
        interval is the closed set of rows from which to take examples...ie [0,10]. use 0-start
        indexes, not numbers
        None will be return
        full range.
        if randomize, have randomize function...

        can have an argument for type of data needed

        unsure
        """
        print("IMPORTING")
        X, Y = [], [] # Empty Lists

        # treating tuple inputs (ignore)
        (path, interval) = (path[0], path[1]) if type(path) == tuple else (path, interval)

        # Pulling in File
        lines = open(path).readlines()
        line_generator = enumerate((line for line in lines))

        # interval conditioned operations for sub-intervals of file, like in MSD
        if interval:
            interval[0] = interval[0] + len(lines) if interval[0] < 0 else interval[0]
            interval[1] = interval[1] + len(lines) if interval[1] < 0 else interval[1]
        for l, line in line_generator:
            if interval:
                if l < interval[0]:
                    continue
                elif l > interval[1]:
                    break
            # lines ----> feature rows, output
            row = [float(i) for i in line.split(delim)]
            y = row.pop(0)  # hot vectorize below this is good y for regression
            Y.append(y)  # may need to make single figure array
            X.append(row)  # y popped out already

       # Randomization and/or hot vectorization
        X, Y = (X, Y) if not randomize else randomize_helper(X, Y)
        hot = hot_helper(Y)
        Y = Y if mode == "regression" else hot[0]
        Y = np.array(Y)

        # Additional treatment of target (for MSD poisson regression, subtracting min)
        if y_norm_fn:
            assert callable(y_norm_fn)
            Y = y_norm_fn(Y)
        # asserting 2d shape
        Y = Y.reshape(Y.shape[0],1) if mode == "regression" else Y
        X = np.array(X)
        # return input, target, and label dictionary if hot vectorized
        return X, Y, hot[1]

    @staticmethod
    def batch_gen(X, Y, output=1, feat_h=1, feat_d=1, to_standardize=False):
        """
        note that this takes in inputs X and targets Y and ouputs a batched tuple.

        assuming input shape has examples as first dim, followed by sample shape.
        only inputs and targets are batched.

        Input: BATCHES x BATCH SIZE x FEATURE HEIGHT x INPUT FEATURE WIDTH
                N/10 x 10 x 1 x 2
        Weights: BATCHES x FEATURE DEPTH x FEATURE WIDTH (WEIGHTS) x LAYER OUTPUT WIDTH (CLASSES)
                N/10 x 1 x 2 x 3
        Output: BATCHES x BATCH SIZE x FEATURE HEIGHT x LAYER OUTPUT WIDTH (CLASSES)
                N/10 x 10 x 1 x 3
        input entire input matrix, indicating whether normalized or not.

        """
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        assert Y.shape[0] == X.shape[0]
        # slice / reshape X,Y to fit batch size...
        batch_size = BATCH
        classes = Y.shape[-1]  # how many different classes?
        feat_w = X.shape[-1]  # number feat_w
        N = X.shape[0]
        batches = X.shape[0] // batch_size  # number batches
        ex = batches * batch_size  # new number of examples
        sliced_X = X[0:ex, ...]  # using ellipses!
        sliced_Y = Y[0:ex, ...]

        # GENERALIZING INTO TENSORS FOR MULTIDIMENSIONAL FEATURES:
        sliced_tensor_X = sliced_X.reshape(1, 1, ex, feat_w)
        sliced_tensor_Y = sliced_Y.reshape(1, 1, ex, classes)
        sliced_tensor_X = sliced_tensor_X.transpose(0, 2, 1, 3)
        sliced_tensor_Y = sliced_tensor_Y.transpose(0, 2, 1, 3)
        batched_X = sliced_tensor_X.reshape(batches, batch_size, feat_h, feat_w)
        batched_Y = sliced_tensor_Y.reshape(batches, batch_size, feat_h, classes)
        batch_bias = np.ones((batch_size, feat_h, 1)) # ADDING BIAS IN ALL
        for b in range(batches):
            # i = (slice(None),) * fixed_dims + (b,)
            batch_x = batched_X[b, ...]
            batch_x = np.concatenate((  # adding bias column.
                batch_x, batch_bias
            ), axis=-1)
            np.testing.assert_allclose(
                batch_x[..., -1],1.0,err_msg= "Bias Concatenation Failure")
            batch_y = batched_Y[b, ...]
            yield (batch_x, batch_y)
            # should be iterated on in trainer, as a generator.

    @staticmethod
    def cifar(path, mode="classification"):
        """
        mode can be classification or regression
        cifar has 10 classes. This function simply outputs the batch as inputs and labels.
        possibility of reshaping /processing before output if required. outputs hotvector if required.

        input is already randomized
        """
        print("IMPORTING")
        batch = unpickle_helper(path)  # convert from binary
        cifar_labels = {
            0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck'
        }
        Y = batch[b'labels']
        Y = Y if mode == "regression" else hot_helper(Y)[0]
        X = batch[b'data']
        return np.array(X,dtype=np.float32), np.array(Y,dtype=np.float32), cifar_labels


####
## ALL PLOTTING FUNCTIONS
####
class Plot(object):
    """
    Visualize loss curve for test, and training data....
    1. loss curve vs epoch
    2. accuracy vs epoch
    Visualize the partitioning of space....
    - achieve by calling forward on the grid of points in the plot...(arbitrary density)

    this instance gets inputed the 'evaluation matricies' for testing and training data after
    training...maybe the training function outputs it?

    it also gets the loss log
    """

    def __init__(self):
        pass

    @staticmethod
    def curves(independent, *dependent, ind_label, dep_label, title, yscale='linear'):
        """
        everything after unpacker must be kewword arguements!
        must call with unpacker * if inputing a tuple for dependenate
        ..
        single independent variable, multiple dependent allowed
        labels/title are strings

        assuming first dependent is training, next is test

        Currently only supporting 2 depenedents....
        """
        assert iter(independent)
        assert iter(dependent)
        fig, ax1 = plt.subplots(1, 1, sharex=True, sharey=True)
        fig.dpi = 240
        c = random.random
        for each in dependent: #add curves to plot
            print(each)
            ax1.plot(independent, each['data'], color=(c(), c(), c()), label=each['label'])
        ax1.set(xlabel=ind_label, ylabel=dep_label, title=title, yscale=yscale)
        plt.legend(loc="lower right", frameon=False)
        plt.savefig(f"plots/{title.replace(' ','_')}.png")
        plt.show()

    @staticmethod
    def spaces_scatter(X, Y, forward_fn, ind_label, dep_label, title, size=50):
        """
        coming in are raw inputs, non normalized....
        this can only plot 3d data (2 features, 1 output as color). spaces plot (plot pts,
        model output)
        input 2d inputs and 1d y
        fn is the model function.
        assuming 3d tensor inputs
        :return:
        """
        x1_min, x1_max = X[:, 0].min() - .15, X[:, 0].max() + .25 #define numerical range
        x2_min, x2_max = X[:, 1].min() - .15, X[:, 1].max() + .25
        dist = x1_max - x1_min
        dim = dist / size  #  scalar of increment size for plot space
        # this essentially gridifies the range, but separates the arrays into layers for easy
        # array based computation. meshgrid returns indices in a 'ghost grid'
        xx1, xx2 = np.meshgrid( ## so xx1, xx2 are list of values along the x1 and x2 axes
            np.arange(x1_min, x1_max, dim),
            np.arange(x2_min, x2_max, dim),
        )
        bias = np.ones(xx1.shape) #1's tacked on for bias
        X_space = np.c_[xx1.ravel(), xx2.ravel(), bias.ravel()]  # listed coordinates for feeding.
        # now it gets triky...y_grid has 3 possible classes, so will have to get argmax of each row
        # and replace value with argmax, else a 0. then we sum it down across rows, reshape after...
        y_grid = forward_fn(X_space)  # will be the model forward function, calling with grid pts
        height = y_grid.shape[-2]
        for i in range(height):
            row = y_grid[:,i,:]
            m = np.argmax(row)
            for j in range(row.shape[-1]):
                row[...,j] = m if j == m else 0
        h = y_grid.sum(-1)
        h = h.reshape(xx1.shape)

        #Plotting
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
        fig.dpi = 240
        # scat = ax.scatter(X[:, 0], X[:, 1], c=Y, cmap='warm', edgecolor='k')
        # # legend = ax.legend(*scat.legend_elements(), loc="lower left", title="Classes")
        plt.contourf(xx1, xx2, h, cmap='viridis') #contour plot (boundares
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='cool', edgecolor='k')
        plt.xlabel(ind_label)
        plt.ylabel(dep_label)
        plt.title(title)
        plt.savefig(f"plots/{title.replace(' ','_')}.png")
        plt.show()

    @staticmethod
    def histogram(*data, ind_label, dep_label, title):
        """
        each data input is tuple of len3:
        1d data and count dict and data label, although dict isn't really needed.
        everything after unpacker is kwarg
        this is a 1d histogram. hist method auto-counts the data, so no need to count separately.
        output is simple histogram
        """
        fig, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
        ax.set(xlabel=ind_label,
               ylabel=dep_label,
               title=title)
        fig.dpi = 240
        c = lambda: random.random()
        for each in data:
            w, s, l = np.array(each[0]), each[1], each[2]
            w = w.flatten()
            assert w.ndim == 1, w.shape
            assert len(each) == 3, \
                f"{len(each)}\nPlease ensure data input is correct tuple of tuples"
            ax.hist(w, bins=len(s), label=l, color=(c(), c(), c()))
        plt.xlabel(ind_label)
        plt.ylabel(dep_label)
        plt.legend(loc="best", frameon=False)
        plt.savefig(f"plots/{title.replace(' ','_')}.png")
        plt.show()


    @staticmethod
    def confusion(confusion_array, label_dict, y_label="Predicted", x_label="Empirical",
                  title = "Confusion Matrix"):
        """
        takes as input the confusion array (predicted x empirical) and label dict
        remeber this is assuming the keys in dict are same as indices in the y / y_hat vectors
        assert that empircal data is on x axis, and predicted on y
        """
        # assert label dict is dict
        plt.figure(dpi=240, figsize=(5,5))
        plt.matshow(confusion_array, cmap='cool', norm=matplotlib.colors.LogNorm())
        tick_marks = np.arange(len(label_dict))
        plt.xticks(tick_marks, label_dict.values(), rotation=45)
        plt.yticks(tick_marks, label_dict.values())
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.savefig(f"plots/{title.replace(' ','_')}.png")
        plt.show()


class DataSimulatorHelper(object):
    def __init__(self, epoch_count, label_count, label_dict=None):
        size = epoch_count * label_count ** 2
        randarr = np.random.rand((size))
        for i, j in enumerate(randarr):
            randarr[i] = 1 if j > .5 else 0
        # epoch x yhat x y
        self.epoch_count = epoch_count
        self.label_count = label_count
        self.label_dict = label_dict
        self.eval_arr = (randarr.reshape(epoch_count, label_count, label_count)
                         + np.diag([random.randint(0, 10)] * label_count))

    @staticmethod
    def accuracy_list(eval_arr,label):
        """
        ONLY FOR CLASSIFICATION MODELS
        takes the eval array,  outputs list in a list. For concatenatino purposes and the curves
        graph..
        title is a string
        """
        accuracies = []
        for i in range(eval_arr.shape[0]):
            x = eval_arr[i, :, :]
            assert x.shape == (
                eval_arr.shape[-2], eval_arr.shape[-1]), f'shape is {x.shape}'
            # print(eval_arr.shape[-2], eval_arr.shape[-1])
            _x = x.sum(1)
            assert _x.shape == (eval_arr.shape[-2],), f'shape is {eval_arr.shape, _x.shape}'
            _sum_acc = 0
            for j in range(x.shape[0]):
                if _x[j]:
                    accuracy = x[j,j] / _x[j]
                else:
                    continue
                _sum_acc += accuracy
            mean_acc = _sum_acc / x.shape[0]
            assert mean_acc is not nan
            # print(mean_acc, _x, x.shape[0] )
            accuracies.append(mean_acc)
        return {"data":accuracies,"label":label}

    def loss_list(self):
        pass

    @staticmethod
    def hist_list(count):
        # count is number of different items
        data = np.random.randint(1000, count + 1000, count ** 2)
        dict = {}
        for i in data:
            if i not in dict.keys():
                dict[i] = 1
            else:
                dict[i] += 1
        return data, dict, f"test{random.randint(0,9)}"

    @staticmethod
    def data_2d(count):
        """
        This outputs X, Y, and Y_hat
        """
        X, Y = [], []
        for i in range(count):
            # random.seed(134314)
            x_1 = random.random()
            # random.seed(2434567)
            x_2 = random.random()
            X.append([x_1, x_2])
        for x_1, x_2 in X:
            y = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
            y = 2 if x_2 > .5 else y
            Y.append(y)
        return np.array(X), np.array(Y)

    @staticmethod
    def grid_data(X_model, jitter=3415e-5):
        """
        idea here is to bring in X_model, which is generated from plotting function. X_model is
        must output np array (sim yhat functino)
        """
        # j = lambda: jitter * -1 if random.random() > .5 else jitter
        j = lambda: 1

        Y_hat = []
        for x_1, x_2 in X_model:
            y = 1 if x_1 < (0.2 * j()) or x_1 > (0.8 * j()) else 0
            y = 2 if x_2 > (.5 * j()) else y
            Y_hat.append(y)
        return np.array(Y_hat)

    def label_dict(self):
        assert self.label_dict, "No Dict"
        return self.label_dict

    def epoch_list(self):
        return [i for i in range(self.epoch_count)]

    def confusion(self):
        return self.eval_arr.sum(0)




