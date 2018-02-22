import numpy as np
from sklearn import svm
from itertools import combinations

class MulticlassSVM:

    def __init__(self, mode):
        if mode != 'ovr' and mode != 'ovo' and mode != 'crammer-singer':
            raise ValueError('mode must be ovr or ovo or crammer-singer')
        self.mode = mode

    def fit(self, X, y):
        if self.mode == 'ovr':
            self.fit_ovr(X, y)
        elif self.mode == 'ovo':
            self.fit_ovo(X, y)
        elif self.mode == 'crammer-singer':
            self.fit_cs(X, y)

    def fit_ovr(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovr_student(X, y)

    def fit_ovo(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovo_student(X, y)

    def fit_cs(self, X, y):
        self.labels = np.unique(y)
        X_intercept = np.hstack([X, np.ones((len(X), 1))])

        N, d = X_intercept.shape
        K = len(self.labels)

        W = np.zeros((K, d))

        n_iter = 1500
        learning_rate = 1e-8
        for i in range(n_iter):
            W -= learning_rate * self.grad_student(W, X_intercept, y)

        self.W = W

    def predict(self, X):
        if self.mode == 'ovr':
            return self.predict_ovr(X)
        elif self.mode == 'ovo':
            return self.predict_ovo(X)
        else:
            return self.predict_cs(X)

    def predict_ovr(self, X):
        scores = self.scores_ovr_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_ovo(self, X):
        scores = self.scores_ovo_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_cs(self, X):
        X_intercept = np.hstack([X, np.ones((len(X), 1))])
        return np.argmax(self.W.dot(X_intercept.T), axis=0)

    def bsvm_ovr_student(self, X, Y):
        '''
        Train OVR binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with labels as keys,
                        and binary SVM models as values.
        '''
        res = {}
        for label in self.labels:
            pX, pY = self.process_ovr(label, X, Y)
            res[label] = svm.LinearSVC().fit(pX, pY)
        return res

    def process_ovr(self, label, X, Y):
        pY = []
        for i in range(len(Y)):
            y = Y[i]
            if y == label:
                pY.append(1)
            else:
                pY.append(0)
        return X, pY


    def bsvm_ovo_student(self, X, Y):
        '''
        Train OVO binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with label pairs as keys,
                        and binary SVM models as values.
        '''
        res = {}
        for label_1, label_2 in combinations(self.labels, 2):
            pX, pY = self.process_ovo(label_1, label_2, X, Y)
            res[(label_1, label_2)] = svm.LinearSVC().fit(pX, pY)
        return res

    def process_ovo(self, label_1, label_2, X, Y):
        pX = []
        pY = []
        for i in range(len(Y)):
            x = X[i]
            y = Y[i]
            if y == label_1:
                pX.append(x)
                pY.append(1)
            elif y == label_2:
                pX.append(x)
                pY.append(0)

        return pX, pY

    def scores_ovr_student(self, X):
        '''
        Compute class scores for OVR.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        pred = []

        for x in X:
            scores = []
            for l, svm in self.binary_svm.items():
                scores.append(svm.decision_function([x]))
            pred.append(np.array(scores))

        return np.array(pred)

    def scores_ovo_student(self, X):
        '''
        Compute class scores for OVO.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        pred = []
        for x in X:
            scores = np.zeros(len(self.labels))
            for (l1, l2), svm in self.binary_svm.items():
                p = svm.predict([x])
                if p:
                    scores[l1] += 1
                else:
                    scores[l2] += 1
            pred.append(scores)
        return np.array(pred)


    def loss_student(self, W, X, y, C=1.0):
        '''
        Compute loss function given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The value of loss function given W, X and y.
        '''
        N = X.shape[0]
        K = W.shape[0]

        arr = np.arange(K)

        loss = 0
        for i in range(N):
            max_list = (1 - (arr == y[i])) + np.dot(X[i], W.T)
            loss += max(max_list) - np.dot(X[i], W[y[i]].T)

        l2_loss = np.sum(np.square(W))

        return (0.5*l2_loss) + (C*loss)

    def grad_student(self, W, X, y, C=1.0):
        '''
        Compute gradient function w.r.t. W given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The graident of loss function w.r.t. W,
            in a numpy array of shape (K, d).
        '''

        N = X.shape[0]
        K = W.shape[0]

        # d/dw l2_loss = W
        # shape (K, d).
        grad = np.copy(W)

        arr = np.arange(K)

        for i in range(N):
            y_i = y[i]
            x_i = X[i]
            max_list = (1 - (arr == y_i)) + np.dot(x_i, W.T)

            arg_max = np.argmax(max_list)
            if arg_max != y_i:
                g = C * x_i
                # only adjust the revelant gradent row
                grad[arg_max] += g
                grad[y_i] -= g

        return grad
