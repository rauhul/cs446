from sklearn import multiclass, svm


def sklearn_multiclass_prediction(mode, X_train, y_train, X_test):
    '''
    Use Scikit Learn built-in functions multiclass.OneVsRestClassifier
    and multiclass.OneVsOneClassifier to perform multiclass classification.

    Arguments:
        mode: one of 'ovr', 'ovo' or 'crammer'.
        X_train, X_test: numpy ndarray of training and test features.
        y_train: labels of training data, from 0 to 9.

    Returns:
        y_pred_train, y_pred_test: a tuple of 2 numpy ndarrays,
                                   being your prediction of labels on
                                   training and test data, from 0 to 9.
    '''

    if mode == 'ovr':
        return skl_ovr(X_train, y_train, X_test)
    elif mode == 'ovo':
        return skl_ovo(X_train, y_train, X_test)
    elif mode == 'crammer':
        return skl_crammer(X_train, y_train, X_test)
    else:
        return None, None

def skl_ovr(X_train, y_train, X_test):
    model = multiclass.OneVsRestClassifier(svm.LinearSVC()).fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    return y_pred_train, y_pred_test

def skl_ovo(X_train, y_train, X_test):
    model = multiclass.OneVsOneClassifier(svm.LinearSVC()).fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    return y_pred_train, y_pred_test

def skl_crammer(X_train, y_train, X_test):
    model = svm.LinearSVC(multi_class="crammer_singer").fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    return y_pred_train, y_pred_test

