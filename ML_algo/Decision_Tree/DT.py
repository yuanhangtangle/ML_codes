import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
class DecisionTree():
    def __init__(self, x, y, method = 'information_gain', pruning = None,
                 attrs_set = None, layer = 0, max_category = 10, dtypes = None, max_layer = None):
        self.layer = layer
        self.entropy = self.info_entropy(y)
        self.method = method
        self.pruning = pruning
        self.max_category = max_category
        self.sub_trees = {}
        self.label = None
        self.divide_attr = None
        self._labels = np.unique(y)
        self.num_sample = y.shape[0]
        self.attr_evaluation = []
        self.split_point = {}
        self.max_layer = max_layer

        if type(attrs_set) != np.ndarray:
            self.attrs_set = np.array(range(0, x.shape[1]))
            self.dtypes = np.apply_along_axis(self.get_dtype,0,x)  # 1 for discrete columns; 0 for continious columns
        else:
            self.attrs_set = attrs_set
            self.dtypes = dtypes

        classes = np.unique(y)
        if classes.shape[0] == 1:
            self.label = classes[0]
            return
        if np.sum(self.attrs_set) == -x.shape[1] :
            self.use_mode_label(y)
            return
        if self.max_layer is not None and self.layer >= self.max_layer:
            self.use_mode_label(y)
            return

        for attr in self.attrs_set:
            _x = x[:,attr]
            if method == 'information_gain' or method == 'gain_ratio_heuristic':
                a,b = self.info_gain(_x,y,attr)
                self.attr_evaluation.append(a)
                self.split_point[attr] = b
            elif method == 'gain_ratio':
                a, b = self.gain_ratio(_x, y, attr)
                self.attr_evaluation.append(a)
                self.split_point[attr] = b
            elif method == 'gini_index':
                self.attr_evaluation.append(self.neg_gini_index(x, y, attr))
            else:
                raise Exception('Unknown method: {}'.format(self.method))

        if method == 'gain_ratio_heuristic':
            self.attr_evaluation = np.asarray(self.attr_evaluation)
            threshold = np.mean(self.attr_evaluation[self.attr_evaluation != -1])
            self.attr_evaluation[self.attr_evaluation < threshold] = 0
            for attr in self.attrs_set:
                if attr > 0:
                    self.attr_evaluation[attr] = self.attr_evaluation[attr] / \
                                                 self.intrinsic_value(x[:, attr], attr, threshold = self.split_point[attr])
        #select optimal attribute to divide
        self.divide_attr = np.argmax(self.attr_evaluation)
        sub_attrs_set = np.copy(self.attrs_set)
        #recursion
        if self.dtypes[attr]:
            sub_attrs_set[self.divide_attr] = -1
            divide_values = np.unique(x[:,self.divide_attr])
            for v in divide_values:
                _x = x[x[:, self.divide_attr] == v]
                _y = y[x[:, self.divide_attr] == v]
                self.sub_trees[v] = DecisionTree(x = _x,y = _y, method = self.method, pruning = self.pruning,
                                                 attrs_set = sub_attrs_set, layer = self.layer + 1,
                                                 max_category = self.max_category, dtypes = self.dtypes,
                                                 max_layer=self.max_layer)
        else:
            x1 = x[x[:,self.divide_attr] <= self.split_point[self.divide_attr]]
            y1 = y[x[:,self.divide_attr] <= self.split_point[self.divide_attr]]
            if x1.shape[0] != 0:
                self.sub_trees[1] = DecisionTree(x = x1, y = y1,
                                                 method=self.method, pruning=self.pruning, attrs_set=sub_attrs_set,
                                                 layer=self.layer + 1, max_category=self.max_category, dtypes=self.dtypes,
                                                 max_layer=self.max_layer)
            x2 = x[x[:, self.divide_attr] > self.split_point[self.divide_attr]]
            y2 = y[x[:, self.divide_attr] > self.split_point[self.divide_attr]]
            if x2.shape[0] != 0:
                self.sub_trees[0] = DecisionTree(x = x2, y = y2,
                                                 method=self.method, pruning=self.pruning, attrs_set=sub_attrs_set,
                                                 layer=self.layer + 1, max_category=self.max_category, dtypes=self.dtypes,
                                                 max_layer=self.max_layer)

    def info_entropy(self,y):
        entropy = 0
        _labels = np.unique(y)
        n = y.shape[0]
        if n == 0:
            return 0
        for label in _labels:
            m = np.sum(y == label)
            entropy += m*np.log2(m)
        return -entropy/n + np.log2(n)

    def info_gain(self, _x, y, attr):
        if attr < 0:
            return -1, None
        elif self.dtypes[attr]: #1 for discrete data
            l = np.unique(_x)
            e = 0
            for v in l:
                _y = y[_x == v]
                e += self.info_entropy(_y)*_y.shape[0]
            return self.entropy - e/self.num_sample, None

        else:
            thresholds = self.get_thresholds(_x)
            gains = np.apply_along_axis(self.continuous_info_gain, 0, thresholds, _x, y)
            return np.max(gains), thresholds[0, np.argmax(gains)]

    def continuous_info_gain(self, threshold, _x, y):
        y1 = y[_x <= threshold]
        y2 = y[_x > threshold]
        info_gain = self.entropy - (self.info_entropy(y1)*y1.shape[0] + \
                               self.info_entropy(y2)*y2.shape[0])/self.num_sample
        return info_gain

    def intrinsic_value(self, _x, attr, threshold = None):
        intrin = 0
        if _x.shape[0] == 0:
            return intrin

        if self.dtypes[attr]:
            values = np.unique(_x)
            for v in values:
                m = np.sum(_x == v)
                intrin += m*np.log2(m)
        else:
            if threshold is None:
                raise Exception('threshold NOT provided!')
            m1 = np.sum(_x <= threshold)
            m2 = np.sum(_x > threshold)
            intrin1 =  m1*np.log2(m1) if m1 != 0 else 0
            intrin2 =  m2*np.log2(m2) if m2 != 0 else 0
            intrin = intrin1 + intrin2
        intrin = -intrin/self.num_sample + np.log2(self.num_sample)
        return intrin

    def get_thresholds(self, _x):
        _x = np.unique(_x)
        n = _x.shape[0]
        thresholds = []
        if n == 1:
            thresholds.append(_x[0])
        else:
            for i in range(n-1):
                thresholds.append((_x[i] + _x[i+1])/2)
        return np.array(thresholds).reshape(1, len(thresholds))

    def get_dtype(self, x):
        return np.unique(x).shape[0] <= self.max_category

    def gain_ratio(self, _x, y, attr):
        if attr < 0:
            return -1, None
        elif self.dtypes[attr]:
            a,b = self.info_gain(_x, y, attr)
            return a/self.intrinsic_value( _x, attr),b
        else:
            thresholds = self.get_thresholds(_x)
            gain_ratios = np.apply_along_axis(self.continuous_gain_ratio, 0, thresholds, _x, y,attr)
            gain_ratios[np.isnan(gain_ratios)] = -1
            return np.max(gain_ratios), thresholds[0, np.argmax(gain_ratios)]

    def continuous_gain_ratio(self, threshold, _x, y, attr):
        return self.continuous_info_gain(threshold, _x, y)/self.intrinsic_value(_x, attr, threshold)

    def gini_value(self, y):
        values = np.unique(y)
        n = y.shape[0]
        t = 0
        for v in values:
            m = np.sum(y == v)
            t -= m*m
        return 1 + t/(n*n)

    def neg_gini_index(self, x, y, attr):
        if attr < 0:
            return -2 #negative gini index is always greater than or equal to -1
        else:
            values = np.unique(x[:,attr])
            t = 0
            n = y.shape[0]
            for v in values:
                _y = y[x[:,attr] == v]
                m = _y.shape[0]
                t -= m*self.gini_value(_y)
            return t/n

    def use_mode_label(self,y):
        mode = 0
        num_mode = 0
        for label in self._labels:
            num_label = np.sum(y == label)
            if num_label > num_mode:
                num_mode = num_label
                mode = label
        self.label = mode

    def preprocess(self,x ):
        pass

    def predict(self, x):
        x = np.squeeze(x)
        if self.label is not None:
            return self.label
        elif self.dtypes[self.divide_attr]:  #disconcrete
            return self.sub_trees[x[self.divide_attr]].predict(x)
        else:  #continuous
            return self.sub_trees[x[self.divide_attr] <= self.split_point[self.divide_attr]].predict(x)

    def structure(self):
        #print('  '*self.layer,end = '')
        if self.layer:
            print(' |  '*(self.layer-1), end = '')
            print(' |--', end = '')
        if self.divide_attr is None:
            print('label:{}'.format(self.label))
        elif self.dtypes[self.divide_attr]:  #discrete
            print('attr: {}'.format(self.divide_attr))
        else:  #continuous
            print('attr/split:{}/{}'.format(self.divide_attr,self.split_point[self.divide_attr]))
        for v in self.sub_trees:
            self.sub_trees[v].structure()

    def info(self):
        print('method: {}\npruning: {}\nnumber of samples:{}'.format(self.method,self.pruning,self.num_sample))
        self.structure()

if __name__ == '__main__':
    iris = load_iris()
    print('-'*20 + 'without Normalizer' + '-'*20)
    X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, random_state = 1)
    for i in range(1,6):
        tree = DecisionTree(X_train, Y_train, max_layer=i)
        accuracy = np.sum(np.apply_along_axis(tree.predict,1,X_test) == Y_test)/Y_test.shape[0]
        print('i = {}, acc = {}'.format(i,accuracy))

    print('-' * 20 + 'with Normalizer' + '-' * 20)
    nm = MinMaxScaler()
    data = nm.fit_transform(iris.data)
    X_train, X_test, Y_train, Y_test = train_test_split(data, iris.target, random_state=1)
    for i in range(1, 6):
        tree = DecisionTree(X_train, Y_train, max_layer=i)
        accuracy = np.sum(np.apply_along_axis(tree.predict, 1, X_test) == Y_test) / Y_test.shape[0]
        print('i = {}, acc = {}'.format(i, accuracy))
