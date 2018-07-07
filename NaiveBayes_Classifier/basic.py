import numpy as np
sqrt_pi = (2 * np.pi) ** 0.5


class NaiveBayes:

    def __init__(self):
        self._x = self._y = None # 记录训练集的变量
        self._data = None 
        self._func = None # 模型核心：决策函数，根据输入的x,y输出对应的后验概率
        self._n_possibilities = None # 记录各个维度特征取值个数的数组
        self._labelled_x = None # 记录按类别分开后的输入数据的数组
        self._label_zip = None # 记录类别相关信息的数组
        self._cat_counter = None # 记录第i类数据的个数
        self._con_counter = None # 记录数据条件概率的原始极大似然估计
        self.label_dict = None # 记录数值化类别时的转换关系
        self._feat_dicts = None # 记录数值化特征各维度特征时的转换关系


    # 留下抽象方法让子类定义
    def feed_data(self, x, y, sample_weight=None):
        pass

    # 留下抽象方法让子类定义, sample_weight表示权重
    def feed_sample_weight(self, sample_weight=None):
        pass
    
    # 重载  __getitem__运算符以避免定义大量property
    def __getitem__(self,item):
        if isinstance(self,item):
            return getattr(self,"_"+item)


    # 定义计算先验概率的函数， lb就是各个估计中的平滑项\lambda
    # lb默认值为1，即拉普拉斯平滑
    def get_prior_probability(self, lb=1):
        return [(c_num + lb) / (len(self._y) + lb * len(self._cat_counter))
                for c_num in self._cat_counter]

    # 定义具有普适性的训练函数
    def fit(self, x=None, y=None, sample_weight=None, lb=None):
        # 如果有x,y输出，就用x,y初始化模型
        if x is not None and y is not None:
            self.feed_data(x, y, sample_weight)
        # 调用核心算法得到决策函数
        self._func = self._fit(lb)
    
    # 留下抽象算法核心让子类定义
    def _fit(self, lb):
        pass

    # 定义预测单一样本的函数
    # get_raw_result 控制该函数是输出预测的类别还是输出相应的后验概率
    # False:类别，True:后验概率
        def predict_one(self, x, get_raw_result=False):
        # 将输入的数据数值化
        if isinstance(x,np.ndarray)
            x = x.tolist()
        else:
            x = x[:]
        x = self._transfer_x(x)
        m_arg, m_probability = 0, 0
        # 遍历各类别，找到能使后验概率最大化的类别
        for i in range(len(self._cat_counter)):
            p = self._func(x, i)
            if p > m_probability:
                m_arg, m_probability = i, p
        if not get_raw_result:
            return self.label_dict[m_arg]
        return m_probability

    # 定义预测多样本的函数，本质是不断调用上面定义的predict_one函数
    def predict(self, x, get_raw_result=False):
        return np.array([self.predict_one(xx, get_raw_result) for xx in x])
    
    def _transfer_x(self, x):
        return x

    # 定义能对新数据进行评估的方法，这里是准确率
    def evaluate(self,x,y):
        y_pred=self.predict(x)
        print("Accuracy:{:12.6}%".format(100*np.sum(y_pred==y)/len(y)))
        