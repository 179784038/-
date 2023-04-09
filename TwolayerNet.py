###################################################################################
'''
程序功能介绍：
首先，基于numpy库（没有使用Pytorch、tensorflow或keras等框架）实现了sigmoid、softmax、两层神经网络TwoLayerNet、
反向传播、Loss以及梯度的计算等。
在这个代码基础上实现以下需求：
1.学习率下降策略
2.L2正则化
3.优化器SGD
4.保存模型和加载模型
5.参数查找，包括学习率、隐藏层大小和正则化强度，最终根据查找的参数进行训练并保存模型
6.测试，导入模型，用经过参数查找后的模型进行测试，输出分类精度
7.可视化训练和测试的Loss曲线，测试的accuracy曲线并保存，以及可视化每层的网络参数（W1 W2 b1 b2）
8.可以联网的状态下下载MNIST数据集，也可以离线使用
9.为了方便训练的处理，对MNIST数据集进行了标准化和one-hot编码处理
具体说明如下：

1. 学习率下降策略：`lr_decay` 函数实现了学习率下降策略。可以在训练过程中调用此函数来根据 `epoch` 和 `decay_epoch` 下降学习率。

2. L2正则化：在 `TwoLayerNet` 类的 `__init__` 和 `loss` 方法中添加了 L2 正则化。`lambd` 参数表示正则化强度。

3. 优化器SGD：添加了 `SGD` 类作为优化器。

4. 保存模型和加载模型：添加了 `save_model` 和 `load_model` 函数，分别用于保存和加载模型。

5. 参数查找：`parameter_search` 函数实现了参数查找，包括学习率、隐藏层大小和正则化强度。最终会返回最佳的网络模型。

6. 测试：在 `main` 函数中，加载最佳的网络模型，并计算了测试集上的分类精度。

7. 可视化：添加了 `visualize_loss_and_accuracy` 和 `visualize_params` 函数，分别用于可视化训练和测试的 Loss 曲线、测试的 accuracy 曲线以及每层的网络参数（W1、W2、b1、b2）。

注意：为了使代码完整且可运行，

'''

#################################################################################
try:
    import urllib.request # 用于打开URL（主要是针对HTTP）
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path 
import gzip 
import pickle 
import os
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

import itertools
from collections import OrderedDict

url_base = 'http://yann.lecun.com/exdb/mnist/' # url字符串
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}  # 字典变量



dataset_dir = '/working/'
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

# 下载一个文件
def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    # 已经有了就不再下载了
    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")

# 下载 mnist 数据集，四个文件
def download_mnist():
    for v in key_file.values():
       _download(v)
        
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        labels = idx2numpy.convert_from_file(f)
    print("Done")
    
    return labels

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")    
    with gzip.open(file_path, 'rb') as f:
        data = idx2numpy.convert_from_file(f)
    data = data.reshape(-1, img_size)
    print("Done")
    
    return data


    
def _convert_numpy():
    dataset = {}
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])    
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    
    return dataset

def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T
    

def load_mnist(normalize=True, flatten=True, one_hot_label=False):

    if not os.path.exists(save_file):
        init_mnist()
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
            
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])
    
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

# 计算sigmoid层的反向传播导数（根据数学推导知道是y(1-y)）
def sigmoid_grad(x):
        return (1.0 - sigmoid(x)) * sigmoid(x)

def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值
        it.iternext()

        return grad


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, lambd=0.001):
        # 添加 L2 正则化参数 lambd
        self.lambd = lambd
        # 初始化网络
        self.params = {}
        # weight_init_std:权重初始化标准差
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size) 
                            # 用高斯分布随机初始化一个权重参数矩阵
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)


    def predict(self, x):
        # 前向传播，用点乘实现
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y


    def loss(self, x, t):
        y = self.predict(x)
        loss = cross_entropy_error(y, t)
        # 添加 L2 正则化
        weight_decay = 0
        for idx in (1, 2):
            W = self.params[f'W{idx}']
            weight_decay += 0.5 * self.lambd * np.sum(W ** 2)

        return loss + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum( y==t ) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

  # 高速版计算梯度，利用批版本的反向传播实现计算高速
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]  # 把输入的所有列一起计算，因此可以快速

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num  # 输出和标签的平均距离，作为损失值
        grads['W2'] = np.dot(z1.T, dy)
        # numpy数组.T就是转置
        grads['b2'] = np.sum(dy, axis=0)
        # 这里和批版本的Affine层的反向传播导数计算一样


        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads

# 添加 SGD 优化器类
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            if key in grads:
                params[key] -= self.lr * grads[key]

            
# 添加学习率下降策略
def lr_decay(lr, decay_rate, epoch, decay_epoch):
    if epoch % decay_epoch == 0:
        return lr * decay_rate
    else:
        return lr

# 添加保存模型方法
def save_model(network, file_path='model.pkl'):
    with open(file_path, 'wb') as f:
        pickle.dump(network.params, f)

# 添加加载模型方法
def load_model(file_path='model.pkl'):
    with open(file_path, 'rb') as f:
        params = pickle.load(f)

    input_size, hidden_size, output_size = 784, params['W1'].shape[1], params['W2'].shape[1]
    network = TwoLayerNet(input_size, hidden_size, output_size)
    network.params = params

    return network
###########################################################################################
#参数查找函数
#遍历所有可能的参数组合（学习率、隐藏层大小和正则化强度）。
#对于每个组合，我们创建一个simpleNet实例和一个SGD优化器实例。
#接着，我们进行一定次数的迭代训练，并在每个epoch（迭代次数整除训练数据量）更新学习率。
#最后，我们在训练集和验证集上评估模型的准确率，并将结果存储在字典中。我们根据验证集上的准确率选择最佳模型。
############################################################################################
def parameter_search(x_train, t_train, x_val, t_val, learning_rates, hidden_sizes, regularization_strengths):
    results = {}
    best_val = -1
    best_net = None
    best_lr = None
    for lr, hidden_size, reg in itertools.product(learning_rates, hidden_sizes, regularization_strengths):
        net = TwoLayerNet(input_size=784, hidden_size=hidden_size, output_size=10, lambd=reg)
        optimizer = SGD(lr=lr)

        # 训练
        iters_num = 10000
        train_size = x_train.shape[0]
        batch_size = 100
        epoch_size = max(train_size // batch_size, 1)
        decay_epoch = 10
        decay_rate = 0.5

        for i in range(iters_num):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            # 计算梯度
            grads = net.gradient(x_batch, t_batch)

            # 更新参数
            optimizer.update(net.params, grads)

            # 学习率衰减
            if i % epoch_size == 0:
                lr = lr_decay(lr, decay_rate, i // epoch_size, decay_epoch)
                optimizer.lr = lr

            # 添加打印信息
            if i % 1000 == 0:
                print(f"Iteration {i}: Current learning rate: {optimizer.lr}, Hidden size: {hidden_size}, Regularization: {reg}")

        # 评估
        train_acc = net.accuracy(x_train, t_train)
        val_acc = net.accuracy(x_val, t_val)

        results[(lr, hidden_size, reg)] = (train_acc, val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_net = net
            best_lr = lr
        
    return best_net, results,best_lr


def train_with_best_parameters(best_net, best_lr, x_train, t_train, x_test, t_test):
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 超参数
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num):
        # 获取mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 计算梯度
        grad = best_net.gradient(x_batch, t_batch)

        # 更新参数
        for key in ('W1', 'b1', 'W2', 'b2'):
            best_net.params[key] -= best_lr * grad[key]

        # 记录学习过程的损失变化
        train_loss = best_net.loss(x_batch, t_batch)
        test_loss = best_net.loss(x_test, t_test)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        if i % iter_per_epoch == 0:
            train_acc = best_net.accuracy(x_train, t_train)
            test_acc = best_net.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

    return train_loss_list, train_acc_list, test_acc_list, test_loss_list

# 可视化 Loss 曲线和 Accuracy 曲线
def visualize_loss_and_accuracy(train_loss_list, test_loss_list, train_acc_list, test_acc_list):
    # 计算准确率记录的间隔
    accuracy_interval = len(train_loss_list) // len(train_acc_list)

    # 根据准确率记录的间隔调整 epochs 变量
    epochs = np.arange(0, len(train_loss_list), accuracy_interval)

    # 绘制损失曲线
    plt.figure()
    plt.plot(train_loss_list, label='Training Loss')
    plt.plot(test_loss_list, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 绘制准确率曲线
    plt.figure()
    plt.plot(epochs, train_acc_list, label='Training Accuracy')
    plt.plot(epochs, test_acc_list, label='Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()



# 可视化网络参数
def visualize_params(params):
    # 可视化 W1
    plt.figure()
    plt.imshow(params['W1'], cmap='gray')
    plt.title('W1')
    plt.colorbar()
    plt.savefig('W1.png')
    plt.show()

    # 可视化 W2
    plt.figure()
    plt.imshow(params['W2'], cmap='gray')
    plt.title('W2')
    plt.colorbar()
    plt.savefig('W2.png')
    plt.show()

    # 可视化 b1
    plt.figure()
    plt.imshow(params['b1'].reshape(-1, 1), cmap='gray')
    plt.title('b1')
    plt.colorbar()
    plt.savefig('b1.png')
    plt.show()

    # 可视化 b2
    plt.figure()
    plt.imshow(params['b2'].reshape(-1, 1), cmap='gray')
    plt.title('b2')
    plt.colorbar()
    plt.savefig('b2.png')
    plt.show()


    
def main():
    # 数据准备
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    
    # 参数查找
    learning_rates = [1e-3, 1e-2, 1e-1]
    hidden_sizes = [50, 100, 200]
    regularization_strengths = [1e-5, 1e-4, 1e-3]

    
    x_val, t_val = x_train[-10000:], t_train[-10000:]
    x_train, t_train = x_train[:-10000], t_train[:-10000]

    best_net, results, best_lr = parameter_search(x_train, t_train, x_val, t_val, learning_rates, hidden_sizes, regularization_strengths)

    # 保存模型
    save_model(best_net)

    # 加载模型
    loaded_net = load_model()

    # 测试
    test_acc = loaded_net.accuracy(x_test, t_test)
    print("Test accuracy:", test_acc)

    # 训练并获取 Loss 和准确率列表
    train_loss_list, train_acc_list, test_acc_list, test_loss_list = train_with_best_parameters(best_net, best_lr, x_train, t_train, x_test, t_test)
    # 可视化训练和测试的 Loss 曲线，测试的 accuracy 曲线
    visualize_loss_and_accuracy(train_loss_list, test_loss_list, train_acc_list, test_acc_list)

    # 可视化每层的网络参数
    visualize_params(loaded_net.params)

if __name__ == "__main__":
    main()




