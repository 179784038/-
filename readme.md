**训练和测试步骤**

(1)数据准备：
通过load_mnist函数获取MNIST数据集，并将数据进行归一化和One-hot编码处理。

(2)参数查找：
设定学习率、隐藏层大小、正则化强度等参数，使用parameter_search函数对不同参数组合进行搜索，返回最佳模型及相关参数。

(3)模型训练与保存：
使用最佳模型及参数进行模型训练，并将训练好的模型通过save_model方法保存到本地。

(4)模型加载与测试：
使用load_model方法加载本地保存的模型，并使用测试集对模型进行测试，计算并输出测试准确率。

(5)训练过程记录：
使用train_with_best_parameters方法对最佳模型进行训练，并记录训练过程中的Loss和准确率。

(6)可视化分析：
使用visualize_loss_and_accuracy方法将训练和测试的Loss和准确率可视化展示，并使用visualize_params方法将每层的网络参数可视化展示。

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
