import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# 1. 生成二维数据（2 类）
X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.2)

# 2. 训练 SVM 模型（RBF 核）
model = svm.SVC(kernel='rbf', C=1.0)
model.fit(X, y)


# 3. 可视化分类边界
def plot_svm_decision_boundary(model, X, y):
    # 网格范围
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 网格预测
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制区域
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    # 绘制训练样本
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')

    # 绘制支持向量
    sv = model.support_vectors_
    plt.scatter(sv[:, 0], sv[:, 1], s=100, facecolors='none', edgecolors='black', label='Support Vectors')

    plt.title("SVM Decision Boundary (RBF Kernel)")
    plt.legend()
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


# 调用可视化函数
plot_svm_decision_boundary(model, X, y)
