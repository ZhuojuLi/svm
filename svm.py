import numpy as np
import matplotlib.pyplot as plt


# 1. 构造一个简单线性可分的数据集
def generate_data():
    np.random.seed(42)
    X1 = np.random.randn(20, 2) + np.array([2, 2])
    X2 = np.random.randn(20, 2) + np.array([-2, -2])
    X = np.vstack((X1, X2))
    y = np.array([1] * 20 + [-1] * 20)
    return X, y


# 2. SVM 类（硬间隔 + 梯度下降）
class SimpleLinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = y.copy()

        # 初始化权重
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    dw = self.lambda_param * self.w
                    db = 0
                else:
                    dw = self.lambda_param * self.w - y_[idx] * x_i
                    db = -y_[idx]

                # 梯度下降更新
                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)


# 3. 可视化
def plot_svm(X, y, model):
    def decision_boundary(x):
        return -(model.w[0] * x + model.b) / model.w[1]

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')

    # 画决策边界
    x0 = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    x1 = decision_boundary(x0)
    plt.plot(x0, x1, 'k--', label='Decision Boundary')

    # 间隔边界（支持向量附近）
    margin = 1 / np.linalg.norm(model.w)
    x1_margin_up = x1 + margin
    x1_margin_down = x1 - margin
    plt.plot(x0, x1_margin_up, 'k:', alpha=0.5)
    plt.plot(x0, x1_margin_down, 'k:', alpha=0.5)

    plt.title("Hard-Margin Linear SVM (Handwritten)")
    plt.legend()
    plt.show()


# 4. 运行
X, y = generate_data()
model = SimpleLinearSVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
model.fit(X, y)
predictions = model.predict(X)

# 准确率
acc = np.mean(predictions == y)
print(f"✅ Accuracy: {acc:.2f}")

# 可视化
plot_svm(X, y, model)
