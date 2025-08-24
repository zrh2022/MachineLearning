import numpy as np
from DQNNet import DqnNet
from DQNTrainer import DQNTrainer

# 示例使用
if __name__ == "__main__":
    # 1. 创建模型实例
    model = DqnNet()

    # 2. 创建训练器
    trainer = DQNTrainer(model, learning_rate=0.01)

    # 3. 生成一些随机数据用于演示
    # 假设我们有1000个样本，每个样本12个特征
    X_train = np.random.rand(1000, 12).astype(np.float32)
    # 假设输出是4个值的回归目标
    y_train = np.random.rand(1000, 4).astype(np.float32)

    # 4. 训练模型
    print("开始训练...")
    trainer.train(X_train, y_train, epochs=50, batch_size=32)

    # 5. 进行预测
    X_test = np.random.rand(5, 12).astype(np.float32)  # 5个测试样本
    predictions = trainer.predict(X_test)
    print("\n预测结果:")
    print(predictions)