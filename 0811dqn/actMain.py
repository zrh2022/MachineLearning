import numpy as np
from DQNNet import DqnNet
from DQNTrainer import DQNTrainer
from src_dp_gridworld.gridworld import Gridworld
import matplotlib.pyplot as plt
import torch
from collections import defaultdict

def onehot(cur_tate):
    height, width = 3, 4
    vec = np.zeros(height * width, dtype=np.float32)
    y, x = cur_tate
    idx = width * y + x
    vec[idx] = 1.0
    return vec[np.newaxis, :]


# 示例使用
if __name__ == "__main__":
    # 检查是否有可用的GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")  # 选择 GPU
        print("GPU is available")
    else:
        device = torch.device("cpu")  # 否则使用 CPU
        print("GPU is not available, using CPU")

    # 1. 创建模型实例
    model = DqnNet()
    model = model.to(device)

    # 2. 创建训练器
    trainer = DQNTrainer(model, learning_rate=0.01)

    episodes = 10000
    loss_history = []
    world = Gridworld()

    for episode in range(episodes):
        if episode % 1000 == 0:
            print(episode)
        state = (0, 0)
        total_loss, cnt = 0, 0
        done = False
        while not done:
            state_onehot = onehot(state)
            action = model.get_action(state_onehot)
            next_state = world.to_next_position(state, action)
            reward = world.reward(state, action, next_state)
            if next_state == (0, 3):
                done = True
            next_state_onehot = onehot(next_state)
            loss = trainer.update(state_onehot, action, reward, next_state_onehot, done)
            total_loss += loss
            cnt += 1
            state = next_state
        average_loss = total_loss / cnt
        loss_history.append(average_loss)

    loss_history = [i.cpu().numpy() for i in loss_history]
    print(loss_history)
    plt.plot(loss_history)
    plt.show()

    #显示路线
    values = defaultdict()
    actions = defaultdict()

    for state in world.getStates():
        onehot_state = onehot(state)
        qs = model.forward(onehot_state).squeeze()
        action = qs.argmax()
        value = qs[action]
        values[state] = float(value.cpu().detach().numpy().item())
        actions[state] = int(action.cpu().detach().numpy().item())

    print(values)
    print(actions)
    world.render_env(3, 4, values, actions)

    # # 3. 生成一些随机数据用于演示
    # # 假设我们有1000个样本，每个样本12个特征
    # X_train = np.random.rand(1000, 12).astype(np.float32)
    # # 假设输出是4个值的回归目标
    # y_train = np.random.rand(1000, 4).astype(np.float32)
    #
    # # 4. 训练模型
    # print("开始训练...")
    # trainer.train(X_train, y_train, epochs=50, batch_size=32)
    #
    # # 5. 进行预测
    # X_test = np.random.rand(5, 12).astype(np.float32)  # 5个测试样本
    # predictions = trainer.predict(X_test)
    # print("\n预测结果:")
    # print(predictions)



