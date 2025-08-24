import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


class Gridworld:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]
        self.action_meaning = {0: "UP",
                               1: "DOWN",
                               2: "LEFT",
                               3: "RIGHT"}
        self.action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        self.reward_map = np.array([
            [0, 0, 0, 1.0],
            [0, None, 0, -1.0],
            [0, 0, 0, 0]
        ])

        self.goal_state = (0, 3)
        self.wall_state = (1, 1)
        self.start_state = (2, 0)
        self.agent_state = self.start_state

    def getHeight(self):
        return len(self.reward_map)

    def getWidth(self):
        return len(self.reward_map[0])

    def getShape(self):
        return self.reward_map.shape

    def getStates(self):
        states = []
        for h in range(self.getHeight()):
            for w in range(self.getWidth()):
                states.append((h, w))
        # 需要排除掉墙
        states.remove(self.wall_state)
        return states

    def to_next_position(self, state, action):
        current_move = self.action_move_map[action]
        next_position = (state[0] + current_move[0], state[1] + current_move[1])
        ny, nx = next_position

        # 检查越界
        if ny < 0 or ny >= self.getHeight() or nx < 0 or nx >= self.getWidth() or \
                next_position == self.wall_state:
            next_position = state
        return next_position

    def reward(self, state, action, next_state):
        ny, nx = next_state
        return self.reward_map[ny, nx]

    def render_env(self, rows, cols, values, actions):
        # 初始化全灰网格（NaN表示墙）
        grids = np.full((rows, cols), np.nan)

        # 填充颜色值
        for (row, col), value in values.items():
            if 0 <= row < rows and 0 <= col < cols:
                grids[row][col] = np.clip(value, -1, 1)
            else:
                print(f"警告：坐标 ({row}, {col}) 超出网格范围，已忽略")

        # 创建绿-白-红颜色映射
        colors = [(-1.0, '#FF0000'), (0.0, '#FFFFFF'), (1.0, '#00FF00')]
        cmap = LinearSegmentedColormap.from_list('custom', [c[1] for c in colors])
        cmap.set_bad(color='#707070')  # 缺失值显示灰色

        # 绘制图像
        plt.figure(figsize=((cols + 1) * 1.4, (rows + 1) * 1.4))
        ax = plt.gca()
        img = ax.imshow(grids,
                        cmap=cmap,
                        vmin=-1,
                        vmax=1,
                        interpolation='none',
                        origin='upper',
                        aspect='equal',
                        zorder=0)

        # 添加数值标签（关键修改点）
        for row in range(rows):
            for col in range(cols):
                if (row, col) in values:
                    value = values[(row, col)]
                    # 计算标签位置（左上角内部偏移）
                    x = col - 0.5 + 0.05  # 横向起始位置 + 5%偏移
                    y = row - 0.5 + 0.25  # 纵向起始位置 + 5%偏移
                    ax.text(x, y,
                            f"{value:.2f}",  # 保留两位小数
                            ha='left',  # 左对齐
                            va='bottom',  # 下对齐
                            color='black',  # 统一黑色字体
                            fontsize=12,
                            zorder=4)  # 确保在网格线上层

            # 添加方向箭头（新增功能）
            arrow_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}  # 方向映射
            for (row, col), direction in actions.items():
                if 0 <= row < rows and 0 <= col < cols:
                    if direction not in arrow_symbols:
                        print(f"警告(B)：坐标 ({row}, {col}) 方向值 {direction} 无效")
                        continue

                    # 计算箭头位置（右下角）
                    x = col - 0.5 + 0.85  # 横向起始位置 + 85%偏移
                    y = row - 0.5 + 0.85  # 纵向起始位置 + 85%偏移

                    ax.text(x, y,
                            arrow_symbols[direction],
                            ha='right',  # 右对齐
                            va='top',  # 上对齐
                            color='black',
                            fontsize=14,  # 增大字号
                            fontweight='bold',
                            zorder=5)  # 最高层级
                else:
                    print(f"警告(B)：坐标 ({row}, {col}) 超出网格范围，已忽略")

        # 坐标轴与网格线设置
        ax.set_xticks(np.arange(cols))
        ax.set_yticks(np.arange(rows))
        ax.set_xticklabels(np.arange(cols))
        ax.set_yticklabels(np.arange(rows))
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        # 网格线设置
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor",
                color="black",
                linestyle="-",
                linewidth=1,
                zorder=3)

        # 颜色条
        cbar = plt.colorbar(img, shrink=0.8)
        cbar.set_label('Value Scale', rotation=270, labelpad=15)

        # 坐标范围锁定
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)

        plt.show()
