import pygame
from pygame.locals import *

# 初始化游戏
pygame.init()

# 游戏配置

GRID_SIZE = 40
BOARD_SIZE = 15
MARGIN = 60
STONE_RADIUS = 18

# 自动计算窗口尺寸（核心修改）
WIDTH = MARGIN * 2 + GRID_SIZE * (BOARD_SIZE - 1)  # 窗口宽度
HEIGHT = MARGIN * 2 + GRID_SIZE * (BOARD_SIZE - 1)  # 窗口高度

# 颜色配置
COLORS = {
    "background": (238, 200, 154),
    "board_line": (100, 60, 30),
    "text": (50, 50, 50),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "hover": (200, 200, 200, 100)
}

# 初始化屏幕
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("双人五子棋")

# 字体设置
try:
    # Windows 系统推荐使用微软雅黑
    font = pygame.font.Font(r"C:\Windows\Fonts\msyh.ttc", 36)
    small_font = pygame.font.Font(r"C:\Windows\Fonts\msyh.ttc", 24)
except:
    # 如果找不到字体，尝试使用系统默认中文字体
    font = pygame.font.Font(None, 48)
    small_font = pygame.font.Font(None, 36)
    print("警告：未找到指定中文字体，请确认字体路径是否正确")

# 游戏状态
board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
current_player = 1
game_over = False
hover_pos = None


def draw_board():
    # 绘制背景
    screen.fill(COLORS["background"])

    # 绘制棋盘边框
    pygame.draw.rect(screen, COLORS["board_line"],
                     (MARGIN - 2, MARGIN - 2,
                      GRID_SIZE * (BOARD_SIZE - 1) + 4,
                      GRID_SIZE * (BOARD_SIZE - 1) + 4), 4)

    # 绘制网格线
    for i in range(BOARD_SIZE):
        start = MARGIN
        end = MARGIN + GRID_SIZE * (BOARD_SIZE - 1)
        # 水平线
        pygame.draw.line(screen, COLORS["board_line"],
                         (start, MARGIN + i * GRID_SIZE),
                         (end, MARGIN + i * GRID_SIZE), 2)
        # 垂直线
        pygame.draw.line(screen, COLORS["board_line"],
                         (MARGIN + i * GRID_SIZE, start),
                         (MARGIN + i * GRID_SIZE, end), 2)

    # 绘制星位
    star_points = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]
    for x, y in star_points:
        pos = (MARGIN + x * GRID_SIZE, MARGIN + y * GRID_SIZE)
        pygame.draw.circle(screen, COLORS["board_line"], pos, 5)


def draw_stones():
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] != 0:
                pos = (MARGIN + i * GRID_SIZE, MARGIN + j * GRID_SIZE)
                color = COLORS["black"] if board[i][j] == 1 else COLORS["white"]
                # 绘制棋子立体效果
                pygame.draw.circle(screen, color, pos, STONE_RADIUS)
                if color == COLORS["black"]:
                    highlight = (50, 50, 50)
                else:
                    highlight = (200, 200, 200)
                pygame.draw.circle(screen, highlight,
                                   (pos[0] - STONE_RADIUS // 2, pos[1] - STONE_RADIUS // 2),
                                   STONE_RADIUS // 3)


def draw_hover():
    if hover_pos and not game_over:
        x, y = hover_pos
        pos = (MARGIN + x * GRID_SIZE, MARGIN + y * GRID_SIZE)
        surface = pygame.Surface((STONE_RADIUS * 2, STONE_RADIUS * 2), pygame.SRCALPHA)
        pygame.draw.circle(surface, COLORS["hover"], (STONE_RADIUS, STONE_RADIUS), STONE_RADIUS)
        screen.blit(surface, (pos[0] - STONE_RADIUS, pos[1] - STONE_RADIUS))


def check_win(x, y):
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for dx, dy in directions:
        count = 1
        # 正向检查
        i, j = x + dx, y + dy
        while 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE and board[i][j] == current_player:
            count += 1
            i += dx
            j += dy
        # 反向检查
        i, j = x - dx, y - dy
        while 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE and board[i][j] == current_player:
            count += 1
            i -= dx
            j -= dy
        if count >= 5:
            return True
    return False


def show_message(text):
    text_surf = font.render(text, True, COLORS["text"])
    text_rect = text_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(text_surf, text_rect)

    restart_text = small_font.render("点击重新开始", True, COLORS["text"])
    restart_rect = restart_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
    screen.blit(restart_text, restart_rect)


def handle_click(pos):
    global current_player, game_over
    x = (pos[0] - MARGIN + GRID_SIZE // 2) // GRID_SIZE
    y = (pos[1] - MARGIN + GRID_SIZE // 2) // GRID_SIZE

    if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and board[x][y] == 0:
        board[x][y] = current_player
        if check_win(x, y):
            game_over = True
        else:
            current_player = 2 if current_player == 1 else 1


running = True
while running:
    hover_pos = None
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == MOUSEBUTTONDOWN and not game_over:
            handle_click(event.pos)
        elif event.type == MOUSEBUTTONDOWN and game_over:
            # 重置游戏
            board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
            current_player = 1
            game_over = False
        elif event.type == MOUSEMOTION and not game_over:
            # 计算悬停位置
            x = (event.pos[0] - MARGIN + GRID_SIZE // 2) // GRID_SIZE
            y = (event.pos[1] - MARGIN + GRID_SIZE // 2) // GRID_SIZE
            if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and board[x][y] == 0:
                hover_pos = (x, y)
            else:
                hover_pos = None

    # 绘制界面
    draw_board()
    draw_stones()
    draw_hover()

    # 显示当前玩家
    player_text = small_font.render(f"当前玩家: {'黑棋' if current_player == 1 else '白棋'}",
                                    True, COLORS["text"])
    screen.blit(player_text, (20, 20))

    # 显示胜利信息
    if game_over:
        winner = "黑棋" if current_player == 1 else "白棋"
        show_message(f"{winner} 获胜!")

    pygame.display.flip()

pygame.quit()
