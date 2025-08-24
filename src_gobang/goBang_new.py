import pygame
from pygame.locals import *

# 初始化游戏
pygame.init()

# 游戏配置
BOARD_SIZE = 15  # 棋盘尺寸
GRID_SIZE = 40  # 格子大小
MARGIN = 60  # 棋盘边距
STONE_RADIUS = 18  # 棋子半径

# 自动计算窗口尺寸
WIDTH = MARGIN * 2 + GRID_SIZE * (BOARD_SIZE - 1)
HEIGHT = MARGIN * 2 + GRID_SIZE * (BOARD_SIZE - 1) + 80  # 底部留信息区

# 颜色配置
COLORS = {
    "background": (238, 200, 154),
    "board_line": (100, 60, 30),
    "text": (50, 50, 50),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "hover_black": (0, 0, 0, 200),
    "hover_white": (255, 255, 255, 200)
}

# 初始化屏幕
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("双人五子棋")

# 中文字体设置（需要根据系统修改路径）
try:
    font = pygame.font.Font(r"C:\Windows\Fonts\msyh.ttc", 32)
    small_font = pygame.font.Font(r"C:\Windows\Fonts\msyh.ttc", 24)
except:
    font = pygame.font.Font(None, 32)
    small_font = pygame.font.Font(None, 24)
    print("警告：未找到中文字体，使用默认字体")

# 游戏状态
board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
current_player = 1
game_over = False
hover_pos = None


def draw_board():
    # 绘制背景
    screen.fill(COLORS["background"])

    # 绘制棋盘边框
    board_width = GRID_SIZE * (BOARD_SIZE - 1)
    pygame.draw.rect(screen, COLORS["board_line"],
                     (MARGIN - 2, MARGIN - 2, board_width + 4, board_width + 4), 4)

    # 绘制网格线
    for i in range(BOARD_SIZE):
        start = MARGIN
        end = MARGIN + board_width
        pygame.draw.line(screen, COLORS["board_line"],
                         (start, MARGIN + i * GRID_SIZE),
                         (end, MARGIN + i * GRID_SIZE), 2)
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
                pygame.draw.circle(screen, color, pos, STONE_RADIUS)
                # 绘制高光效果
                highlight_color = (50, 50, 50) if board[i][j] == 1 else (220, 220, 220)
                pygame.draw.circle(screen, highlight_color,
                                   (pos[0] - STONE_RADIUS // 3, pos[1] - STONE_RADIUS // 3),
                                   STONE_RADIUS // 4)


def draw_hover():
    if hover_pos and not game_over:
        x, y = hover_pos
        pos = (MARGIN + x * GRID_SIZE, MARGIN + y * GRID_SIZE)
        hover_color = COLORS["hover_black"] if current_player == 1 else COLORS["hover_white"]

        # 创建半透明表面
        surface = pygame.Surface((STONE_RADIUS * 2, STONE_RADIUS * 2), pygame.SRCALPHA)
        pygame.draw.circle(surface, hover_color, (STONE_RADIUS, STONE_RADIUS), STONE_RADIUS)
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
    # 半透明背景
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pygame.draw.rect(overlay, (255, 255, 255, 150), (0, 0, WIDTH, HEIGHT))
    screen.blit(overlay, (0, 0))

    # 文字内容
    text_surf = font.render(text, True, COLORS["text"])
    text_rect = text_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 20))
    screen.blit(text_surf, text_rect)

    restart_text = small_font.render("点击任意位置重新开始", True, COLORS["text"])
    restart_rect = restart_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 20))
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


# 游戏主循环
running = True
while running:
    hover_pos = None
    mouse_pos = pygame.mouse.get_pos()

    # 处理事件
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == MOUSEBUTTONDOWN:
            if not game_over:
                handle_click(event.pos)
            else:
                # 重置游戏
                board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
                current_player = 1
                game_over = False

    # 更新悬停位置
    if not game_over:
        x = (mouse_pos[0] - MARGIN + GRID_SIZE // 2) // GRID_SIZE
        y = (mouse_pos[1] - MARGIN + GRID_SIZE // 2) // GRID_SIZE
        if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and board[x][y] == 0:
            hover_pos = (x, y)

    # 绘制界面
    draw_board()
    draw_stones()
    draw_hover()

    # 绘制状态信息
    info_y = HEIGHT - 60
    player_text = small_font.render(f"当前玩家: {'黑棋' if current_player == 1 else '白棋'}",
                                    True, COLORS["text"])
    screen.blit(player_text, (MARGIN, info_y))

    # 显示胜利信息
    if game_over:
        winner = "黑棋" if current_player == 1 else "白棋"
        show_message(f"{winner} 获胜!")

    pygame.display.update()

pygame.quit()