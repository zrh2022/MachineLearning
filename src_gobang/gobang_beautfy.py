import pygame
import math
import random
from pygame.locals import *

# 初始化游戏
pygame.init()
pygame.mixer.init()

# 游戏配置
BOARD_SIZE = 15
GRID_SIZE = 40
MARGIN = 60
STONE_RADIUS = 18

# 窗口尺寸
WIDTH = MARGIN * 2 + GRID_SIZE * (BOARD_SIZE - 1)
HEIGHT = MARGIN * 2 + GRID_SIZE * (BOARD_SIZE - 1) + 160

# 颜色配置
COLORS = {
    "background": (238, 200, 154),
    "board_line": (100, 60, 30),
    "text": (50, 50, 50),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "hover_black": (100, 100, 100, 150),
    "hover_white": (200, 200, 200, 150),
    "win_effect": (255, 50, 50)
}

# 加载资源
try:
    black_icon = [pygame.image.load(f'black_{i}.png') for i in range(3)]  # 需要准备3帧黑棋角色图
    white_icon = [pygame.image.load(f'white_{i}.png') for i in range(3)]  # 需要准备3帧白棋角色图
except:
    # 简易替代方案
    black_icon = [pygame.Surface((60, 60)) for _ in range(3)]
    white_icon = [pygame.Surface((60, 60)) for _ in range(3)]
    for i in range(3):
        black_icon[i].fill((0, 0, 0))
        white_icon[i].fill((255, 255, 255))

# 初始化屏幕
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("五子棋-豪华版")

# 字体设置
try:
    font = pygame.font.Font(r"C:\Windows\Fonts\msyh.ttc", 32)
    small_font = pygame.font.Font(r"C:\Windows\Fonts\msyh.ttc", 24)
except:
    font = pygame.font.Font(None, 32)
    small_font = pygame.font.Font(None, 24)


# 游戏状态
class GameState:
    def __init__(self):
        self.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.current_player = 1
        self.game_over = False
        self.hover_pos = None
        self.animations = []
        self.win_stones = []
        self.thinking_frame = 0
        self.player_switch_alpha = 0


game_state = GameState()


# 音效
# place_sound = pygame.mixer.Sound("place.wav")  # 需要准备音效文件
# win_sound = pygame.mixer.Sound("win.wav")


def draw_board():
    screen.fill(COLORS["background"])
    board_width = GRID_SIZE * (BOARD_SIZE - 1)

    # 绘制木质纹理
    for i in range(0, WIDTH, 64):
        for j in range(0, HEIGHT, 64):
            tex = pygame.Surface((64, 64), pygame.SRCALPHA)
            pygame.draw.line(tex, (210, 180, 140), (0, 0), (64, 64), 3)
            pygame.draw.line(tex, (210, 180, 140), (64, 0), (0, 64), 3)
            screen.blit(tex, (i, j))

    # 棋盘边框
    pygame.draw.rect(screen, COLORS["board_line"],
                     (MARGIN - 4, MARGIN - 4, board_width + 8, board_width + 8), 4)

    # 网格线
    for i in range(BOARD_SIZE):
        start = MARGIN
        end = MARGIN + board_width
        pygame.draw.line(screen, COLORS["board_line"],
                         (start, MARGIN + i * GRID_SIZE),
                         (end, MARGIN + i * GRID_SIZE), 2)
        pygame.draw.line(screen, COLORS["board_line"],
                         (MARGIN + i * GRID_SIZE, start),
                         (MARGIN + i * GRID_SIZE, end), 2)

    # 星位
    star_points = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]
    for x, y in star_points:
        pos = (MARGIN + x * GRID_SIZE, MARGIN + y * GRID_SIZE)
        pygame.draw.circle(screen, COLORS["board_line"], pos, 5)


def draw_players():
    # 绘制角色图标
    frame = (pygame.time.get_ticks() // 300) % 3
    y_offset = math.sin(pygame.time.get_ticks() / 300) * 5

    if game_state.current_player == 1:
        screen.blit(black_icon[frame], (MARGIN, HEIGHT - 120))
        # 思考气泡
        bubble_pos = (MARGIN + 80, HEIGHT - 100 + y_offset)
        pygame.draw.circle(screen, (255, 255, 255), bubble_pos, 20)
        pygame.draw.polygon(screen, (255, 255, 255),
                            [(bubble_pos[0] - 10, bubble_pos[1] + 15),
                             (bubble_pos[0], bubble_pos[1] + 5),
                             (bubble_pos[0] + 10, bubble_pos[1] + 15)])
        text = small_font.render("...", True, (0, 0, 0))
        screen.blit(text, (bubble_pos[0] - 10, bubble_pos[1] - 10))
    else:
        screen.blit(white_icon[frame], (WIDTH - MARGIN - 60, 20))
        # 思考气泡
        bubble_pos = (WIDTH - MARGIN - 80, 60 + y_offset)
        pygame.draw.circle(screen, (255, 255, 255), bubble_pos, 20)
        pygame.draw.polygon(screen, (255, 255, 255),
                            [(bubble_pos[0] - 10, bubble_pos[1] - 15),
                             (bubble_pos[0], bubble_pos[1] - 5),
                             (bubble_pos[0] + 10, bubble_pos[1] - 15)])
        text = small_font.render("...", True, (0, 0, 0))
        screen.blit(text, (bubble_pos[0] - 10, bubble_pos[1] - 10))


def draw_stones(game_over: bool):
    # 绘制常规棋子
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if game_state.board[i][j] != 0:
                pos = (MARGIN + i * GRID_SIZE, MARGIN + j * GRID_SIZE)
                color = COLORS["black"] if game_state.board[i][j] == 1 else COLORS["white"]
                pygame.draw.circle(screen, color, pos, STONE_RADIUS)

                # 高光效果
                highlight = (color[0] + 50, color[1] + 50, color[2] + 50) if game_state.board[i][j] == 1 \
                    else (color[0] - 50, color[1] - 50, color[2] - 50)
                pygame.draw.circle(screen, highlight,
                                   (pos[0] - STONE_RADIUS // 3, pos[1] - STONE_RADIUS // 3),
                                   STONE_RADIUS // 3)

    # 绘制动画效果

    for anim in game_state.animations[:]:
        progress = (pygame.time.get_ticks() - anim["start"]) / 300
        if progress > 1 or game_over:
            game_state.animations.remove(anim)
            continue

        size = STONE_RADIUS * (0.5 + 0.5 * progress)
        pos = (anim["pos"][0], anim["pos"][1])
        color = COLORS["black"] if anim["player"] == 1 else COLORS["white"]

        surface = pygame.Surface((STONE_RADIUS * 2, STONE_RADIUS * 2), pygame.SRCALPHA)
        pygame.draw.circle(surface, (*color, int(255 * (0.5 + 0.5 * progress))),
                           (STONE_RADIUS, STONE_RADIUS), int(size))
        screen.blit(surface, (pos[0] - STONE_RADIUS, pos[1] - STONE_RADIUS))


def draw_win_effect():
    if game_state.win_stones:
        # 连线效果
        for x, y in game_state.win_stones:
            pos = (MARGIN + x * GRID_SIZE, MARGIN + y * GRID_SIZE)
            radius = STONE_RADIUS + 5 * math.sin(pygame.time.get_ticks() / 200)
            pygame.draw.circle(screen, COLORS["win_effect"], pos, int(radius), 3)

        # 粒子效果
        if random.random() < 0.3:
            px = MARGIN + game_state.win_stones[0][0] * GRID_SIZE
            py = MARGIN + game_state.win_stones[0][1] * GRID_SIZE
            game_state.animations.append({
                "type": "particle",
                "pos": (px + random.randint(-20, 20), py + random.randint(-20, 20)),
                "color": COLORS["win_effect"],
                "start": pygame.time.get_ticks()
            })

        # 绘制粒子
        for anim in game_state.animations[:]:
            if anim["type"] == "particle":
                progress = (pygame.time.get_ticks() - anim["start"]) / 1000
                if progress > 1:
                    game_state.animations.remove(anim)
                    continue
                size = 5 * (1 - progress)
                alpha = 255 * (1 - progress)
                pos = anim["pos"]
                surface = pygame.Surface((10, 10), pygame.SRCALPHA)
                pygame.draw.circle(surface, (*anim["color"], int(alpha)),
                                   (5, 5), int(size))
                screen.blit(surface, pos)


def check_win(x, y):
    player = game_state.current_player
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    for dx, dy in directions:
        stones = [(x, y)]
        # 正向检查
        i, j = x + dx, y + dy
        while 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE and game_state.board[i][j] == player:
            stones.append((i, j))
            i += dx
            j += dy
        # 反向检查
        i, j = x - dx, y - dy
        while 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE and game_state.board[i][j] == player:
            stones.append((i, j))
            i -= dx
            j -= dy

        if len(stones) >= 5:
            game_state.win_stones = stones[:5]
            return True
    return False


def handle_click(pos):
    x = (pos[0] - MARGIN + GRID_SIZE // 2) // GRID_SIZE
    y = (pos[1] - MARGIN + GRID_SIZE // 2) // GRID_SIZE

    if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and game_state.board[x][y] == 0:
        game_state.board[x][y] = game_state.current_player
        # place_sound.play()

        # 添加落子动画
        game_state.animations.append({
            "type": "place",
            "pos": (MARGIN + x * GRID_SIZE, MARGIN + y * GRID_SIZE),
            "player": game_state.current_player,
            "start": pygame.time.get_ticks()
        })

        if check_win(x, y):
            # win_sound.play()
            game_state.game_over = True
        else:
            # 添加切换玩家动画
            game_state.player_switch_alpha = 255
            game_state.current_player = 2 if game_state.current_player == 1 else 1


# 主循环
clock = pygame.time.Clock()
running = True
while running:
    mouse_pos = pygame.mouse.get_pos()
    game_state.hover_pos = None

    # 事件处理
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == MOUSEBUTTONDOWN:
            if not game_state.game_over:
                handle_click(event.pos)
            else:
                game_state = GameState()

    # 更新悬停位置
    if not game_state.game_over:
        x = (mouse_pos[0] - MARGIN + GRID_SIZE // 2) // GRID_SIZE
        y = (mouse_pos[1] - MARGIN + GRID_SIZE // 2) // GRID_SIZE
        if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and game_state.board[x][y] == 0:
            game_state.hover_pos = (x, y)

    # 绘制界面
    draw_board()
    draw_stones(game_state.game_over)
    draw_players()

    # 绘制悬停效果
    if game_state.hover_pos and not game_state.game_over:
        x, y = game_state.hover_pos
        pos = (MARGIN + x * GRID_SIZE, MARGIN + y * GRID_SIZE)
        color = COLORS["hover_black"] if game_state.current_player == 1 else COLORS["hover_white"]
        surface = pygame.Surface((STONE_RADIUS * 2, STONE_RADIUS * 2), pygame.SRCALPHA)
        pygame.draw.circle(surface, color, (STONE_RADIUS, STONE_RADIUS), STONE_RADIUS)
        screen.blit(surface, (pos[0] - STONE_RADIUS, pos[1] - STONE_RADIUS))

    # 绘制玩家切换动画
    if game_state.player_switch_alpha > 0:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        text = font.render("轮到你了！", True, (255, 255, 255, game_state.player_switch_alpha))
        overlay.blit(text, text.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
        screen.blit(overlay, (0, 0))
        game_state.player_switch_alpha = max(0, game_state.player_switch_alpha - 5)

    # 绘制胜利效果
    if game_state.game_over:
        draw_win_effect()
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(overlay, (255, 255, 255, 150), (0, 0, WIDTH, HEIGHT))
        text = font.render(f"{'黑棋' if game_state.current_player == 1 else '白棋'} 胜利！",
                           True, (255, 0, 0))
        screen.blit(overlay, (0, 0))
        screen.blit(text, text.get_rect(center=(WIDTH // 2, HEIGHT // 2)))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
