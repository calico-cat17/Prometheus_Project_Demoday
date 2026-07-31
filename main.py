from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pygame

try:
    from back.score.inference import evaluate_deduction
except Exception as exc:
    evaluate_deduction = None
    EVALUATOR_IMPORT_ERROR = exc
else:
    EVALUATOR_IMPORT_ERROR = None

try:
    from game_engine import GameSession, handle_player_message
except Exception as exc:
    GameSession = None
    handle_player_message = None
    GAME_ENGINE_IMPORT_ERROR = exc
else:
    GAME_ENGINE_IMPORT_ERROR = None


# ============================================================
# 봉인된 모의고사 - 2D 추리 게임 프로토타입
#
# 조작
#   WASD : 이동
#   Q    : 문 열기/닫기, 단서 확인, NPC 대화, 사건 추론
#   ESC  : 창 닫기 / 게임 종료
#   Enter: 대화 입력 / 추리 제출
#   F11  : 전체화면 전환
#
# assets/
#   background.png
#   chanhyung.png
#   sieun.png
#   jiyeon.png
#   juni.png
# ============================================================

WIDTH, HEIGHT = 1120, 720
FPS = 60
TILE = 32

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(BASE_DIR, "front", "assets")
BACKGROUND_PATH = os.path.join(ASSET_DIR, "background.png")

START_FULLSCREEN = False

COLORS = {
    "bg": (22, 16, 12),
    "floor": (84, 49, 31),
    "floor2": (105, 61, 35),
    "wall": (44, 28, 22),
    "line": (186, 91, 43),
    "light": (255, 174, 83),
    "text": (255, 235, 206),
    "muted": (201, 164, 129),
    "panel": (44, 30, 26),
    "panel2": (67, 42, 31),
    "danger": (236, 92, 62),
    "green": (93, 145, 82),
    "blue": (94, 136, 176),
    "shadow": (12, 9, 8),
}


# ------------------------------------------------------------
# 폰트
# ------------------------------------------------------------
def find_korean_font() -> Optional[str]:
    candidates = [
        "malgungothic",
        "Malgun Gothic",
        "AppleGothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "NotoSansCJK",
        "UnDotum",
        "Arial Unicode MS",
    ]

    for name in candidates:
        path = pygame.font.match_font(name)
        if path:
            return path

    return None


pygame.init()
pygame.key.start_text_input()
pygame.display.set_caption("봉인된 모의고사")

fullscreen = START_FULLSCREEN


def setup_display() -> pygame.Surface:
    flags = pygame.SCALED
    if fullscreen:
        flags |= pygame.FULLSCREEN

    return pygame.display.set_mode((WIDTH, HEIGHT), flags)


def toggle_fullscreen() -> None:
    global screen, fullscreen
    fullscreen = not fullscreen
    screen = setup_display()


screen = setup_display()
clock = pygame.time.Clock()

FONT_PATH = find_korean_font()
FONT = pygame.font.Font(FONT_PATH, 20) if FONT_PATH else pygame.font.SysFont(None, 22)
FONT_SM = pygame.font.Font(FONT_PATH, 16) if FONT_PATH else pygame.font.SysFont(None, 18)
FONT_XS = pygame.font.Font(FONT_PATH, 14) if FONT_PATH else pygame.font.SysFont(None, 16)
FONT_LG = pygame.font.Font(FONT_PATH, 28) if FONT_PATH else pygame.font.SysFont(None, 30)
FONT_XL = pygame.font.Font(FONT_PATH, 40) if FONT_PATH else pygame.font.SysFont(None, 42)


def draw_text(
    surface: pygame.Surface,
    text: str,
    pos,
    font=FONT,
    color=None,
    max_width: Optional[int] = None,
    line_gap: int = 6,
) -> int:
    color = color or COLORS["text"]
    x, y = pos

    if max_width is None:
        for line in text.splitlines() or [""]:
            surface.blit(font.render(line, True, color), (x, y))
            y += font.get_height() + line_gap
        return y

    lines: List[str] = []

    for paragraph in text.splitlines() or [""]:
        current = ""

        for ch in paragraph:
            test = current + ch
            if font.size(test)[0] <= max_width or not current:
                current = test
            else:
                lines.append(current)
                current = ch

        if current:
            lines.append(current)
        elif not paragraph:
            lines.append("")

    for line in lines:
        surface.blit(font.render(line, True, color), (x, y))
        y += font.get_height() + line_gap

    return y


def load_background() -> Optional[pygame.Surface]:
    if not os.path.exists(BACKGROUND_PATH):
        print(f"[WARN] 배경 이미지가 없습니다: {BACKGROUND_PATH}")
        return None

    try:
        image = pygame.image.load(BACKGROUND_PATH).convert()
        return pygame.transform.smoothscale(image, (WIDTH, HEIGHT))
    except pygame.error as exc:
        print(f"[WARN] 배경 이미지 로드 실패: {exc}")
        return None


def load_portrait(filename: str, size=(180, 130)) -> pygame.Surface:
    path = os.path.join(ASSET_DIR, filename)

    if not os.path.exists(path):
        fallback = pygame.Surface(size)
        fallback.fill((55, 42, 36))
        pygame.draw.rect(fallback, COLORS["line"], fallback.get_rect(), 2)
        draw_text(fallback, "이미지 없음", (30, 50), FONT_SM, COLORS["muted"])
        return fallback

    try:
        image = pygame.image.load(path).convert_alpha()
        width, height = image.get_size()

        target_ratio = size[0] / size[1]
        source_ratio = width / height

        if source_ratio > target_ratio:
            crop_height = height
            crop_width = int(height * target_ratio)
        else:
            crop_width = width
            crop_height = int(width / target_ratio)

        rect = pygame.Rect(
            (width - crop_width) // 2,
            (height - crop_height) // 2,
            crop_width,
            crop_height,
        )

        image = image.subsurface(rect).copy()
        return pygame.transform.smoothscale(image, size)

    except pygame.error as exc:
        print(f"[WARN] 초상화 로드 실패 ({filename}): {exc}")
        fallback = pygame.Surface(size)
        fallback.fill((55, 42, 36))
        return fallback


def load_map_sprite(filename: str, size=(58, 78)) -> pygame.Surface:
    path = os.path.join(ASSET_DIR, filename)

    if not os.path.exists(path):
        fallback = pygame.Surface(size, pygame.SRCALPHA)
        pygame.draw.ellipse(fallback, (70, 48, 38), fallback.get_rect())
        pygame.draw.ellipse(fallback, COLORS["line"], fallback.get_rect(), 2)
        return fallback

    try:
        image = pygame.image.load(path).convert_alpha()
        width, height = image.get_size()

        # The current character art is centered in each source image.
        crop_width = min(width, int(width * 0.34))
        crop_height = min(height, int(height * 0.62))
        crop_x = (width - crop_width) // 2
        crop_y = int(height * 0.16)

        if crop_y + crop_height > height:
            crop_y = height - crop_height

        crop = image.subsurface(
            pygame.Rect(crop_x, crop_y, crop_width, crop_height)
        ).copy()
        sprite = pygame.transform.smoothscale(crop, size)

        mask = pygame.Surface(size, pygame.SRCALPHA)
        pygame.draw.rect(
            mask,
            (255, 255, 255, 255),
            mask.get_rect(),
            border_radius=18,
        )
        sprite.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        pygame.draw.rect(
            sprite,
            (252, 188, 112, 180),
            sprite.get_rect(),
            2,
            border_radius=18,
        )
        return sprite

    except pygame.error as exc:
        print(f"[WARN] 留??ㅽ봽?쇱씠??濡쒕뱶 ?ㅽ뙣 ({filename}): {exc}")
        fallback = pygame.Surface(size, pygame.SRCALPHA)
        pygame.draw.ellipse(fallback, (70, 48, 38), fallback.get_rect())
        return fallback


BACKGROUND = load_background()


# ------------------------------------------------------------
# 게임 상태
# ------------------------------------------------------------
@dataclass
class GameState:
    mode: str = "start"  # start, play, clue, dialogue, judge, judging, result
    discovered: List[str] = field(default_factory=list)
    conversation: Dict[str, List[str]] = field(default_factory=dict)
    active_npc: Optional[str] = None
    active_clue: Optional[str] = None
    input_text: str = ""
    composing_text: str = ""
    last_reply: str = ""
    judge_result: Optional[Dict] = None
    judging: bool = False
    result_scroll: int = 0
    notification: str = ""
    notification_until: int = 0

    def add_clue(self, clue_id: str) -> None:
        if clue_id not in self.discovered:
            self.discovered.append(clue_id)

    def notify(self, message: str, duration_ms: int = 1600) -> None:
        self.notification = message
        self.notification_until = pygame.time.get_ticks() + duration_ms


state = GameState()
dialogue_session = GameSession() if GameSession is not None else None
SHOW_COLLISION_DEBUG = False
START_BUTTON_RECT = pygame.Rect(440, 610, 240, 54)
RETRY_BUTTON_RECT = pygame.Rect(455, 596, 210, 50)


# ------------------------------------------------------------
# 단서 DB
# ------------------------------------------------------------
CLUES = {
    "copy_log": {
        "name": "복사기 스캔 기록",
        "location": "인쇄실",
        "desc": "복사기 로그에 7월 8일 22:30, 7월 9일 03:20 시험지 스캔 기록이 남아 있다.",
        "tags": ["홍지연 의심", "이찬형 핵심"],
    },
    "white_powder": {
        "name": "복사기 유리면의 흰 가루",
        "location": "인쇄실",
        "desc": "복사기 유리면 가장자리에 흰 가루가 묻어 있다. 강사실의 분필가루와 비슷해 보인다.",
        "tags": ["홍지연 미끼"],
    },
    "account_memo": {
        "name": "계좌 번호 메모장",
        "location": "인쇄실",
        "desc": "인쇄실 쓰레기통에서 상담실장의 계좌번호가 적힌 메모장이 발견됐다.",
        "tags": ["이찬형 핵심"],
    },
    "safe_keypad": {
        "name": "금고 키패드 지문",
        "location": "원장실",
        "desc": "금고 키패드에 지문이 거의 남아 있지 않고, 닦아낸 흔적이 있다.",
        "tags": ["은폐"],
    },
    "performance_sheet": {
        "name": "성과급 평가표",
        "location": "원장실",
        "desc": "성과급 기준은 전국 모의고사 평균, 정예반 유지율, 우수 성적자 비율이다. 낮은 반 강사에게 불리하다.",
        "tags": ["홍지연 동기처럼 보임"],
    },
    "consult_schedule": {
        "name": "상담 예약표",
        "location": "원장실",
        "desc": "이찬형은 아침 8시 상담 예정이며, 예약표에 별표가 표시되어 있다.",
        "tags": ["이찬형"],
    },
    "coffee_stain": {
        "name": "상담실 앞 커피 자국",
        "location": "복도",
        "desc": "상담실 앞 복도 바닥에 커피를 쏟은 흔적이 있다. 누군가 급하게 움직인 듯하다.",
        "tags": ["정황"],
    },
    "chalk_box": {
        "name": "홍지연 자리의 분필통",
        "location": "강사실",
        "desc": "홍지연 자리에는 분필통이 있다. 복사기 유리면의 흰 가루와 연결될 수 있다.",
        "tags": ["홍지연 미끼"],
    },
    "timetable": {
        "name": "강의 시간표",
        "location": "강사실",
        "desc": "7월 8일 22:30에 홍지연의 강의가 종료된 것으로 표시되어 있다.",
        "tags": ["홍지연"],
    },
    "attendance": {
        "name": "등원 기록",
        "location": "데스크",
        "desc": "7월 9일 06:30 김준이 학생의 등원 기록이 남아 있다.",
        "tags": ["준이 미끼"],
    },
    "postit_pw": {
        "name": "상담실 모니터 비밀번호 포스트잇",
        "location": "상담실",
        "desc": "상담실 모니터 비밀번호가 포스트잇에 적혀 있다. 보안이 허술하다.",
        "tags": ["상담실"],
    },
    "monitor_chat": {
        "name": "상담실 모니터 대화",
        "location": "상담실",
        "desc": "학부모와의 대화에서 ‘같은 계좌로 보내주세요’라는 내용이 보인다. 인쇄실 메모장의 계좌와 같다.",
        "tags": ["이찬형 핵심"],
    },
}


# ------------------------------------------------------------
# NPC DB
# ------------------------------------------------------------
NPCS = {
    "chanhyung": {
        "name": "이찬형",
        "role": "상담실장",
        "file": "chanhyung.png",
        "color": (75, 96, 128),
        "pos": (185, 525),
        "intro": "학부모 상담 준비 때문에 아침에 일찍 왔습니다. 시험지에는 관심도 없습니다.",
    },
    "sieun": {
        "name": "김시은",
        "role": "부원장",
        "file": "sieun.png",
        "color": (160, 110, 90),
        "pos": (915, 165),
        "intro": "시험지는 제가 어제 원장실 금고에 넣었습니다. 금고 비밀번호는 원장님과 저만 압니다.",
    },
    "jiyeon": {
        "name": "홍지연",
        "role": "수학 강사",
        "file": "jiyeon.png",
        "color": (120, 104, 92),
        "pos": (545, 175),
        "intro": "시험지 봉투는 오늘 처음 봤습니다. 어제는 수업 끝나고 자료만 프린트했어요.",
    },
    "juni": {
        "name": "김준이",
        "role": "학생",
        "file": "juni.png",
        "color": (85, 117, 97),
        "pos": (955, 535),
        "intro": "강의 듣고 바로 집에 갔어요. 아침에는 일찍 와서 시험장에만 있었어요.",
    },
}

PORTRAITS = {
    npc_id: load_portrait(npc["file"])
    for npc_id, npc in NPCS.items()
}

MAP_SPRITES = {
    npc_id: load_map_sprite(npc["file"])
    for npc_id, npc in NPCS.items()
}
PLAYER_SPRITE = load_map_sprite("chanhyung.png", (52, 72))


# ------------------------------------------------------------
# 맵 오브젝트 / 문
# ------------------------------------------------------------
@dataclass
class Interactable:
    id: str
    kind: str  # clue, npc, judge
    name: str
    rect: pygame.Rect
    clue_ids: List[str] = field(default_factory=list)
    npc_id: Optional[str] = None


@dataclass
class Door:
    id: str
    name: str
    rect: pygame.Rect
    orientation: str  # horizontal, vertical
    room_a: str
    room_b: str
    open: bool = True
    locked: bool = False


ROOMS = {
    "print_room": {
        "name": "인쇄실",
        "rect": pygame.Rect(36, 24, 294, 222),
    },
    "teacher_room": {
        "name": "강사실",
        "rect": pygame.Rect(344, 24, 414, 222),
    },
    "principal_room": {
        "name": "원장실",
        "rect": pygame.Rect(776, 24, 308, 222),
    },
    "hall": {
        "name": "복도",
        "rect": pygame.Rect(36, 258, 1048, 122),
    },
    "counsel_room": {
        "name": "상담실",
        "rect": pygame.Rect(36, 378, 307, 244),
    },
    "desk": {
        "name": "데스크",
        "rect": pygame.Rect(352, 378, 446, 244),
    },
    "classroom": {
        "name": "교실",
        "rect": pygame.Rect(814, 378, 270, 244),
    },
    "entrance": {
        "name": "입구",
        "rect": pygame.Rect(430, 620, 260, 90),
    },
}


# 문 위치는 배경 이미지에 맞춰 조정할 수 있음.
# 현재 아래쪽 상담실/교실은 각각 안쪽 벽의 출입구를 사용한다.
doors: List[Door] = [
    # background.png를 1120×720으로 축소했을 때의 실제 입구 좌표
    Door(
        id="door_print",
        name="인쇄실 문",
        rect=pygame.Rect(206, 244, 56, 15),
        orientation="horizontal",
        room_a="hall",
        room_b="print_room",
        open=False,
    ),
    Door(
        id="door_teacher",
        name="강사실 문",
        rect=pygame.Rect(520, 244, 54, 15),
        orientation="horizontal",
        room_a="hall",
        room_b="teacher_room",
        open=False,
    ),
    Door(
        id="door_principal",
        name="원장실 문",
        rect=pygame.Rect(885, 244, 54, 15),
        orientation="horizontal",
        room_a="hall",
        room_b="principal_room",
        open=False,
    ),
    Door(
        id="door_counsel",
        name="상담실 문",
        rect=pygame.Rect(337, 493, 15, 66),
        orientation="vertical",
        room_a="hall",
        room_b="counsel_room",
        open=False,
    ),
    Door(
        id="door_classroom",
        name="교실 문",
        rect=pygame.Rect(799, 493, 15, 66),
        orientation="vertical",
        room_a="hall",
        room_b="classroom",
        open=False,
    ),
]


objects: List[Interactable] = [
    Interactable(
        "copy_machine",
        "clue",
        "복사기",
        pygame.Rect(105, 65, 135, 100),
        ["copy_log", "white_powder"],
    ),
    Interactable(
        "trash",
        "clue",
        "인쇄실 쓰레기통",
        pygame.Rect(255, 155, 55, 55),
        ["account_memo"],
    ),
    Interactable(
        "safe",
        "clue",
        "원장실 금고",
        pygame.Rect(980, 85, 75, 85),
        ["safe_keypad"],
    ),
    Interactable(
        "desk_docs",
        "clue",
        "원장실 서류",
        pygame.Rect(860, 125, 105, 62),
        ["performance_sheet", "consult_schedule"],
    ),
    Interactable(
        "coffee",
        "clue",
        "커피 자국",
        pygame.Rect(760, 345, 95, 58),
        ["coffee_stain"],
    ),
    Interactable(
        "chalk",
        "clue",
        "분필통",
        pygame.Rect(500, 110, 75, 55),
        ["chalk_box", "timetable"],
    ),
    Interactable(
        "attendance",
        "clue",
        "데스크 기록지",
        pygame.Rect(500, 485, 110, 52),
        ["attendance"],
    ),
    Interactable(
        "postit",
        "clue",
        "포스트잇",
        pygame.Rect(112, 455, 55, 40),
        ["postit_pw"],
    ),
    Interactable(
        "monitor",
        "clue",
        "상담실 모니터",
        pygame.Rect(125, 415, 130, 70),
        ["monitor_chat"],
    ),
    Interactable(
        "judge",
        "judge",
        "사건 추론함",
        pygame.Rect(500, 585, 170, 62),
    ),
]

objects = [obj for obj in objects if obj.kind != "judge"]

for npc_id, npc in NPCS.items():
    x, y = npc["pos"]
    objects.append(
        Interactable(
            id=f"npc_{npc_id}",
            kind="npc",
            name=npc["name"],
            rect=pygame.Rect(x - 24, y - 30, 48, 64),
            npc_id=npc_id,
        )
    )


# ------------------------------------------------------------
# 플레이어 / 충돌
# ------------------------------------------------------------
player = pygame.Rect(545, 650, 28, 36)
player_speed = 3.2


# 문 위치는 비워두고, 나머지 벽만 충돌 처리한다.
walls = [
    # --------------------------------------------------------
    # 바깥 경계
    # --------------------------------------------------------
    pygame.Rect(0, 0, WIDTH, 18),
    pygame.Rect(0, 0, 28, HEIGHT),
    pygame.Rect(WIDTH - 28, 0, 28, HEIGHT),
    pygame.Rect(0, HEIGHT - 16, WIDTH, 16),

    # --------------------------------------------------------
    # 상단 세 방의 아래 벽
    # 실제 문 구간은 비워 둔다.
    # --------------------------------------------------------
    pygame.Rect(35, 244, 171, 16),
    pygame.Rect(262, 244, 258, 16),
    pygame.Rect(574, 244, 311, 16),
    pygame.Rect(939, 244, 146, 16),

    # 상단 방 사이 세로 벽
    pygame.Rect(329, 18, 16, 226),
    pygame.Rect(758, 18, 16, 226),

    # --------------------------------------------------------
    # 상담실 외벽
    # 오른쪽 벽의 y=493~559 구간만 문으로 비운다.
    # --------------------------------------------------------
    pygame.Rect(34, 370, 318, 16),
    pygame.Rect(34, 370, 16, 258),
    pygame.Rect(34, 622, 318, 16),
    pygame.Rect(337, 370, 15, 123),
    pygame.Rect(337, 559, 15, 79),

    # --------------------------------------------------------
    # 교실 외벽
    # 왼쪽 벽의 y=493~559 구간만 문으로 비운다.
    # --------------------------------------------------------
    pygame.Rect(799, 370, 286, 16),
    pygame.Rect(799, 370, 15, 123),
    pygame.Rect(799, 559, 15, 79),
    pygame.Rect(1070, 370, 15, 268),
    pygame.Rect(799, 622, 286, 16),

    # --------------------------------------------------------
    # 중앙 안내 데스크
    # 책상 전체가 아니라 실제 카운터 부분만 막는다.
    # --------------------------------------------------------
    pygame.Rect(420, 409, 304, 104),

    # --------------------------------------------------------
    # 최소한의 가구 충돌
    # 이동 가능 공간이 지나치게 좁아지지 않도록 핵심 가구만 사용
    # --------------------------------------------------------
    # 인쇄실
    pygame.Rect(91, 56, 126, 94),       # 복사기
    pygame.Rect(277, 64, 42, 118),      # 우측 수납장

    # 강사실
    pygame.Rect(374, 113, 85, 64),      # 왼쪽 책상
    pygame.Rect(471, 113, 87, 64),      # 가운데 책상
    pygame.Rect(579, 113, 104, 64),     # 오른쪽 책상

    # 원장실
    pygame.Rect(834, 112, 171, 74),     # 큰 책상
    pygame.Rect(1001, 57, 53, 80),      # 금고/수납장

    # 상담실
    pygame.Rect(66, 463, 90, 72),       # 원형 탁자
    pygame.Rect(177, 475, 111, 73),      # 컴퓨터 책상

    # 교실
    pygame.Rect(839, 474, 205, 116),    # 학생 책상 영역
]


def nearest_interactable(max_dist: float = 82.0) -> Optional[Interactable]:
    center = pygame.Vector2(player.center)
    best = None
    best_dist = float("inf")

    for obj in objects:
        obj_center = pygame.Vector2(obj.rect.center)
        dist = center.distance_to(obj_center)

        if dist < max_dist and dist < best_dist:
            best = obj
            best_dist = dist

    return best


def nearest_door(max_dist: float = 56.0) -> Optional[Door]:
    center = pygame.Vector2(player.center)
    best = None
    best_dist = float("inf")

    for door in doors:
        dist = center.distance_to(pygame.Vector2(door.rect.center))
        if dist < max_dist and dist < best_dist:
            best = door
            best_dist = dist

    return best


def current_room_name() -> str:
    center = player.center

    for room_info in ROOMS.values():
        if room_info["rect"].collidepoint(center):
            return room_info["name"]

    return "복도"


def collides_with_world(rect: pygame.Rect) -> bool:
    for wall in walls:
        if rect.colliderect(wall):
            return True

    for door in doors:
        if not door.open and rect.colliderect(door.rect):
            return True

    return False


def toggle_near_door() -> bool:
    door = nearest_door()

    if door is None:
        return False

    if door.locked:
        state.notify(f"{door.name}은(는) 잠겨 있다.")
        return True

    # 플레이어가 문 내부에 서 있을 때 닫지 못하게 함
    if door.open and player.colliderect(door.rect.inflate(12, 12)):
        state.notify("문 사이에 서 있어 닫을 수 없다.")
        return True

    door.open = not door.open
    state.notify(f"{door.name}: {'열림' if door.open else '닫힘'}")
    return True


# ------------------------------------------------------------
# NPC 응답
# ------------------------------------------------------------
def npc_reply(npc_id: str, message: str, discovered: List[str]) -> str:
    msg = message.lower()
    has = lambda clue_id: clue_id in discovered

    if npc_id == "jiyeon":
        if "10" in msg or "22" in msg or "인쇄실" in msg or "복사" in msg:
            return "네, 22시 30분쯤 인쇄실에 간 건 맞아요. 수업 끝나고 학생들 보충자료를 뽑으러 간 거예요. 시험지는 본 적 없습니다."
        if "분필" in msg or "가루" in msg:
            return "수업 끝나고 바로 갔으니까 손에 분필가루가 묻어 있었을 수는 있어요. 그걸로 제가 시험지를 봤다고 할 수는 없죠."
        if "성과급" in msg or "원장" in msg:
            return "성과급 기준이 불공평하다고 항의한 건 사실이에요. 낮은 반을 맡으면 애초에 불리하니까요. 그래도 시험지를 훔칠 이유는 없어요."
        return "저는 억울합니다. 어제 수업이 끝난 뒤 자료를 뽑고 바로 퇴근했어요."

    if npc_id == "sieun":
        if "금고" in msg or "비밀번호" in msg:
            return "금고 비밀번호는 원장님과 저만 알고 있습니다. 다만 누군가 제 컴퓨터나 원장실 자료를 봤다면 가능성이 아예 없진 않겠네요."
        if "발견" in msg or "아침" in msg:
            return "오늘 8시 30분쯤 모의고사 준비를 하려고 금고를 열었는데, 시험지 봉투가 이미 뜯겨 있었습니다."
        if "지문" in msg or "닦" in msg:
            return "키패드가 닦여 있었다면 누군가 흔적을 의도적으로 지운 거예요. 단순 실수는 아닌 것 같습니다."
        return "시험지는 분명 어제 원장실 금고에 넣었습니다. 보관 담당자로서 저도 당황스럽습니다."

    if npc_id == "juni":
        if "6" in msg or "일찍" in msg or "등원" in msg:
            return "오늘 시험이 너무 불안해서 6시 30분에 왔어요. 정예반에 남으려면 잘 봐야 해서요."
        if "7" in msg or "상담실장" in msg or "찬형" in msg or "봤" in msg:
            return "7시쯤 복도에서 상담실장님을 봤어요. 그냥 상담 준비하러 오신 줄 알았는데, 저를 보고 조금 놀라신 것 같긴 했어요."
        if "시험지" in msg:
            return "보고 싶다는 생각이 아예 없었다면 거짓말이겠지만, 진짜 본 적은 없어요. 내용도 몰라요."
        return "저는 시험장에만 있었어요. 아침에 일찍 온 건 맞지만 시험지를 본 건 아닙니다."

    if npc_id == "chanhyung":
        pressure = sum(
            has(clue_id)
            for clue_id in [
                "account_memo",
                "monitor_chat",
                "copy_log",
                "consult_schedule",
                "coffee_stain",
            ]
        )

        if "계좌" in msg or "메모" in msg:
            if has("account_memo") or has("monitor_chat"):
                return "제 계좌가 적힌 메모요? 상담비 정산 때문에 적어둔 걸 누가 버렸을 수도 있죠. 그걸 시험지 유출과 연결하는 건 억측입니다."
            return "계좌라니요? 무슨 말씀인지 모르겠습니다."

        if "3" in msg or "새벽" in msg or "03" in msg:
            if pressure >= 3:
                return "그 시간에 누가 들어왔는지는 제가 알 수 없습니다. 기록만으로 저라고 단정할 수는 없잖아요."
            return "저는 새벽에 온 적 없습니다. 오늘 아침 상담 때문에 7시쯤 왔습니다."

        if "상담" in msg or "별표" in msg:
            return "8시에 중요한 학부모 상담이 있어서 표시해둔 겁니다. 민원이 많은 상담이라 따로 챙겨야 했어요."

        if pressure >= 4:
            return "처음부터 그럴 생각은 아니었습니다. 하지만 아직 저라고 단정할 증거가 충분한 건 아니잖습니까."

        return "저는 상담 준비 때문에 일찍 왔을 뿐입니다. 시험지는 부원장님 담당 아닌가요?"

    return "잘 모르겠습니다."


def judge_answer(answer: str, discovered: List[str]) -> Dict:
    text = answer.lower()
    score = 0
    correct: List[str] = []
    missing: List[str] = []
    wrong: List[str] = []

    if any(word in text for word in ["이찬형", "찬형", "상담실장"]):
        score += 30
        correct.append("범인을 이찬형 상담실장으로 지목했습니다.")
    else:
        wrong.append("범인 지목이 틀렸거나 불명확합니다. 핵심 범인은 이찬형입니다.")

    if any(word in text for word in ["돈", "계좌", "학부모", "입금", "거래", "뇌물"]):
        score += 20
        correct.append("학부모와의 금전 거래 동기를 파악했습니다.")
    else:
        missing.append("학부모에게 돈을 받고 시험지를 유출했다는 동기 설명이 부족합니다.")

    method_points = [
        ("새벽 3시 학원 출입", ["새벽", "3시", "03"]),
        ("금고 비밀번호 획득", ["비밀번호", "컴퓨터", "로그인"]),
        ("금고에서 시험지 탈취", ["금고", "꺼내", "탈취"]),
        ("03:20 인쇄실 스캔", ["3시 20", "03:20", "스캔", "복사기"]),
        ("시험지를 다시 넣어둠", ["다시", "넣", "되돌", "봉투"]),
    ]

    for label, keywords in method_points:
        if any(keyword in text for keyword in keywords):
            score += 5
            correct.append(label + " 과정을 언급했습니다.")
        else:
            missing.append(label + " 설명이 부족합니다.")

    evidence_points = [
        ("03:20 스캔 기록", 7, ["3시 20", "03:20", "스캔", "복사기"]),
        ("상담실 모니터 계좌 대화", 7, ["모니터", "채팅", "대화", "같은 계좌"]),
        ("인쇄실 쓰레기통 계좌 메모장", 6, ["메모", "쓰레기통", "계좌"]),
        ("준이의 7시 목격·상담 예약표·커피 자국", 5, ["준이", "7시", "목격", "예약표", "별표", "커피"]),
    ]

    for label, points, keywords in evidence_points:
        if any(keyword in text for keyword in keywords):
            score += points
            correct.append(label + "을 근거로 제시했습니다.")
        else:
            missing.append(label + " 근거가 빠졌습니다.")

    if score >= 90:
        grade = "정답"
    elif score >= 70:
        grade = "거의 정답"
    elif score >= 45:
        grade = "부분 정답"
    else:
        grade = "오답에 가까움"

    return {
        "score": min(score, 100),
        "grade": grade,
        "correct": correct[:8],
        "missing": missing[:8],
        "wrong": wrong,
        "feedback": "핵심은 이찬형의 금전 거래, 새벽 3시대 범행, 03:20 스캔 기록, 계좌 증거를 연결하는 것입니다.",
    }


# ------------------------------------------------------------
# 배경 / 맵 렌더링
# ------------------------------------------------------------
def grade_from_percentage(percentage: float) -> str:
    if percentage >= 90:
        return "정답"
    if percentage >= 70:
        return "거의 정답"
    if percentage >= 45:
        return "부분 정답"
    return "오답에 가까움"


def judge_answer(answer: str, discovered: List[str]) -> Dict:
    if evaluate_deduction is None:
        return {
            "score": 0,
            "grade": "채점 불가",
            "correct": [],
            "missing": ["LLM 채점 모듈을 불러오지 못했습니다."],
            "wrong": [],
            "feedback": (
                "back/score/inference.py import에 실패했습니다.\n"
                f"오류: {type(EVALUATOR_IMPORT_ERROR).__name__}: {EVALUATOR_IMPORT_ERROR}"
            ),
        }

    try:
        result = evaluate_deduction(answer, debug=True)
    except Exception as exc:
        return {
            "score": 0,
            "grade": "채점 실패",
            "correct": [],
            "missing": [
                "LLM 채점 호출에 실패했습니다.",
                "OPENAI_API_KEY, openai 패키지 설치, 네트워크 연결을 확인하세요.",
            ],
            "wrong": [],
            "feedback": f"{type(exc).__name__}: {exc}",
        }

    feedback = result.feedback
    if result.details:
        warnings = result.details.get("warnings") or []
        if warnings:
            feedback += "\n\n[채점 경고]\n" + "\n".join(f"- {item}" for item in warnings)

    return {
        "score": round(float(result.percentage), 1),
        "grade": grade_from_percentage(float(result.percentage)),
        "correct": [f"LLM 채점 완료: {result.score:.2f}/{result.max_score:.2f}점"],
        "missing": [],
        "wrong": [],
        "feedback": feedback,
    }


def draw_fallback_map() -> None:
    screen.fill(COLORS["bg"])

    for y in range(20, HEIGHT - 20, TILE):
        for x in range(20, WIDTH - 20, TILE):
            color = (
                COLORS["floor"]
                if (x // TILE + y // TILE) % 2 == 0
                else COLORS["floor2"]
            )
            pygame.draw.rect(screen, color, (x, y, TILE, TILE))
            pygame.draw.rect(screen, (73, 43, 31), (x, y, TILE, TILE), 1)

    for wall in walls:
        pygame.draw.rect(screen, COLORS["wall"], wall)
        pygame.draw.rect(screen, COLORS["line"], wall, 2)


def draw_background_map() -> None:
    if BACKGROUND is not None:
        screen.blit(BACKGROUND, (0, 0))
    else:
        draw_fallback_map()


def draw_doors() -> None:
    """문을 항상 표시한다.

    닫힌 문:
      입구를 가로막는 나무 판으로 표시하고 충돌한다.

    열린 문:
      문짝을 벽 옆으로 90도 회전한 것처럼 표시하고 통과할 수 있다.
    """
    for door in doors:
        frame_color = (119, 66, 39)
        wood_dark = (54, 30, 21)
        wood_mid = (91, 49, 29)
        wood_light = (157, 83, 43)
        knob = (240, 174, 75)

        # 문틀은 열림/닫힘과 상관없이 항상 보이게 한다.
        pygame.draw.rect(
            screen,
            frame_color,
            door.rect.inflate(8, 8),
            3,
            border_radius=3,
        )

        if not door.open:
            # 닫힌 문짝
            pygame.draw.rect(screen, wood_dark, door.rect, border_radius=3)
            inner = door.rect.inflate(-4, -4)
            if inner.width > 0 and inner.height > 0:
                pygame.draw.rect(screen, wood_mid, inner, border_radius=2)
                pygame.draw.rect(screen, wood_light, inner, 1, border_radius=2)

            if door.orientation == "horizontal":
                knob_pos = (door.rect.right - 9, door.rect.centery)
            else:
                knob_pos = (door.rect.centerx, door.rect.bottom - 9)

            pygame.draw.circle(screen, knob, knob_pos, 2)

        else:
            # 열린 문짝: 통로 옆 벽을 따라 보이도록 회전해서 그림
            if door.orientation == "horizontal":
                opened_rect = pygame.Rect(
                    door.rect.left - 2,
                    door.rect.top - door.rect.width + 4,
                    max(8, door.rect.height),
                    door.rect.width,
                )
            else:
                opened_rect = pygame.Rect(
                    door.rect.left - door.rect.height + 4,
                    door.rect.top,
                    door.rect.height,
                    max(8, door.rect.width),
                )

            pygame.draw.rect(screen, wood_dark, opened_rect, border_radius=3)
            inner = opened_rect.inflate(-4, -4)
            if inner.width > 0 and inner.height > 0:
                pygame.draw.rect(screen, wood_mid, inner, border_radius=2)
                pygame.draw.rect(screen, wood_light, inner, 1, border_radius=2)

            pygame.draw.circle(screen, knob, opened_rect.center, 2)

def draw_interactable_highlight() -> None:
    if state.mode != "play":
        return

    near = nearest_interactable()
    if near is not None:
        pulse = 2 + int((math.sin(pygame.time.get_ticks() * 0.008) + 1) * 1.5)
        pygame.draw.rect(
            screen,
            COLORS["light"],
            near.rect.inflate(8, 8),
            pulse,
            border_radius=8,
        )

    door = nearest_door()
    if door is not None:
        # 문 자체는 항상 보이고, 가까이 가면 작은 빛만 추가한다.
        pulse_radius = 4 + int((math.sin(pygame.time.get_ticks() * 0.01) + 1) * 2)
        pygame.draw.circle(
            screen,
            COLORS["light"],
            door.rect.center,
            pulse_radius,
            1,
        )


def draw_npcs() -> None:
    for npc_id, npc in NPCS.items():
        x, y = npc["pos"]
        sprite = MAP_SPRITES[npc_id]
        sprite_rect = sprite.get_rect(midbottom=(x, y + 38))

        pygame.draw.ellipse(
            screen,
            COLORS["shadow"],
            (x - 30, y + 24, 60, 16),
        )
        screen.blit(sprite, sprite_rect)
        draw_text(screen, npc["name"], (x - 30, y + 45), FONT_SM, COLORS["text"])


def draw_player() -> None:
    x, y = player.center
    moving = any(pygame.key.get_pressed()[key] for key in (
        pygame.K_a,
        pygame.K_d,
        pygame.K_w,
        pygame.K_s,
    ))
    bob = int(math.sin(pygame.time.get_ticks() * 0.018) * 2) if moving else 0
    sprite_rect = PLAYER_SPRITE.get_rect(midbottom=(x, y + 29 + bob))

    pygame.draw.ellipse(
        screen,
        COLORS["shadow"],
        (x - 20, y + 12, 40, 12),
    )
    screen.blit(PLAYER_SPRITE, sprite_rect)


def draw_collision_debug() -> None:
    """F2로 켜고 끄는 좌표 보정용 충돌 영역 표시."""
    if not SHOW_COLLISION_DEBUG:
        return

    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    for wall in walls:
        pygame.draw.rect(overlay, (255, 70, 70, 95), wall)
        pygame.draw.rect(overlay, (255, 120, 120, 210), wall, 1)

    for door in doors:
        color = (80, 230, 120, 110) if door.open else (255, 210, 60, 120)
        pygame.draw.rect(overlay, color, door.rect)
        pygame.draw.rect(overlay, (255, 255, 255, 220), door.rect, 1)

    for room in ROOMS.values():
        pygame.draw.rect(overlay, (80, 150, 255, 100), room["rect"], 1)

    screen.blit(overlay, (0, 0))


# ------------------------------------------------------------
# HUD
# ------------------------------------------------------------
def draw_pixel_panel(
    rect: pygame.Rect,
    fill=(35, 25, 21),
    border=None,
    shadow=True,
) -> None:
    border = border or COLORS["line"]

    if shadow:
        pygame.draw.rect(
            screen,
            (8, 6, 5),
            rect.move(5, 5),
            border_radius=8,
        )

    pygame.draw.rect(screen, fill, rect, border_radius=8)
    pygame.draw.rect(screen, border, rect, 2, border_radius=8)


def draw_hud() -> None:
    # 상단 타이틀 바
    pygame.draw.rect(screen, (23, 16, 13), (0, 0, WIDTH, 40))
    pygame.draw.line(screen, COLORS["line"], (0, 39), (WIDTH, 39), 2)

    draw_text(screen, "봉인된 모의고사", (18, 8), FONT, COLORS["light"])
    draw_text(
        screen,
        "WASD 이동  |  Q 상호작용  |  F2 충돌영역  |  F11 전체화면",
        (695, 11),
        FONT_SM,
        COLORS["muted"],
    )

    # 현재 방 자막
    room_name = current_room_name()
    label_width = max(120, FONT_LG.size(room_name)[0] + 38)
    room_rect = pygame.Rect(20, 52, label_width, 48)
    draw_pixel_panel(room_rect, fill=(39, 27, 22))

    draw_text(
        screen,
        room_name,
        (
            room_rect.centerx - FONT_LG.size(room_name)[0] // 2,
            room_rect.y + 9,
        ),
        FONT_LG,
        COLORS["light"],
    )

    # 사건 노트
    note_rect = pygame.Rect(875, 50, 225, 122)
    draw_pixel_panel(note_rect, fill=(34, 24, 20))

    draw_text(
        screen,
        f"사건노트  {len(state.discovered)}",
        (890, 63),
        FONT_SM,
        COLORS["light"],
    )

    y = 88
    for clue_id in state.discovered[-4:]:
        draw_text(
            screen,
            "• " + CLUES[clue_id]["name"],
            (890, y),
            FONT_SM,
            COLORS["text"],
            max_width=195,
            line_gap=2,
        )
        y += 20

    if state.mode != "play":
        return

    near_door = nearest_door()
    near_object = nearest_interactable()

    if near_door is not None:
        action = "문 닫기" if near_door.open else "문 열기"
        message = f"Q  {action} · {near_door.name}"

    elif near_object is not None:
        if near_object.kind == "npc":
            action = "대화하기"
        elif near_object.kind == "judge":
            action = "사건 추론하기"
        else:
            action = "확인하기"

        message = f"Q  {action} · {near_object.name}"

    else:
        message = ""

    if message:
        width = FONT.size(message)[0] + 44
        action_rect = pygame.Rect(
            (WIDTH - width) // 2,
            HEIGHT - 68,
            width,
            42,
        )
        draw_pixel_panel(action_rect, fill=(40, 28, 23))
        draw_text(
            screen,
            message,
            (action_rect.x + 22, action_rect.y + 10),
            FONT,
            COLORS["light"],
        )

    if (
        state.notification
        and pygame.time.get_ticks() < state.notification_until
    ):
        width = FONT_SM.size(state.notification)[0] + 34
        notify_rect = pygame.Rect(
            (WIDTH - width) // 2,
            52,
            width,
            34,
        )
        draw_pixel_panel(notify_rect, fill=(48, 31, 24))
        draw_text(
            screen,
            state.notification,
            (notify_rect.x + 17, notify_rect.y + 8),
            FONT_SM,
            COLORS["text"],
        )


# ------------------------------------------------------------
# 모달 창
# ------------------------------------------------------------
def draw_panel(rect: pygame.Rect, title: Optional[str] = None) -> None:
    pygame.draw.rect(screen, (0, 0, 0), rect.move(6, 6), border_radius=14)
    pygame.draw.rect(screen, COLORS["panel"], rect, border_radius=14)
    pygame.draw.rect(screen, COLORS["line"], rect, 3, border_radius=14)

    if title:
        draw_text(
            screen,
            title,
            (rect.x + 22, rect.y + 18),
            FONT_LG,
            COLORS["light"],
        )


def draw_start_screen() -> None:
    if BACKGROUND:
        screen.blit(BACKGROUND, (0, 0))
    else:
        screen.fill(COLORS["bg"])

    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((16, 10, 8, 172))
    screen.blit(overlay, (0, 0))

    title = "봉인된 모의고사"
    subtitle = "사라진 시험지의 진실을 찾는 2D 추리 게임"

    draw_text(
        screen,
        title,
        ((WIDTH - FONT_XL.size(title)[0]) // 2, 90),
        FONT_XL,
        COLORS["light"],
    )
    draw_text(
        screen,
        subtitle,
        ((WIDTH - FONT.size(subtitle)[0]) // 2, 146),
        FONT,
        COLORS["text"],
    )

    panel_rect = pygame.Rect(190, 205, 740, 300)
    draw_pixel_panel(panel_rect, fill=(36, 24, 20))

    y = panel_rect.y + 28
    draw_text(screen, "어떤 게임인가요?", (230, y), FONT_LG, COLORS["light"])
    y += 44
    y = draw_text(
        screen,
        "학원에서 봉인되어 있던 모의고사 시험지가 사라졌습니다. "
        "방을 돌아다니며 단서를 조사하고, NPC에게 질문해서 범인과 동기, 방법을 추리하세요.",
        (230, y),
        FONT,
        COLORS["text"],
        max_width=660,
    )

    y += 30
    draw_text(screen, "조작 방법", (230, y), FONT_LG, COLORS["light"])
    y += 42

    controls = [
        "W/A/S/D : 이동",
        "Q : 문 열기, 단서 확인, NPC와 대화",
        "Enter : 질문 입력 또는 최종 추리 제출",
        "ESC : 창 닫기 / 게임 종료",
    ]

    for item in controls:
        draw_text(screen, item, (250, y), FONT, COLORS["text"])
        y += 30

    mouse_pos = pygame.mouse.get_pos()
    hovering = START_BUTTON_RECT.collidepoint(mouse_pos)
    button_fill = (92, 54, 34) if hovering else (70, 42, 30)
    pygame.draw.rect(
        screen,
        (8, 6, 5),
        START_BUTTON_RECT.move(5, 5),
        border_radius=10,
    )
    pygame.draw.rect(screen, button_fill, START_BUTTON_RECT, border_radius=10)
    pygame.draw.rect(screen, COLORS["light"], START_BUTTON_RECT, 2, border_radius=10)

    button_text = "START"
    draw_text(
        screen,
        button_text,
        (
            START_BUTTON_RECT.centerx - FONT_LG.size(button_text)[0] // 2,
            START_BUTTON_RECT.y + 13,
        ),
        FONT_LG,
        COLORS["light"],
    )

    hint = "Enter 또는 Space로도 시작할 수 있어요"
    draw_text(
        screen,
        hint,
        ((WIDTH - FONT_SM.size(hint)[0]) // 2, 622),
        FONT_SM,
        COLORS["muted"],
    )


def draw_start_screen() -> None:
    if BACKGROUND:
        screen.blit(BACKGROUND, (0, 0))
    else:
        screen.fill(COLORS["bg"])

    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((16, 10, 8, 172))
    screen.blit(overlay, (0, 0))

    title = "봉인된 모의고사"
    subtitle = "사라진 시험지의 진실을 밝히는 2D 추리 게임"

    draw_text(
        screen,
        title,
        ((WIDTH - FONT_XL.size(title)[0]) // 2, 42),
        FONT_XL,
        COLORS["light"],
    )
    draw_text(
        screen,
        subtitle,
        ((WIDTH - FONT.size(subtitle)[0]) // 2, 96),
        FONT,
        COLORS["text"],
    )

    panel_rect = pygame.Rect(170, 132, 780, 465)
    draw_pixel_panel(panel_rect, fill=(36, 24, 20))

    y = panel_rect.y + 24
    intro_lines = [
        "모의고사 시작 30분 전,",
        "원장실 금고에 보관되어 있던 시험지 봉투가 개봉된 채 발견되었다.",
        "",
        "시험지에 접근할 수 있었던 사람은 학원 안에 있던 네 명.",
        "그리고 그중 한 명이 시험지를 유출한 범인이다.",
        "",
        "플레이어는 학원 곳곳을 돌아다니며 용의자들과 대화하고, 숨겨진 단서와 서로 맞지 않는 진술을 조사해야 한다. 복사기 기록, 등원 기록, 수상한 메모와 대화 내역을 바탕으로 사건의 타임라인을 완성하고 진범을 밝혀내자.",
        "",
        "과연 시험지를 유출한 사람은 누구이며,",
        "그날 새벽 학원에서는 무슨 일이 있었을까?",
    ]

    for line in intro_lines:
        if line:
            y = draw_text(
                screen,
                line,
                (210, y),
                FONT_SM,
                COLORS["text"],
                max_width=700,
                line_gap=3,
            )
        else:
            y += 12

    y += 12
    draw_text(screen, "게임 방법", (210, y), FONT_LG, COLORS["light"])
    y += 38

    guide_lines = [
        "학원 내부를 탐색하여 사건과 관련된 단서를 수집하세요.",
        "네 명의 용의자와 대화하며 진술을 확인하세요.",
        "단서와 진술의 모순을 비교하여 사건의 타임라인을 추리하세요.",
        "충분한 증거를 모은 뒤 최종 범인을 지목하세요.",
        "",
        "단 한 번의 추리가 사건의 결말을 결정합니다.",
    ]

    for line in guide_lines:
        if line:
            y = draw_text(
                screen,
                line,
                (230, y),
                FONT_SM,
                COLORS["text"] if "단 한 번" not in line else COLORS["light"],
                max_width=670,
                line_gap=3,
            )
        else:
            y += 10

    mouse_pos = pygame.mouse.get_pos()
    hovering = START_BUTTON_RECT.collidepoint(mouse_pos)
    button_fill = (92, 54, 34) if hovering else (70, 42, 30)
    pygame.draw.rect(
        screen,
        (8, 6, 5),
        START_BUTTON_RECT.move(5, 5),
        border_radius=10,
    )
    pygame.draw.rect(screen, button_fill, START_BUTTON_RECT, border_radius=10)
    pygame.draw.rect(screen, COLORS["light"], START_BUTTON_RECT, 2, border_radius=10)

    button_text = "START"
    draw_text(
        screen,
        button_text,
        (
            START_BUTTON_RECT.centerx - FONT_LG.size(button_text)[0] // 2,
            START_BUTTON_RECT.y + 13,
        ),
        FONT_LG,
        COLORS["light"],
    )

    hint = "Enter 또는 Space로도 시작할 수 있어요"
    draw_text(
        screen,
        hint,
        ((WIDTH - FONT_SM.size(hint)[0]) // 2, 622),
        FONT_SM,
        COLORS["muted"],
    )


def draw_start_screen() -> None:
    if BACKGROUND:
        screen.blit(BACKGROUND, (0, 0))
    else:
        screen.fill(COLORS["bg"])

    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((16, 10, 8, 174))
    screen.blit(overlay, (0, 0))

    title = "봉인된 모의고사"
    subtitle = "사라진 시험지의 진실을 밝히는 2D 추리 게임"

    draw_text(
        screen,
        title,
        ((WIDTH - FONT_XL.size(title)[0]) // 2, 42),
        FONT_XL,
        COLORS["light"],
    )
    draw_text(
        screen,
        subtitle,
        ((WIDTH - FONT.size(subtitle)[0]) // 2, 94),
        FONT,
        COLORS["text"],
    )

    panel_rect = pygame.Rect(95, 128, 930, 390)
    draw_pixel_panel(panel_rect, fill=(36, 24, 20))

    left_x = panel_rect.x + 34
    right_x = panel_rect.x + 600
    y = panel_rect.y + 24

    story_lines = [
        "모의고사 시작 30분 전,",
        "원장실 금고에 보관되어 있던 시험지 봉투가 개봉된 채 발견되었다.",
        "",
        "시험지에 접근할 수 있었던 사람은 학원 안에 있던 네 명.",
        "그리고 그중 한 명이 시험지를 유출한 범인이다.",
        "",
        "플레이어는 학원 곳곳을 돌아다니며 용의자들과 대화하고, 숨겨진 단서와 서로 맞지 않는 진술을 조사해야 한다. 복사기 기록, 등원 기록, 수상한 메모와 대화 내역을 바탕으로 사건의 타임라인을 완성하고 진범을 밝혀내자.",
        "",
        "과연 시험지를 유출한 사람은 누구이며,",
        "그날 새벽 학원에서는 무슨 일이 있었을까?",
    ]

    for line in story_lines:
        if line:
            y = draw_text(
                screen,
                line,
                (left_x, y),
                FONT_SM,
                COLORS["text"],
                max_width=520,
                line_gap=2,
            )
        else:
            y += 10

    y = panel_rect.y + 24
    draw_text(screen, "게임 방법", (right_x, y), FONT_LG, COLORS["light"])
    y += 40

    guide_lines = [
        "학원 내부를 탐색하여 사건과 관련된 단서를 수집하세요.",
        "네 명의 용의자와 대화하며 진술을 확인하세요.",
        "단서와 진술의 모순을 비교하여 사건의 타임라인을 추리하세요.",
        "충분한 증거를 모은 뒤 최종 범인을 지목하세요.",
        "",
        "단 한 번의 추리가 사건의 결말을 결정합니다.",
    ]

    for line in guide_lines:
        if line:
            y = draw_text(
                screen,
                line,
                (right_x, y),
                FONT_SM,
                COLORS["light"] if "단 한 번" in line else COLORS["text"],
                max_width=280,
                line_gap=2,
            )
        else:
            y += 12

    controls_title_y = panel_rect.y + 286
    draw_text(screen, "조작", (right_x, controls_title_y), FONT, COLORS["light"])
    controls = [
        "W/A/S/D : 이동",
        "Q : 조사 / 대화 / 문 열기",
        "Enter : 입력 / 추리 제출",
        "ESC : 창 닫기",
    ]

    y = controls_title_y + 30
    for line in controls:
        draw_text(screen, line, (right_x, y), FONT_SM, COLORS["muted"])
        y += 22

    mouse_pos = pygame.mouse.get_pos()
    hovering = START_BUTTON_RECT.collidepoint(mouse_pos)
    button_fill = (92, 54, 34) if hovering else (70, 42, 30)
    pygame.draw.rect(
        screen,
        (8, 6, 5),
        START_BUTTON_RECT.move(5, 5),
        border_radius=10,
    )
    pygame.draw.rect(screen, button_fill, START_BUTTON_RECT, border_radius=10)
    pygame.draw.rect(screen, COLORS["light"], START_BUTTON_RECT, 2, border_radius=10)

    button_text = "START"
    draw_text(
        screen,
        button_text,
        (
            START_BUTTON_RECT.centerx - FONT_LG.size(button_text)[0] // 2,
            START_BUTTON_RECT.y + 13,
        ),
        FONT_LG,
        COLORS["light"],
    )

    hint = "Enter 또는 Space로도 시작할 수 있어요"
    draw_text(
        screen,
        hint,
        ((WIDTH - FONT_SM.size(hint)[0]) // 2, 622),
        FONT_SM,
        COLORS["muted"],
    )


def draw_start_screen() -> None:
    if BACKGROUND:
        screen.blit(BACKGROUND, (0, 0))
    else:
        screen.fill(COLORS["bg"])

    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((16, 10, 8, 174))
    screen.blit(overlay, (0, 0))

    title = "봉인된 모의고사"
    subtitle = "사라진 시험지의 진실을 밝히는 2D 추리 게임"

    draw_text(
        screen,
        title,
        ((WIDTH - FONT_XL.size(title)[0]) // 2, 36),
        FONT_XL,
        COLORS["light"],
    )
    draw_text(
        screen,
        subtitle,
        ((WIDTH - FONT.size(subtitle)[0]) // 2, 86),
        FONT,
        COLORS["text"],
    )

    panel_rect = pygame.Rect(105, 116, 910, 410)
    draw_pixel_panel(panel_rect, fill=(36, 24, 20))

    x = panel_rect.x + 34
    y = panel_rect.y + 22

    story_lines = [
        "모의고사 시작 30분 전,",
        "원장실 금고에 보관되어 있던 시험지 봉투가 개봉된 채 발견되었다.",
        "",
        "시험지에 접근할 수 있었던 사람은 학원 안에 있던 네 명.",
        "그리고 그중 한 명이 시험지를 유출한 범인이다.",
        "",
        "플레이어는 학원 곳곳을 돌아다니며 용의자들과 대화하고, 숨겨진 단서와 서로 맞지 않는 진술을 조사해야 한다. 복사기 기록, 등원 기록, 수상한 메모와 대화 내역을 바탕으로 사건의 타임라인을 완성하고 진범을 밝혀내자.",
        "",
        "과연 시험지를 유출한 사람은 누구이며,",
        "그날 새벽 학원에서는 무슨 일이 있었을까?",
    ]

    for line in story_lines:
        if line:
            y = draw_text(
                screen,
                line,
                (x, y),
                FONT_SM,
                COLORS["text"],
                max_width=840,
                line_gap=2,
            )
        else:
            y += 9

    y += 10
    draw_text(screen, "게임 방법", (x, y), FONT, COLORS["light"])
    y += 28

    guide_lines = [
        "학원 내부를 탐색하여 사건과 관련된 단서를 수집하세요.",
        "네 명의 용의자와 대화하며 진술을 확인하세요.",
        "단서와 진술의 모순을 비교하여 사건의 타임라인을 추리하세요.",
        "충분한 증거를 모은 뒤 최종 범인을 지목하세요.",
        "단 한 번의 추리가 사건의 결말을 결정합니다.",
    ]

    for line in guide_lines:
        y = draw_text(
            screen,
            line,
            (x + 18, y),
            FONT_SM,
            COLORS["light"] if "단 한 번" in line else COLORS["text"],
            max_width=805,
            line_gap=2,
        )

    y += 8
    draw_text(screen, "조작 방법", (x, y), FONT, COLORS["light"])
    y += 28

    controls = [
        "W/A/S/D : 이동",
        "Q : 조사 / 대화 / 문 열기",
        "Enter : 입력 / 최종 추리 제출",
        "ESC : 창 닫기 / 게임 종료",
    ]

    control_x = x + 18
    for index, line in enumerate(controls):
        col = index % 2
        row = index // 2
        draw_text(
            screen,
            line,
            (control_x + col * 360, y + row * 23),
            FONT_SM,
            COLORS["muted"],
        )

    mouse_pos = pygame.mouse.get_pos()
    hovering = START_BUTTON_RECT.collidepoint(mouse_pos)
    button_fill = (92, 54, 34) if hovering else (70, 42, 30)
    pygame.draw.rect(
        screen,
        (8, 6, 5),
        START_BUTTON_RECT.move(5, 5),
        border_radius=10,
    )
    pygame.draw.rect(screen, button_fill, START_BUTTON_RECT, border_radius=10)
    pygame.draw.rect(screen, COLORS["light"], START_BUTTON_RECT, 2, border_radius=10)

    button_text = "START"
    draw_text(
        screen,
        button_text,
        (
            START_BUTTON_RECT.centerx - FONT_LG.size(button_text)[0] // 2,
            START_BUTTON_RECT.y + 13,
        ),
        FONT_LG,
        COLORS["light"],
    )

    hint = "Enter 또는 Space로도 시작할 수 있어요"
    draw_text(
        screen,
        hint,
        ((WIDTH - FONT_SM.size(hint)[0]) // 2, 622),
        FONT_SM,
        COLORS["muted"],
    )


def draw_start_screen() -> None:
    if BACKGROUND:
        screen.blit(BACKGROUND, (0, 0))
    else:
        screen.fill(COLORS["bg"])

    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((16, 10, 8, 174))
    screen.blit(overlay, (0, 0))

    title = "봉인된 모의고사"
    subtitle = "사라진 시험지의 진실을 밝히는 2D 추리 게임"

    draw_text(
        screen,
        title,
        ((WIDTH - FONT_XL.size(title)[0]) // 2, 32),
        FONT_XL,
        COLORS["light"],
    )
    draw_text(
        screen,
        subtitle,
        ((WIDTH - FONT.size(subtitle)[0]) // 2, 82),
        FONT,
        COLORS["text"],
    )

    panel_rect = pygame.Rect(95, 114, 930, 480)
    draw_pixel_panel(panel_rect, fill=(36, 24, 20))

    x = panel_rect.x + 34
    y = panel_rect.y + 22

    story_lines = [
        "모의고사 시작 30분 전,",
        "원장실 금고에 보관되어 있던 시험지 봉투가 개봉된 채 발견되었다.",
        "",
        "시험지에 접근할 수 있었던 사람은 학원 안에 있던 네 명.",
        "그리고 그중 한 명이 시험지를 유출한 범인이다.",
        "",
        "플레이어는 학원 곳곳을 돌아다니며 용의자들과 대화하고, 숨겨진 단서와 서로 맞지 않는 진술을 조사해야 한다. 복사기 기록, 등원 기록, 수상한 메모와 대화 내역을 바탕으로 사건의 타임라인을 완성하고 진범을 밝혀내자.",
        "",
        "과연 시험지를 유출한 사람은 누구이며,",
        "그날 새벽 학원에서는 무슨 일이 있었을까?",
    ]

    for line in story_lines:
        if line:
            y = draw_text(
                screen,
                line,
                (x, y),
                FONT_SM,
                COLORS["text"],
                max_width=840,
                line_gap=2,
            )
        else:
            y += 9

    y += 10
    draw_text(screen, "게임 방법", (x, y), FONT, COLORS["light"])
    y += 27

    guide_lines = [
        "학원 내부를 탐색하여 사건과 관련된 단서를 수집하세요.",
        "네 명의 용의자와 대화하며 진술을 확인하세요.",
        "단서와 진술의 모순을 비교하여 사건의 타임라인을 추리하세요.",
        "충분한 증거를 모은 뒤 최종 범인을 지목하세요.",
        "단 한 번의 추리가 사건의 결말을 결정합니다.",
    ]

    for line in guide_lines:
        y = draw_text(
            screen,
            line,
            (x + 18, y),
            FONT_SM,
            COLORS["light"] if "단 한 번" in line else COLORS["text"],
            max_width=805,
            line_gap=2,
        )

    y += 8
    draw_text(screen, "조작 방법", (x, y), FONT_SM, COLORS["light"])
    y += 24

    controls = [
        "W/A/S/D : 이동",
        "Q : 조사 / 대화 / 문 열기",
        "Enter : 입력 / 최종 추리 제출",
        "ESC : 창 닫기 / 게임 종료",
    ]

    control_x = x + 18
    for index, line in enumerate(controls):
        col = index % 2
        row = index // 2
        draw_text(
            screen,
            line,
            (control_x + col * 360, y + row * 20),
            FONT_XS,
            COLORS["muted"],
        )

    mouse_pos = pygame.mouse.get_pos()
    hovering = START_BUTTON_RECT.collidepoint(mouse_pos)
    button_fill = (92, 54, 34) if hovering else (70, 42, 30)
    pygame.draw.rect(
        screen,
        (8, 6, 5),
        START_BUTTON_RECT.move(5, 5),
        border_radius=10,
    )
    pygame.draw.rect(screen, button_fill, START_BUTTON_RECT, border_radius=10)
    pygame.draw.rect(screen, COLORS["light"], START_BUTTON_RECT, 2, border_radius=10)

    button_text = "START"
    draw_text(
        screen,
        button_text,
        (
            START_BUTTON_RECT.centerx - FONT_LG.size(button_text)[0] // 2,
            START_BUTTON_RECT.y + 11,
        ),
        FONT_LG,
        COLORS["light"],
    )

    hint = "Enter 또는 Space로도 시작할 수 있어요"
    draw_text(
        screen,
        hint,
        ((WIDTH - FONT_SM.size(hint)[0]) // 2, 678),
        FONT_SM,
        COLORS["muted"],
    )


def draw_clue_window() -> None:
    rect = pygame.Rect(120, 100, 880, 500)
    draw_panel(rect, "단서 확인")

    obj = next(
        (item for item in objects if item.id == state.active_clue),
        None,
    )

    if obj is None:
        return

    draw_text(
        screen,
        f"조사 대상: {obj.name}",
        (150, 155),
        FONT,
        COLORS["text"],
    )

    y = 200

    for clue_id in obj.clue_ids:
        clue = CLUES[clue_id]
        state.add_clue(clue_id)

        clue_rect = pygame.Rect(150, y, 820, 92)
        pygame.draw.rect(
            screen,
            COLORS["panel2"],
            clue_rect,
            border_radius=10,
        )
        pygame.draw.rect(
            screen,
            COLORS["line"],
            clue_rect,
            1,
            border_radius=10,
        )

        draw_text(
            screen,
            "• " + clue["name"],
            (170, y + 12),
            FONT,
            COLORS["light"],
        )
        draw_text(
            screen,
            clue["desc"],
            (170, y + 43),
            FONT_SM,
            COLORS["text"],
            max_width=770,
        )

        y += 108

    draw_text(
        screen,
        "ESC  돌아가기",
        (800, 555),
        FONT,
        COLORS["muted"],
    )


def draw_dialogue_window() -> None:
    if state.active_npc is None:
        return

    npc = NPCS[state.active_npc]
    rect = pygame.Rect(80, 420, 960, 260)
    draw_panel(rect)

    screen.blit(PORTRAITS[state.active_npc], (105, 448))

    draw_text(
        screen,
        f"{npc['name']}  |  {npc['role']}",
        (310, 445),
        FONT_LG,
        COLORS["light"],
    )

    reply = state.last_reply or npc["intro"]
    draw_text(
        screen,
        reply,
        (310, 488),
        FONT,
        COLORS["text"],
        max_width=690,
    )

    input_rect = pygame.Rect(310, 610, 680, 42)
    pygame.draw.rect(
        screen,
        (25, 18, 16),
        input_rect,
        border_radius=8,
    )
    pygame.draw.rect(
        screen,
        COLORS["line"],
        input_rect,
        2,
        border_radius=8,
    )

    placeholder = "질문을 입력하고 Enter..."
    draw_text(
        screen,
        visible_input_text(placeholder),
        (324, 620),
        FONT_SM,
        COLORS["text"] if state.input_text or state.composing_text else COLORS["muted"],
    )

    draw_text(
        screen,
        "ESC  돌아가기",
        (860, 390),
        FONT_SM,
        COLORS["muted"],
    )


def draw_judge_window() -> None:
    rect = pygame.Rect(90, 80, 940, 560)
    draw_panel(rect, "사건 추론하기")

    draw_text(
        screen,
        "범인, 동기, 방법, 근거를 한 문장으로 작성한 뒤 Enter를 누르세요.",
        (125, 135),
        FONT,
        COLORS["text"],
    )

    input_rect = pygame.Rect(125, 180, 870, 170)
    pygame.draw.rect(
        screen,
        (25, 18, 16),
        input_rect,
        border_radius=10,
    )
    pygame.draw.rect(
        screen,
        COLORS["line"],
        input_rect,
        2,
        border_radius=10,
    )

    placeholder = "예: 범인은 ○○이고, 새벽에 ... 근거는 ..."
    draw_text(
        screen,
        visible_input_text(placeholder),
        (145, 205),
        FONT,
        COLORS["text"] if state.input_text or state.composing_text else COLORS["muted"],
        max_width=830,
    )

    draw_text(
        screen,
        "현재 발견한 단서",
        (125, 380),
        FONT,
        COLORS["light"],
    )

    y = 415
    for clue_id in state.discovered[-8:]:
        draw_text(
            screen,
            "• " + CLUES[clue_id]["name"],
            (145, y),
            FONT_SM,
            COLORS["text"],
        )
        y += 22

    draw_text(
        screen,
        "ESC  돌아가기",
        (845, 590),
        FONT,
        COLORS["muted"],
    )


def draw_result_window() -> None:
    rect = pygame.Rect(90, 50, 940, 620)
    draw_panel(rect, "추리 판정 결과")

    result = state.judge_result
    if result is None:
        return

    draw_text(
        screen,
        f"점수: {result['score']}점  |  판정: {result['grade']}",
        (125, 115),
        FONT_LG,
        COLORS["light"],
    )

    y = 165

    draw_text(screen, "맞힌 부분", (125, y), FONT, COLORS["green"])
    y += 34

    for item in result["correct"]:
        y = draw_text(
            screen,
            "• " + item,
            (145, y),
            FONT_SM,
            COLORS["text"],
            max_width=820,
            line_gap=3,
        )

    y += 12
    draw_text(screen, "부족한 부분", (125, y), FONT, COLORS["danger"])
    y += 34

    for item in result["missing"]:
        y = draw_text(
            screen,
            "• " + item,
            (145, y),
            FONT_SM,
            COLORS["text"],
            max_width=820,
            line_gap=3,
        )

    y += 12
    draw_text(screen, "피드백", (125, y), FONT, COLORS["light"])
    y += 34

    draw_text(
        screen,
        result["feedback"],
        (145, y),
        FONT,
        COLORS["text"],
        max_width=820,
    )

    draw_text(
        screen,
        "ESC  돌아가기",
        (835, 625),
        FONT,
        COLORS["muted"],
    )


# ------------------------------------------------------------
# 업데이트 / 입력
# ------------------------------------------------------------
def draw_result_window() -> None:
    rect = pygame.Rect(90, 50, 940, 620)
    draw_panel(rect, "추리 결과")

    result = state.judge_result
    if result is None:
        return

    draw_text(
        screen,
        f"점수: {result['score']}점 | 판정: {result['grade']}",
        (125, 115),
        FONT_LG,
        COLORS["light"],
    )

    y = 165
    draw_text(screen, "맞은 부분", (125, y), FONT, COLORS["green"])
    y += 34

    for item in result["correct"]:
        y = draw_text(
            screen,
            "- " + item,
            (145, y),
            FONT_SM,
            COLORS["text"],
            max_width=820,
            line_gap=3,
        )

    y += 12
    draw_text(screen, "부족한 부분", (125, y), FONT, COLORS["danger"])
    y += 34

    for item in result["missing"]:
        y = draw_text(
            screen,
            "- " + item,
            (145, y),
            FONT_SM,
            COLORS["text"],
            max_width=820,
            line_gap=3,
        )

    y += 12
    draw_text(screen, "피드백", (125, y), FONT, COLORS["light"])
    y += 34

    draw_text(
        screen,
        result["feedback"],
        (145, y),
        FONT,
        COLORS["text"],
        max_width=820,
    )

    mouse_pos = pygame.mouse.get_pos()
    hovering = RETRY_BUTTON_RECT.collidepoint(mouse_pos)
    button_fill = (92, 54, 34) if hovering else (70, 42, 30)
    pygame.draw.rect(
        screen,
        (8, 6, 5),
        RETRY_BUTTON_RECT.move(5, 5),
        border_radius=10,
    )
    pygame.draw.rect(screen, button_fill, RETRY_BUTTON_RECT, border_radius=10)
    pygame.draw.rect(screen, COLORS["light"], RETRY_BUTTON_RECT, 2, border_radius=10)

    button_text = "RETRY"
    draw_text(
        screen,
        button_text,
        (
            RETRY_BUTTON_RECT.centerx - FONT_LG.size(button_text)[0] // 2,
            RETRY_BUTTON_RECT.y + 9,
        ),
        FONT_LG,
        COLORS["light"],
    )

    draw_text(screen, "ESC: 돌아가기", (805, 625), FONT_SM, COLORS["muted"])


def compact_feedback_text(text: str) -> str:
    lines = []
    previous_blank = False

    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            if not previous_blank and lines:
                lines.append("")
            previous_blank = True
            continue
        lines.append(line)
        previous_blank = False

    return "\n".join(lines).strip()


def draw_text_limited(
    surface: pygame.Surface,
    text: str,
    pos,
    font,
    color,
    max_width: int,
    max_height: int,
    line_gap: int = 2,
) -> int:
    x, y = pos
    bottom = y + max_height
    lines: List[str] = []

    for paragraph in compact_feedback_text(text).splitlines() or [""]:
        current = ""

        for ch in paragraph:
            test = current + ch
            if font.size(test)[0] <= max_width or not current:
                current = test
            else:
                lines.append(current)
                current = ch

        if current:
            lines.append(current)
        elif not paragraph:
            lines.append("")

    for index, line in enumerate(lines):
        next_y = y + font.get_height()
        if next_y > bottom:
            surface.blit(font.render("...", True, COLORS["muted"]), (x, y))
            return y + font.get_height()

        if line:
            surface.blit(font.render(line, True, color), (x, y))
        y += font.get_height() + line_gap

    return y


def wrapped_lines(text: str, font, max_width: int) -> List[str]:
    lines: List[str] = []

    for paragraph in compact_feedback_text(text).splitlines() or [""]:
        current = ""

        for ch in paragraph:
            test = current + ch
            if font.size(test)[0] <= max_width or not current:
                current = test
            else:
                lines.append(current)
                current = ch

        if current:
            lines.append(current)
        elif not paragraph:
            lines.append("")

    return lines


def max_result_scroll() -> int:
    if not state.judge_result:
        return 0

    line_count = len(wrapped_lines(state.judge_result["feedback"], FONT_XS, 874))
    visible_lines = 24
    return max(0, line_count - visible_lines)


def scroll_result(delta: int) -> None:
    state.result_scroll = max(
        0,
        min(max_result_scroll(), state.result_scroll + delta),
    )


def draw_scrollable_text(
    surface: pygame.Surface,
    text: str,
    rect: pygame.Rect,
    font,
    color,
    line_gap: int = 1,
) -> None:
    lines = wrapped_lines(text, font, rect.width)
    line_height = font.get_height() + line_gap
    visible_count = max(1, rect.height // line_height)
    max_scroll = max(0, len(lines) - visible_count)
    state.result_scroll = max(0, min(max_scroll, state.result_scroll))

    clip = surface.get_clip()
    surface.set_clip(rect)

    y = rect.y
    for line in lines[state.result_scroll:state.result_scroll + visible_count]:
        if line:
            surface.blit(font.render(line, True, color), (rect.x, y))
        y += line_height

    surface.set_clip(clip)

    if max_scroll > 0:
        track = pygame.Rect(rect.right + 8, rect.y, 6, rect.height)
        pygame.draw.rect(surface, (60, 43, 35), track, border_radius=3)

        thumb_height = max(24, int(rect.height * visible_count / len(lines)))
        thumb_y = rect.y + int((rect.height - thumb_height) * state.result_scroll / max_scroll)
        thumb = pygame.Rect(track.x, thumb_y, track.width, thumb_height)
        pygame.draw.rect(surface, COLORS["light"], thumb, border_radius=3)



def draw_result_window() -> None:
    rect = pygame.Rect(70, 34, 980, 650)
    draw_panel(rect, "추리 결과")

    result = state.judge_result
    if result is None:
        return

    draw_text(
        screen,
        f"점수: {result['score']}점 | 판정: {result['grade']}",
        (105, 92),
        FONT,
        COLORS["light"],
    )

    feedback_rect = pygame.Rect(105, 132, 910, 430)
    pygame.draw.rect(screen, (25, 18, 16), feedback_rect, border_radius=8)
    pygame.draw.rect(screen, COLORS["line"], feedback_rect, 1, border_radius=8)

    draw_text(screen, "피드백", (feedback_rect.x + 18, feedback_rect.y + 14), FONT_SM, COLORS["light"])
    draw_scrollable_text(
        screen,
        result["feedback"],
        pygame.Rect(
            feedback_rect.x + 18,
            feedback_rect.y + 44,
            feedback_rect.width - 54,
            feedback_rect.height - 70,
        ),
        FONT_XS,
        COLORS["text"],
        line_gap=1,
    )

    mouse_pos = pygame.mouse.get_pos()
    hovering = RETRY_BUTTON_RECT.collidepoint(mouse_pos)
    button_fill = (92, 54, 34) if hovering else (70, 42, 30)
    pygame.draw.rect(
        screen,
        (8, 6, 5),
        RETRY_BUTTON_RECT.move(5, 5),
        border_radius=10,
    )
    pygame.draw.rect(screen, button_fill, RETRY_BUTTON_RECT, border_radius=10)
    pygame.draw.rect(screen, COLORS["light"], RETRY_BUTTON_RECT, 2, border_radius=10)

    button_text = "RETRY"
    draw_text(
        screen,
        button_text,
        (
            RETRY_BUTTON_RECT.centerx - FONT_LG.size(button_text)[0] // 2,
            RETRY_BUTTON_RECT.y + 9,
        ),
        FONT_LG,
        COLORS["light"],
    )

    draw_text(screen, "ESC: 돌아가기", (806, 632), FONT_XS, COLORS["muted"])


def draw_judging_window() -> None:
    rect = pygame.Rect(330, 245, 460, 190)
    draw_panel(rect, "사건 추론 중")

    ticks = pygame.time.get_ticks()
    dots = "." * ((ticks // 450) % 4)
    message = "LLM이 추리를 채점하고 있습니다" + dots

    draw_text(
        screen,
        message,
        (
            rect.centerx - FONT.size(message)[0] // 2,
            rect.y + 72,
        ),
        FONT,
        COLORS["text"],
    )

    sub = "잠시만 기다려 주세요"
    draw_text(
        screen,
        sub,
        (
            rect.centerx - FONT_SM.size(sub)[0] // 2,
            rect.y + 112,
        ),
        FONT_SM,
        COLORS["muted"],
    )


def move_player(keys) -> None:
    dx = 0.0
    dy = 0.0

    if keys[pygame.K_a]:
        dx -= player_speed
    if keys[pygame.K_d]:
        dx += player_speed
    if keys[pygame.K_w]:
        dy -= player_speed
    if keys[pygame.K_s]:
        dy += player_speed

    if dx and dy:
        dx *= 0.707
        dy *= 0.707

    old = player.copy()
    player.x += int(dx)

    if collides_with_world(player):
        player.x = old.x

    old = player.copy()
    player.y += int(dy)

    if collides_with_world(player):
        player.y = old.y


def handle_q() -> None:
    if toggle_near_door():
        return

    near = nearest_interactable()

    if near is None:
        return

    if near.kind == "clue":
        state.active_clue = near.id
        state.mode = "clue"

    elif near.kind == "npc":
        state.active_npc = near.npc_id
        state.input_text = ""
        state.composing_text = ""
        state.last_reply = NPCS[near.npc_id]["intro"]
        state.mode = "dialogue"

    elif near.kind == "judge":
        state.input_text = ""
        state.composing_text = ""
        state.mode = "judge"


def submit_dialogue() -> None:
    if not state.input_text.strip() or state.active_npc is None:
        return

    npc_id = state.active_npc
    question = state.input_text.strip()
    if dialogue_session is not None and handle_player_message is not None:
        dialogue_session.sync_discovered_clues(state.discovered)
        reply = handle_player_message(npc_id, question, dialogue_session)
    else:
        reply = npc_reply(npc_id, question, state.discovered)
        if GAME_ENGINE_IMPORT_ERROR is not None:
            reply += f"\n\n[대화 엔진 오류: {type(GAME_ENGINE_IMPORT_ERROR).__name__}]"

    state.last_reply = reply
    state.conversation.setdefault(npc_id, []).append(f"P: {question}")
    state.conversation.setdefault(npc_id, []).append(f"N: {reply}")
    state.input_text = ""


def submit_judge() -> None:
    if not state.input_text.strip():
        return

    answer = state.input_text.strip()
    discovered = list(state.discovered)
    state.input_text = ""
    state.composing_text = ""
    state.judge_result = None
    state.result_scroll = 0
    state.judging = True
    state.mode = "judging"

    def worker() -> None:
        state.judge_result = judge_answer(answer, discovered)
        state.result_scroll = 0
        state.judging = False
        state.mode = "result"

    threading.Thread(target=worker, daemon=True).start()


# ------------------------------------------------------------
# 메인 루프
# ------------------------------------------------------------
def open_judge_template() -> None:
    state.input_text = ""
    state.composing_text = ""
    state.mode = "judge"


def retry_game() -> None:
    global player

    state.mode = "start"
    state.discovered.clear()
    state.conversation.clear()
    state.active_npc = None
    state.active_clue = None
    state.input_text = ""
    state.composing_text = ""
    state.last_reply = ""
    state.judge_result = None
    state.judging = False
    state.result_scroll = 0
    state.notification = ""
    state.notification_until = 0
    if dialogue_session is not None:
        dialogue_session.reset()
    for door in doors:
        door.open = False
    player = pygame.Rect(545, 650, 28, 36)


def active_input_limit() -> int:
    if state.mode == "dialogue":
        return 90
    if state.mode == "judge":
        return 240
    return 0


def append_input_text(text: str) -> None:
    limit = active_input_limit()
    if not limit:
        return

    available = limit - len(state.input_text)
    if available <= 0:
        return

    state.input_text += text[:available]


def visible_input_text(placeholder: str) -> str:
    if state.input_text or state.composing_text:
        return state.input_text + state.composing_text
    return placeholder


def main() -> None:
    global SHOW_COLLISION_DEBUG

    running = True

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
            ):
                if state.mode == "start" and START_BUTTON_RECT.collidepoint(event.pos):
                    state.mode = "play"
                elif state.mode == "result" and RETRY_BUTTON_RECT.collidepoint(event.pos):
                    retry_game()

            elif event.type == pygame.MOUSEWHEEL:
                if state.mode == "result":
                    scroll_result(-event.y * 3)

            elif event.type == pygame.TEXTEDITING:
                if state.mode in ("dialogue", "judge"):
                    state.composing_text = event.text

            elif event.type == pygame.TEXTINPUT:
                if state.mode in ("dialogue", "judge"):
                    append_input_text(event.text)
                    state.composing_text = ""

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F2:
                    SHOW_COLLISION_DEBUG = not SHOW_COLLISION_DEBUG
                    state.notify(
                        "충돌 영역 표시 ON" if SHOW_COLLISION_DEBUG else "충돌 영역 표시 OFF"
                    )

                elif event.key == pygame.K_F11 or (
                    event.key == pygame.K_RETURN
                    and event.mod & pygame.KMOD_ALT
                ):
                    toggle_fullscreen()

                elif event.key == pygame.K_ESCAPE:
                    if state.mode == "judging":
                        state.notify("추론 중입니다. 잠시만 기다려 주세요.")
                        continue

                    if state.mode in [
                        "clue",
                        "dialogue",
                        "judge",
                        "result",
                    ]:
                        state.mode = "play"
                        state.input_text = ""
                        state.composing_text = ""
                    else:
                        running = False

                elif state.mode == "start":
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        state.mode = "play"

                elif state.mode == "play":
                    if event.key == pygame.K_q:
                        handle_q()
                    elif event.key == pygame.K_SLASH or event.unicode == "/":
                        open_judge_template()

                elif state.mode == "dialogue":
                    if event.key == pygame.K_RETURN:
                        state.composing_text = ""
                        submit_dialogue()
                    elif event.key == pygame.K_BACKSPACE:
                        state.input_text = state.input_text[:-1]

                elif state.mode == "judge":
                    if event.key == pygame.K_RETURN:
                        state.composing_text = ""
                        submit_judge()
                    elif event.key == pygame.K_BACKSPACE:
                        state.input_text = state.input_text[:-1]

                elif state.mode == "result":
                    if event.key == pygame.K_DOWN:
                        scroll_result(1)
                    elif event.key == pygame.K_UP:
                        scroll_result(-1)
                    elif event.key == pygame.K_PAGEDOWN:
                        scroll_result(8)
                    elif event.key == pygame.K_PAGEUP:
                        scroll_result(-8)

        if state.mode == "play":
            keys = pygame.key.get_pressed()
            move_player(keys)

        if state.mode == "start":
            draw_start_screen()
            pygame.display.flip()
            continue

        draw_background_map()
        draw_doors()
        draw_interactable_highlight()
        draw_npcs()
        draw_player()
        draw_collision_debug()
        draw_hud()

        if state.mode == "clue":
            draw_clue_window()
        elif state.mode == "dialogue":
            draw_dialogue_window()
        elif state.mode == "judge":
            draw_judge_window()
        elif state.mode == "judging":
            draw_judging_window()
        elif state.mode == "result":
            draw_result_window()

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
