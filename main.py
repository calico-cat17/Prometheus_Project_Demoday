# ============================================================
# main.py — 게임 진행/로직 전담. 그리기는 전부 ui.py에 위임한다.
# ============================================================
# 실행 전 준비물:
#   1) game_engine.py  (ipynb를 `jupyter nbconvert --to script`로 변환한 것.
#      GameSession, ask_npc, move_to, explore_here, get_state, save_note,
#      get_notes, delete_note, submit_reasoning, LOCATIONS 를 제공해야 한다.
#      submit_reasoning()의 반환값에는 "player_storyboard"/"true_storyboard"
#      — [{"character":..., "location":..., "action":...}, ...] 리스트 — 가
#      포함되어 있어야 한다. case_reconstruction_guide.md 참고.)
#   2) ui.py            (같은 폴더의 기본 렌더링 모듈. 나중에 지연님 버전으로 교체)
#
# 실행: python main.py

from __future__ import annotations

import queue
import sys
import threading

import pygame
from pygame.math import Vector2

import game_engine as ge
import ui

# ------------------------------------------------------------
# 화면/맵 설정
# ------------------------------------------------------------
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
MAP_WIDTH = 860           # 화면 왼쪽: 맵, 오른쪽: HUD
HUD_RECT = pygame.Rect(MAP_WIDTH, 0, SCREEN_WIDTH - MAP_WIDTH, SCREEN_HEIGHT)

FPS = 60
PLAYER_SIZE = 28
PLAYER_SPEED = 4

# 장소 이름은 game_engine.LOCATIONS(= ["강의실","인쇄실","복도","상담실","원장실"])와
# 정확히 일치해야 한다. 좌표는 지연님이 실제 맵 디자인을 만들면 그때 교체한다.
ROOM_LAYOUT = {
    "강의실": pygame.Rect(20, 20, 400, 260),
    "인쇄실": pygame.Rect(440, 20, 400, 260),
    "복도":   pygame.Rect(20, 300, 820, 120),
    "상담실": pygame.Rect(20, 440, 400, 260),
    "원장실": pygame.Rect(440, 440, 400, 260),
}

# ------------------------------------------------------------
# 상태(모드) 정의
# ------------------------------------------------------------
MODE_MOVE = "move"
MODE_DIALOGUE_INPUT = "dialogue_input"
MODE_DIALOGUE_WAIT = "dialogue_wait"
MODE_DIALOGUE_RESULT = "dialogue_result"
MODE_EXPLORE_RESULT = "explore_result"
MODE_MEMO = "memo"
MODE_SUBMIT = "submit"
MODE_REPLAY_PLAYER = "replay_player"   # 플레이어의 추리대로 캐릭터가 걸어다니며 재현
MODE_REPLAY_TRUE = "replay_true"       # 실제 사건대로 다시 재현
MODE_RESULT = "result"

SUBMIT_FIELDS = ["culprit", "explanation"]  # 나머지(동기/증거)는 상세 설명에 통합해서 입력받는다
REPLAY_MOVE_SPEED = 6  # 리플레이 중 캐릭터가 프레임당 이동하는 픽셀 수


# ------------------------------------------------------------
# LLM 호출을 메인 루프를 막지 않고 실행하기 위한 워커
# ------------------------------------------------------------
class LLMWorker:
    """ask_npc / submit_reasoning처럼 시간이 걸리는 game_engine 호출을
    별도 스레드에서 실행하고, 결과를 큐로 돌려준다.

    job_id는 "이 결과가 지금 화면이 기다리는 바로 그 요청인지" 판별하기 위한
    표식이다. 실제 game_engine 호출(스레드 안에서 이미 시작된 MODEL.generate() 등)을
    중간에 강제로 멈출 방법은 없으므로, ESC로 "취소"해도 백그라운드 호출 자체는
    끝까지 실행된다 — 다만 그 결과가 나중에 도착했을 때 job_id가 더 이상 화면이
    기다리는 값이 아니면 조용히 버린다."""

    def __init__(self) -> None:
        self._in_queue: "queue.Queue[tuple[str, object, object, tuple]]" = queue.Queue()
        self._out_queue: "queue.Queue[tuple[str, object, object]]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            job_type, job_id, fn, args = self._in_queue.get()
            try:
                result = fn(*args)
            except Exception as exc:  # noqa: BLE001 — 백그라운드 스레드 예외를 그대로 전달
                result = {"error": str(exc)} if job_type == "submit" else f"(오류: {exc})"
            self._out_queue.put((job_type, job_id, result))

    def submit(self, job_type: str, job_id, fn, *args) -> None:
        self._in_queue.put((job_type, job_id, fn, args))

    def poll(self):
        try:
            return self._out_queue.get_nowait()
        except queue.Empty:
            return None


def detect_room(player_rect: pygame.Rect, room_layout: dict[str, pygame.Rect]) -> str | None:
    for name, rect in room_layout.items():
        if rect.collidepoint(player_rect.center):
            return name
    return None


def room_center(location: str, room_layout: dict[str, pygame.Rect]) -> Vector2:
    return Vector2(room_layout[location].center)


def clamp_player_to_map(player_rect: pygame.Rect) -> None:
    player_rect.left = max(0, min(player_rect.left, MAP_WIDTH - player_rect.width))
    player_rect.top = max(0, min(player_rect.top, SCREEN_HEIGHT - player_rect.height))


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("봉인된 모의고사")
    clock = pygame.time.Clock()
    # 한글 등 IME 조합 입력은 KEYDOWN.unicode가 아니라 TEXTINPUT 이벤트로 들어온다.
    pygame.key.start_text_input()

    session = ge.GameSession()
    llm_worker = LLMWorker()

    mode = MODE_MOVE
    input_text = ""
    last_npc_name = ""
    last_npc_reply = ""
    last_explore_result: dict | None = None
    submit_form = {key: "" for key in SUBMIT_FIELDS}
    submit_field_idx = 0
    final_result: dict | None = None
    memo_selected_index = 0  # 메모장에서 위/아래로 고른 메모의 인덱스 (DELETE로 그 메모를 지움)

    # WASD 이동은 keycode(pygame.K_w 등) 대신 scancode(물리적 키 위치)로 판정한다.
    # 한글 IME가 켜져 있으면 W 키 위치에서 'ㅈ' 등이 입력되면서 keycode 매핑이
    # 깨져 pygame.K_w로 못 잡는 경우가 있는데, scancode는 키보드 레이아웃/IME
    # 상태와 무관하게 항상 같은 물리적 키를 가리켜서 그 문제가 생기지 않는다.
    pressed_scancodes: set[int] = set()

    # ask_npc 요청을 ESC로 "취소"하기 위한 표식.
    # 백그라운드 스레드에서 이미 시작된 game_engine 호출 자체를 멈출 수는 없으므로,
    # 화면만 먼저 MOVE로 돌려보내고 pending_ask_id를 비운다. 나중에 그 요청의
    # 결과가 뒤늦게 도착하면(job_id가 더 이상 pending_ask_id와 다름) 조용히 버린다.
    ask_request_id = 0
    pending_ask_id: int | None = None

    # 리플레이(캐릭터가 걸어다니며 사건을 보여주는 장면) 상태.
    # dict의 값만 갱신하므로 nonlocal 선언 없이 내부 함수에서 그대로 수정 가능하다.
    replay = {
        "steps": [],       # [{"character":..., "location":..., "action":...}, ...]
        "index": 0,
        "phase": "player",     # "player" | "true"
        "actor_pos": Vector2(0, 0),   # 캐릭터의 실제 픽셀 좌표 (걸어다니는 중간 위치 포함)
        "target_pos": Vector2(0, 0),  # 지금 걸어가고 있는 목적지 좌표
        "arrived": True,              # 목적지에 도착해서 대사를 보여줄 차례인지
    }

    def begin_replay_phase(steps: list[dict], phase: str) -> None:
        """플레이어 재구성 또는 정답 재구성을 처음부터 시작한다."""
        replay["steps"] = steps
        replay["index"] = 0
        replay["phase"] = phase
        replay["actor_pos"] = room_center("복도", ROOM_LAYOUT)  # 항상 복도에서 출발
        if steps:
            replay["target_pos"] = room_center(steps[0]["location"], ROOM_LAYOUT)
            replay["arrived"] = replay["actor_pos"] == replay["target_pos"]
        else:
            replay["target_pos"] = replay["actor_pos"]
            replay["arrived"] = True

    def advance_replay_step() -> bool:
        """다음 장면으로 넘어간다. 이 phase의 마지막 장면이었으면 False를 반환한다."""
        replay["index"] += 1
        if replay["index"] >= len(replay["steps"]):
            return False
        replay["target_pos"] = room_center(replay["steps"][replay["index"]]["location"], ROOM_LAYOUT)
        replay["arrived"] = replay["actor_pos"] == replay["target_pos"]
        return True

    start_rect = ROOM_LAYOUT["복도"]
    player_rect = pygame.Rect(0, 0, PLAYER_SIZE, PLAYER_SIZE)
    player_rect.center = start_rect.center

    running = True
    while running:
        # ---------------- 이벤트 처리 ----------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYUP:
                pressed_scancodes.discard(event.scancode)

            elif event.type == pygame.KEYDOWN:
                pressed_scancodes.add(event.scancode)
                if mode == MODE_MOVE:
                    if event.key == pygame.K_q:
                        state = ge.get_state(session)
                        if state["npc_name"]:
                            last_npc_name = state["npc_name"]
                            mode, input_text = MODE_DIALOGUE_INPUT, ""

                    elif event.key == pygame.K_e:
                        last_explore_result = ge.explore_here(session)
                        mode = MODE_EXPLORE_RESULT

                    elif event.key == pygame.K_TAB:
                        mode = MODE_MEMO
                        input_text = ""
                        memo_selected_index = 0

                    elif event.key == pygame.K_RETURN:
                        mode = MODE_SUBMIT
                        submit_field_idx = 0

                elif mode == MODE_DIALOGUE_INPUT:
                    if event.key == pygame.K_RETURN and input_text.strip():
                        ask_request_id += 1
                        pending_ask_id = ask_request_id
                        llm_worker.submit("ask", pending_ask_id, ge.ask_npc, session, input_text)
                        mode, input_text = MODE_DIALOGUE_WAIT, ""
                    elif event.key == pygame.K_ESCAPE:
                        mode = MODE_MOVE
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]

                elif mode == MODE_DIALOGUE_WAIT:
                    # submit_reasoning 대기도 같은 모드를 쓰지만, 그건 취소 대상이 아니다
                    # (pending_ask_id가 None이면 지금은 질문 응답이 아니라 추리 채점을
                    # 기다리는 중이라는 뜻이므로 ESC를 무시한다).
                    if event.key == pygame.K_ESCAPE and pending_ask_id is not None:
                        pending_ask_id = None
                        mode = MODE_MOVE

                elif mode == MODE_DIALOGUE_RESULT:
                    if event.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                        mode = MODE_MOVE

                elif mode == MODE_EXPLORE_RESULT:
                    if event.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                        mode = MODE_MOVE

                elif mode == MODE_MEMO:
                    notes = ge.get_notes(session)
                    if event.key == pygame.K_ESCAPE:
                        mode = MODE_MOVE
                    elif event.key == pygame.K_RETURN and input_text.strip():
                        ge.save_note(session, input_text)
                        input_text = ""
                        memo_selected_index = max(0, len(ge.get_notes(session)) - 1)
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    elif event.key == pygame.K_UP and notes:
                        memo_selected_index = max(0, memo_selected_index - 1)
                    elif event.key == pygame.K_DOWN and notes:
                        memo_selected_index = min(len(notes) - 1, memo_selected_index + 1)
                    elif event.key == pygame.K_DELETE and notes:
                        ge.delete_note(session, memo_selected_index)
                        memo_selected_index = min(memo_selected_index, max(0, len(notes) - 2))

                elif mode == MODE_SUBMIT:
                    if event.key == pygame.K_ESCAPE:
                        mode = MODE_MOVE
                    elif event.key == pygame.K_TAB:
                        submit_field_idx = (submit_field_idx + 1) % len(SUBMIT_FIELDS)
                    elif event.key == pygame.K_RETURN:
                        pending_ask_id = None  # 지금부터의 대기는 질문이 아니라 채점이다 (ESC로 취소 불가)
                        # 동기/증거 입력칸을 따로 두지 않고, 상세 설명 하나로 통합해서 받는다.
                        # evaluate_deduction()/스토리보드 생성 쪽에서 motive·evidence·explanation
                        # 중복 문자열은 알아서 한 번만 남기고 합치므로, 같은 텍스트를
                        # 세 자리에 그대로 넘겨도 된다.
                        explanation = submit_form["explanation"]
                        llm_worker.submit(
                            "submit", None, ge.submit_reasoning, session,
                            submit_form["culprit"], explanation, explanation, explanation,
                        )
                        mode = MODE_DIALOGUE_WAIT
                    elif event.key == pygame.K_BACKSPACE:
                        key = SUBMIT_FIELDS[submit_field_idx]
                        submit_form[key] = submit_form[key][:-1]

                elif mode == MODE_RESULT:
                    if event.key == pygame.K_RETURN:
                        session = ge.GameSession()
                        final_result = None
                        mode = MODE_MOVE
                        player_rect.center = ROOM_LAYOUT["복도"].center

                elif mode in (MODE_REPLAY_PLAYER, MODE_REPLAY_TRUE):
                    # 캐릭터가 목적지에 도착해서 대사가 보이는 중일 때만 다음 장면으로 넘긴다.
                    # 걸어가는 도중에는 입력을 무시한다 (정지 장면 전환이 아니라 실제로 다 걸어가야 함).
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE) and replay["arrived"]:
                        has_more = advance_replay_step()
                        if not has_more:
                            if mode == MODE_REPLAY_PLAYER:
                                begin_replay_phase(final_result["true_storyboard"], "true")
                                mode = MODE_REPLAY_TRUE
                            else:
                                mode = MODE_RESULT

            elif event.type == pygame.TEXTINPUT:
                # 한글 등 IME 조합 문자는 KEYDOWN이 아니라 여기(event.text)로 완성되어 들어온다.
                if mode == MODE_DIALOGUE_INPUT:
                    input_text += event.text
                elif mode == MODE_MEMO:
                    input_text += event.text
                elif mode == MODE_SUBMIT:
                    key = SUBMIT_FIELDS[submit_field_idx]
                    submit_form[key] += event.text

        # ---------------- 워커 결과 수거 ----------------
        job = llm_worker.poll()
        if job is not None:
            job_type, job_id, result = job
            if job_type == "ask":
                # ESC로 취소된 뒤 뒤늦게 도착한 응답(job_id가 더 이상 pending_ask_id와
                # 다름)이면 화면에 반영하지 않고 조용히 버린다.
                if job_id is not None and job_id == pending_ask_id:
                    last_npc_reply = result
                    pending_ask_id = None
                    mode = MODE_DIALOGUE_RESULT
            elif job_type == "submit":
                if "error" in result:
                    # submit_reasoning 실행 중 예외가 났을 때 — player_storyboard가
                    # 없는 채로 리플레이를 시작하면 곧장 죽으므로, 대신 대화 결과
                    # 화면을 재사용해 에러 메시지를 보여주고 이동 화면으로 돌아간다.
                    last_npc_name = "오류"
                    last_npc_reply = f"추리 채점 중 오류가 발생했습니다: {result['error']}"
                    mode = MODE_DIALOGUE_RESULT
                else:
                    final_result = result
                    begin_replay_phase(result["player_storyboard"], "player")
                    mode = MODE_REPLAY_PLAYER

        # ---------------- 리플레이 캐릭터 이동(걷기 애니메이션) ----------------
        if mode in (MODE_REPLAY_PLAYER, MODE_REPLAY_TRUE) and not replay["arrived"]:
            direction = replay["target_pos"] - replay["actor_pos"]
            distance = direction.length()
            if distance <= REPLAY_MOVE_SPEED:
                replay["actor_pos"] = replay["target_pos"].copy()
                replay["arrived"] = True
            else:
                replay["actor_pos"] += direction.normalize() * REPLAY_MOVE_SPEED

        # ---------------- 이동 처리 ----------------
        if mode == MODE_MOVE:
            dx = (
                (pygame.KSCAN_D in pressed_scancodes) - (pygame.KSCAN_A in pressed_scancodes)
            ) * PLAYER_SPEED
            dy = (
                (pygame.KSCAN_S in pressed_scancodes) - (pygame.KSCAN_W in pressed_scancodes)
            ) * PLAYER_SPEED
            if dx or dy:
                player_rect.move_ip(dx, dy)
                clamp_player_to_map(player_rect)

                entered = detect_room(player_rect, ROOM_LAYOUT)
                current_location = ge.get_state(session)["location"]
                if entered and entered != current_location:
                    ge.move_to(session, entered)

        # ---------------- 렌더링 ----------------
        state = ge.get_state(session)

        if mode == MODE_MOVE:
            ui.draw_map(screen, state, player_rect, ROOM_LAYOUT, ge.LOCATION_NPC)
        elif mode == MODE_DIALOGUE_INPUT:
            ui.draw_dialogue_input(screen, last_npc_name, input_text, waiting=False)
        elif mode == MODE_DIALOGUE_WAIT:
            if pending_ask_id is not None:
                ui.draw_dialogue_input(screen, last_npc_name, "", waiting=True, cancellable=True)
            else:
                ui.draw_grading_wait(screen)
        elif mode == MODE_DIALOGUE_RESULT:
            ui.draw_dialogue_result(screen, last_npc_name, last_npc_reply)
        elif mode == MODE_EXPLORE_RESULT and last_explore_result is not None:
            ui.draw_explore_result(screen, last_explore_result)
        elif mode == MODE_MEMO:
            ui.draw_memo(screen, ge.get_notes(session), input_text, memo_selected_index)
        elif mode == MODE_SUBMIT:
            ui.draw_submit_form(screen, submit_form, SUBMIT_FIELDS[submit_field_idx])
        elif mode in (MODE_REPLAY_PLAYER, MODE_REPLAY_TRUE):
            if replay["steps"]:
                current_step = replay["steps"][replay["index"]]
                phase_label = "플레이어의 추리대로라면..." if mode == MODE_REPLAY_PLAYER else "실제로는..."
                ui.draw_replay(
                    screen, current_step, phase_label, ROOM_LAYOUT,
                    replay["actor_pos"], replay["arrived"],
                )
        elif mode == MODE_RESULT and final_result is not None:
            ui.draw_result(screen, final_result)

        ui.draw_hud(screen, state, HUD_RECT)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
