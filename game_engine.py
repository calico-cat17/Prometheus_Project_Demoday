from __future__ import annotations

import sys

# Windows 콘솔 기본 코드페이지(cp949)에서 이모지 등 출력 시 UnicodeEncodeError가
# 나는 것을 막기 위해 stdout/stderr을 UTF-8로 강제한다.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ============================================================
# 0. OpenAI 설정
# ============================================================
import json as _json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

PROJECT_ROOT = next(
    (path for path in (Path.cwd(), *Path.cwd().parents) if (path / ".git").exists()),
    Path.cwd(),
)
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ============================================================
# 1. 대화 생성 설정
# ============================================================

MAX_HISTORY_MESSAGES = 16
MAX_NEW_TOKENS = 120

# NPC별이 아니라 게임 전체가 공유하는 질문권 총 30회.
QUESTION_LIMIT = 30


# ============================================================
# 2. 게임 정보 — back 폴더의 캐릭터 JSON 4개 + 게임 JSON을 불러온다.
# ============================================================

BACK_DIR = Path(__file__).resolve().parent / "back"
GAME_JSON_PATH = BACK_DIR / "general.json"

# main.py의 NPCS 딕셔너리가 쓰는 npc_id와 동일한 키를 사용한다.
CHARACTER_FILES = {
    "chanhyung": "chanhyung_character.json",
    "sieun": "sieun_character.json",
    "jiyeon": "jiyeon_character.json",
    "juni": "junyi_character.json",
}


def load_game_info() -> dict:
    with open(GAME_JSON_PATH, encoding="utf-8") as f:
        return _json.load(f)


def load_all_profiles() -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    for character_id, filename in CHARACTER_FILES.items():
        path = BACK_DIR / filename
        with open(path, encoding="utf-8") as f:
            profiles[character_id] = _json.load(f)
    return profiles


def get_culprit_name() -> str | None:
    """PROFILES 중 is_culprit=true인 캐릭터의 이름을 반환한다."""
    for profile in PROFILES.values():
        if profile.get("is_culprit"):
            return profile["name"]
    return None


def contains_any(text: str, keywords: list[str]) -> bool:
    normalized = text.replace(" ", "").lower()
    return any(keyword.replace(" ", "").lower() in normalized for keyword in keywords)


GAME_INFO = load_game_info()
PROFILES = load_all_profiles()


# 2D 탐색으로 단서를 발견했을 때 보여줄 설명
EVIDENCE_DESCRIPTIONS = {
    "등원 기록": "데스크에 06시 30분 준이 학생 등원 기록이 남아 있다.",
    "강의 시간표": "강사실 강의 시간표에는 22시 30분에 강의가 종료되는 것으로 나와 있다.",
    "복사기": "인쇄실 복사기에 밤 10시 30분과 새벽 3시 20분 시험지 스캔 기록이 있다.",
    "상담 예약표": "원장실 상담 예약표에 이찬형 상담실장이 아침 8시 상담 예정이라고 적혀 있다.",
    "성과급 평가표": "원장실 성과급 평가표에 전국 모의고사 평균, 정예반 유지율 등이 기준으로 적혀 있다.",
    "금고 키패드 지문": "원장실 금고 키패드에 지문이 거의 남아있지 않고 닦아낸 흔적이 있다.",
    "분필통": "강사실 홍지연 자리에 분필통이 있다.",
    "계좌 번호가 적힌 메모장": "인쇄실 쓰레기통에서 상담실장 계좌번호가 적힌 메모가 발견됐다.",
    "상담실 모니터": "상담실 모니터에서 학부모와 '인쇄실 쓰레기통에 있는 계좌로 보내주세요'라는 대화가 발견됐다.",
}

CLUE_KEYWORDS = {
    "등원 기록": ["등원 기록", "등원기록", "출석 기록", "출석기록", "몇 시에 왔"],
    "강의 시간표": ["강의 시간표", "강의시간표", "수업 시간표", "시간표"],
    "복사기": ["복사기", "복사기 로그", "스캔 기록", "복사 기록", "인쇄실 복사기"],
    "상담 예약표": ["상담 예약표", "상담예약표", "예약표", "상담 일정"],
    "성과급 평가표": ["성과급 평가표", "성과급", "평가표"],
    "금고 키패드 지문": ["금고 키패드 지문", "금고 지문", "키패드 지문", "금고"],
    "분필통": ["분필통", "분필 통"],
    "계좌 번호가 적힌 메모장": ["계좌 번호", "계좌번호", "계좌 메모", "메모장"],
    "상담실 모니터": ["상담실 모니터", "모니터"],
}


# ============================================================
# 3. 세션 관리
# ============================================================
class GameSession:
    """
    - 전역(게임) 상태: 발견한 단서, 남은 질문권(30회 공유).
    - 캐릭터별 상태: 그 NPC와 나눈 대화 이력, 직접 제시했던 단서.
    """

    def __init__(self, question_limit: int = QUESTION_LIMIT):
        self._question_limit = question_limit
        self._npc_state: dict[str, dict[str, Any]] = {}
        self._global: dict[str, Any] = {
            "discovered_clues": [],
            "questions_used": 0,
        }

    def _ensure(self, character_id: str) -> dict[str, Any]:
        if character_id not in self._npc_state:
            self._npc_state[character_id] = {
                "presented_clues": [],
                "dialogue_history": [],
                "turn_count": 0,
            }
        return self._npc_state[character_id]

    def get_discovered_clues(self) -> list[str]:
        return list(self._global["discovered_clues"])

    def sync_discovered_clues(self, clue_ids: list[str]) -> None:
        """main.py(pygame 2D 탐색)에서 발견한 단서 목록을 세션에 반영한다."""
        for clue_id in clue_ids:
            if clue_id not in self._global["discovered_clues"]:
                self._global["discovered_clues"].append(clue_id)

    def get_presented_clues(self, character_id: str) -> list[str]:
        return list(self._ensure(character_id)["presented_clues"])

    def get_dialogue_history(self, character_id: str) -> list[dict[str, str]]:
        return list(self._ensure(character_id)["dialogue_history"])

    def mark_presented(self, character_id: str, clue_label: str) -> None:
        state = self._ensure(character_id)
        if clue_label not in state["presented_clues"]:
            state["presented_clues"].append(clue_label)

    def update_dialogue_history(
        self, character_id: str, player_message: str, response: str,
    ) -> None:
        state = self._ensure(character_id)
        state["dialogue_history"].append({"role": "user", "content": player_message})
        state["dialogue_history"].append({"role": "assistant", "content": response})
        state["turn_count"] += 1

    def questions_remaining(self) -> int:
        return max(0, self._question_limit - self._global["questions_used"])

    def has_questions_left(self) -> bool:
        return self.questions_remaining() > 0

    def use_question(self) -> None:
        self._global["questions_used"] = min(
            self._question_limit, self._global["questions_used"] + 1,
        )

    def reset(self, character_id: str | None = None) -> None:
        """character_id를 주면 그 NPC와의 대화만, 생략하면 게임 전체를 초기화한다."""
        if character_id is not None:
            self._npc_state[character_id] = {
                "presented_clues": [],
                "dialogue_history": [],
                "turn_count": 0,
            }
            return
        self._npc_state = {}
        self._global = {
            "discovered_clues": [],
            "questions_used": 0,
        }


# ============================================================
# 4. 프롬프트 빌더
# ============================================================

def load_profile(character_id: str) -> dict[str, Any] | None:
    return PROFILES.get(character_id)


def mentioned_clues(
    text: str, candidate_labels: list[str], keywords_map: dict[str, list[str]],
) -> list[str]:
    """candidate_labels 중 이번 텍스트(플레이어 메시지)에서 실제로 언급된 것만 반환한다."""
    return [
        label
        for label in candidate_labels
        if contains_any(text, keywords_map.get(label, [label]))
    ]


def get_turn_reactions(
    discovered_clues: list[str],
    player_message: str,
) -> tuple[list[str], list[str]]:
    """
    캐릭터 JSON에 clue_reactions/correlation_rules 같은 필드가 없으므로,
    "반응 텍스트"를 찾아주는 대신 "이번 메시지에서 어떤 단서가 언급됐는지"만
    판정한다. 실제 반응 내용은 프롬프트 안의 [아는 사실]에서 모델이 스스로 찾는다.

    1) direct_mentioned : 이미 발견됐고, 이번 메시지에서 '직접 제시'된 단서 라벨.
                           (발견만 하고 이번에 안 물어봤다면 여기 포함되지 않는다)
    2) undiscovered_mentioned : 아직 발견되지 않은 단서를 플레이어가 먼저 언급한 라벨.
    """
    direct_mentioned = mentioned_clues(player_message, discovered_clues, CLUE_KEYWORDS)

    all_labels = list(CLUE_KEYWORDS.keys())
    undiscovered_labels = [c for c in all_labels if c not in discovered_clues]
    undiscovered_mentioned = mentioned_clues(player_message, undiscovered_labels, CLUE_KEYWORDS)

    return direct_mentioned, undiscovered_mentioned


def prior_questions_summary(dialogue_history: list[dict[str, Any]], limit: int = 8) -> str:
    """대화 이력이 길어져 LLM에 보내는 윈도우 밖으로 밀려나도
    이전에 어떤 질문을 받았는지 요약해서 일관성을 유지하도록 돕는다."""
    questions: list[str] = []
    for item in dialogue_history:
        if item.get("role") == "user":
            content = (item.get("content") or "").strip()
            if content and content not in questions:
                questions.append(content)

    if not questions:
        return "- 아직 질문한 내용이 없다."

    recent = questions[-limit:]
    return "\n".join(f"- {q}" for q in recent)


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- 없음"


def build_prompt(
    profile: dict[str, Any],
    direct_mentioned: list[str],
    undiscovered_mentioned: list[str],
    prior_questions: str,
) -> str:
    name = profile["name"]
    game = GAME_INFO["game"]
    policy = profile["conversation_policy"]

    direct_text = (
        "\n".join(f"- {c}" for c in direct_mentioned)
        if direct_mentioned
        else "- 이번 질문에서 플레이어가 직접 제시한, 이미 발견된 단서는 없다."
    )
    undiscovered_text = (
        "\n".join(f"- {c}" for c in undiscovered_mentioned)
        if undiscovered_mentioned
        else "- 없음"
    )
    examples_text = "\n\n".join(
        f"질문: {ex['question']}\n{name}: {ex['answer']}"
        for ex in profile.get("response_examples", [])
    ) or "- 없음"

    return f'''
당신은 추리 게임 《{game['title']}》의 NPC '{name}'다.
현재 상황: {game['current_situation']}
사건 개요: {game['incident']['description']} (발생 인지 시각: {game['incident']['discovered_at']})
플레이어는 이 사건을 조사하며 당신을 심문하고 있다.

[직업/역할]
{profile['job']}

[말투]
{bullets(profile['speaking_style'])}

[성격]
{bullets(profile['personality'])}

[타임라인]
{bullets(profile['time_line'])}

[아는 사실 — 확실히 겪은 일이므로 모른다고 하지 않고 이 내용 그대로 답한다]
{bullets(profile['known_facts'])}

[모르는 사실 — 물어보면 반드시 모른다/못 봤다로 답한다]
{bullets(profile['unknown_facts'])}

[업무/일상]
{bullets(profile['daily_context']['work'])}

[인간관계]
{bullets(profile['daily_context']['relationships'])}

[평소 태도]
{bullets(profile['daily_context']['general_behavior'])}

[이번 질문에서 플레이어가 직접 제시한, 이미 발견된 단서]
{direct_text}
→ 위 [아는 사실]에서 관련된 내용을 찾아 자연스럽게 답한다. 이미 발견된 단서라도
  이번 질문에서 언급되지 않았다면 스스로 먼저 꺼내지 않는다.

[플레이어가 이번 질문에서 언급했지만 아직 발견되지 않은 단서]
{undiscovered_text}
→ 세부 내용을 아는 것처럼 답하지 않고 모른다/처음 듣는다고만 답한다.

[지금까지 플레이어가 물어봤던 질문 — 일관성 유지용 참고, 답변을 반복할 필요는 없음]
{prior_questions}

[대화 원칙]
{bullets(policy['general_rules'])}

[사건과 무관한 질문 대응 원칙]
{bullets(policy['unrelated_question_rules'])}

[응답 형식]
{bullets(policy['response_format'])}

[대화 예시 — 말투와 태도만 참고, 문장을 그대로 재사용하지 말 것]
{examples_text}

[추가로 지켜야 할 게임 공통 규칙]
1. 순수 한글로만 답하고, 문장 수를 채우려고 근거 없는 세부 사항을 지어내지 않는다.
2. "모른다"는 실제로 모르는 것을 물었을 때만 말하고, 한 번의 답변 안에서 같은
   취지의 말을 두 번 반복하지 않는다.
3. 한 응답 안에서 서로 다른 시간·장소·사실을 섞어서 스스로 모순되게 말하지 않는다.
4. "누가 범인 같아?" 류 질문에는 장황하게 추리하지 않고 아주 짧게(1~2문장) 답한다.
5. 플레이어가 사실을 단정적으로 말해도 그대로 따라가지 않는다.
6. 이전 답변과 모순되지 않게 하되, 비슷한 질문을 다시 받으면 토씨 하나까지 그대로
   반복하지 말고 다른 표현으로 짧게 답한다.
'''.strip()


# ============================================================
# 5. 응답 후처리 가드레일
# ============================================================


UNAWARE_MARKERS = ["몰라", "모르", "처음 듣", "글쎄", "잘 모르겠", "본 적 없", "안 가"]

_HANJA_PATTERN = re.compile(r"[一-鿿㐀-䶿]")


def contains_hanja(text: str) -> bool:
    """응답에 한자가 섞여 나왔는지 검사한다.
    (실제 관찰된 오류: '복도' -> '복道', '이찬형' -> '이창현'처럼
    순한글이어야 할 부분에 한자가 끼어들거나 이름이 깨지는 경우가 있었다)"""
    return bool(_HANJA_PATTERN.search(text))


def is_too_long(text: str, max_sentences: int = 3) -> bool:
    """캐릭터 JSON의 response_format이 공통적으로 '한 문장부터 세 문장 정도'를
    명시하므로, 그 기준을 그대로 가드레일 임계값으로 쓴다."""
    sentences = [s for s in re.split(r"[.!?…]+", text) if s.strip()]
    return len(sentences) > max_sentences


def leaks_undiscovered_clue(
    response: str,
    undiscovered_mentions: list[str],
    keywords_map: dict[str, list[str]],
) -> bool:
    """플레이어가 이번 턴에 언급한, 아직 발견되지 않은 단서에 대해
    NPC가 마치 알고 있다는 듯 구체적으로 답했는지 검사한다."""
    if not undiscovered_mentions:
        return False
    if contains_any(response, UNAWARE_MARKERS):
        return False
    return any(
        contains_any(response, keywords_map.get(clue, [clue]))
        for clue in undiscovered_mentions
    )


def violates_guardrail(
    response: str,
    unknown_facts: list[str],
    culprit: str | None,
    undiscovered_mentions: list[str] | None = None,
) -> bool:
    # TIMELINE_CONTRADICTIONS(과거에 썼던 "집에서 공부"/"출근"/"회사" 키워드 목록)는
    # 쓰지 않는다. 캐릭터가 1명일 땐 몰라도, 4명으로 늘어난 지금은 예를 들어
    # 홍지연의 known_facts에 실제로 "아침 8시에 출근했다"가 있어서, 그 사실을
    # 그대로 답해도 "출근"이라는 단어 때문에 가드레일에 오탐지될 수 있었다.
    # 캐릭터마다 사실이 달라 공용 키워드 목록으로는 일반화가 안 되는 방식이었다.
    if contains_any(response, unknown_facts):
        return True

    if culprit:
        culprit_patterns = [
            f"{culprit}이 범인", f"{culprit}이가 범인", f"범인은 {culprit}",
            f"{culprit}이 훔쳤", f"{culprit}이 시험지를 유출",
        ]
        if contains_any(response, culprit_patterns):
            return True

    if leaks_undiscovered_clue(response, undiscovered_mentions or [], CLUE_KEYWORDS):
        return True

    if contains_hanja(response):
        return True

    if is_too_long(response):
        return True

    return False


STRICT_REMINDER = (
    "\n\n[재생성 지시] 방금 응답에 문제가 있었다 — 공개 금지 정보를 언급했거나, "
    "아직 발견되지 않은 단서를 이미 아는 것처럼 답했거나, 한자가 섞여 나왔거나, "
    "너무 길고 장황했다. 순수 한글로만, 1~3문장 이내로 짧고 담백하게, "
    "'이번 질문에서 플레이어가 직접 제시한, 이미 발견된 단서'와 '아는 사실'에 있는 "
    "내용만 사용해서 다시 답하라. "
    "만약 확실하지 않다면 '잘 모르겠습니다.'로만 답하라."
)


# ============================================================
# 6. OpenAI 호출
# ============================================================

def normalize_history(
    dialogue_history: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in (dialogue_history or [])[-MAX_HISTORY_MESSAGES:]:
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"}:
            continue
        if not isinstance(content, str):
            continue
        messages.append({"role": role, "content": content})
    return messages


def clean_reply(text: str) -> str:
    text = (text or "").strip()
    names = [profile["name"] for profile in PROFILES.values()]
    name_pattern = "|".join(re.escape(name) for name in names)
    text = re.sub(
        rf"^({name_pattern}|assistant)\s*[:：]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = fix_common_typos(text)
    text = text.split("\n\n")[0]
    return text.strip()


_TYPO_FIXES = [
    (re.compile(r"모른습니다"), "모릅니다"),
    (re.compile(r"모른어요"), "몰라요"),
    (re.compile(r"모른아요"), "몰라요"),
    (re.compile(r"모른네요"), "모르겠네요"),
]


def fix_common_typos(text: str) -> str:
    for pattern, repl in _TYPO_FIXES:
        text = pattern.sub(repl, text)
    return text


def build_user_turn(player_message: str) -> str:
    """매 턴마다 [아는 사실]만 근거로 답하라는 지시를 한 번 더 얹는다."""
    return (
        f"플레이어 질문:\n\n{player_message}\n\n"
        "위 질문에 대해\n"
        "반드시 시스템 프롬프트의 [아는 사실]만 근거로 답하세요.\n"
        "모르는 내용은 절대 추측하지 마세요."
    )


def call_llm(system_prompt: str, dialogue_history: list[dict[str, Any]], player_message: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(normalize_history(dialogue_history))
    messages.append({"role": "user", "content": build_user_turn(player_message)})

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
    )
    text = resp.choices[0].message.content or ""
    return clean_reply(text)


# ============================================================
# 7. 오케스트레이션
# ============================================================

QUESTIONS_EXHAUSTED_MESSAGE = (
    "질문권을 모두 사용하셨어요... 더 이상 심문할 수 없습니다. "
    "('초기화' 후 다시 시작해 주세요.)"
)

_CULPRIT_NAME = get_culprit_name()


def handle_player_message(character_id: str, player_message: str, session: GameSession) -> str:
    profile = load_profile(character_id)
    if profile is None:
        return "지금 이 자리에는 그런 사람이 없는 것 같은데요."

    # 0. 게임 전체가 공유하는 질문권이 남아있는지 먼저 확인 (모델 호출 없이 즉시 반환)
    if not session.has_questions_left():
        return QUESTIONS_EXHAUSTED_MESSAGE

    discovered_clues = session.get_discovered_clues()
    dialogue_history = session.get_dialogue_history(character_id)

    # 1. 이번 메시지 기준으로: 직접 제시된 단서 / 미발견 단서 언급을 판정
    direct_mentioned, undiscovered_mentioned = get_turn_reactions(
        discovered_clues, player_message,
    )

    # 2. 대화 윈도우 밖으로 밀려난 과거 질문까지 반영하기 위한 요약
    prior_questions = prior_questions_summary(dialogue_history)

    # 3. 시스템 프롬프트 조립 (캐릭터 JSON 전체 + 게임 로직이 계산한 단서 정보)
    system_prompt = build_prompt(
        profile=profile,
        direct_mentioned=direct_mentioned,
        undiscovered_mentioned=undiscovered_mentioned,
        prior_questions=prior_questions,
    )

    # 4. LLM 호출
    response = call_llm(system_prompt, dialogue_history, player_message)

    # 5. 가드레일 검증 (실패 시 1회 재생성)
    if violates_guardrail(
        response, profile["unknown_facts"], culprit=_CULPRIT_NAME,
        undiscovered_mentions=undiscovered_mentioned,
    ):
        response = call_llm(system_prompt + STRICT_REMINDER, dialogue_history, player_message)
        response = clean_reply(response)
        if not response or violates_guardrail(
            response, profile["unknown_facts"], culprit=_CULPRIT_NAME,
            undiscovered_mentions=undiscovered_mentioned,
        ):
            response = "그건 제가 아는 부분이 아니라서 말씀드리기 어려워요."

    # 6. 상태 업데이트 (질문권은 게임 전역에서 1회 차감)
    session.update_dialogue_history(character_id, player_message, response)
    session.use_question()

    for clue in direct_mentioned:
        session.mark_presented(character_id, clue)

    return response
