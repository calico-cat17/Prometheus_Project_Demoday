from __future__ import annotations

import sys

# Windows 콘솔 기본 코드페이지(cp949)에서 이모지 등 출력 시 UnicodeEncodeError가
# 나는 것을 막기 위해 stdout/stderr을 UTF-8로 강제한다.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ============================================================
# 0. LLM 백엔드 설정
# ============================================================
import os

# "qwen"      : 로컬 GPU(또는 CPU)에서 Qwen2.5-3B-Instruct 실행 (개발용, 무료)
# "openai"    : OpenAI GPT API 호출
# "anthropic" : Anthropic Claude API 호출
LLM_BACKEND = "openai"



QWEN_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
OPENAI_MODEL = "gpt-4o-mini"

# 여기에 API 키를 직접 삽입하지 말고 꼭!!!! 환경변수로 분리하기
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

if LLM_BACKEND not in {"qwen", "openai", "anthropic"}:
    raise ValueError(f"알 수 없는 LLM_BACKEND: {LLM_BACKEND!r} (qwen/openai/anthropic 중 하나)")

print(f"✅ LLM_BACKEND = {LLM_BACKEND!r}")
if LLM_BACKEND == "openai" and not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY가 비어 있습니다. 위 안내대로 설정한 뒤 이 셀을 다시 실행하세요.")
if LLM_BACKEND == "anthropic" and not ANTHROPIC_API_KEY:
    print("⚠️ ANTHROPIC_API_KEY가 비어 있습니다. 위 안내대로 설정한 뒤 이 셀을 다시 실행하세요.")


import torch

if LLM_BACKEND == "qwen":
    print("CUDA 사용 가능:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print(
            "⚠️ GPU가 없습니다. Qwen은 CPU로도 돌아가지만 응답이 많이 느려집니다. "
            "로컬에 GPU가 없다면 LLM_BACKEND를 'openai' 또는 'anthropic'으로 "
            "바꾸는 것을 권장합니다."
        )
    else:
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "VRAM:",
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "GB",
        )
else:
    # API 백엔드는 로컬 GPU가 필요 없음
    print(f"LLM_BACKEND = {LLM_BACKEND!r} — API를 호출하므로 GPU 체크를 건너뜁니다.")


import copy
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ============================================================
# 1. 모델/생성 설정
# ============================================================

MAX_HISTORY_MESSAGES = 16
MAX_NEW_TOKENS = 120

# NPC별이 아니라 게임 전체가 공유하는 질문권 총 30회.
QUESTION_LIMIT = 30


def load_qwen_model():
    if not torch.cuda.is_available():
        print("⚠️ GPU 없이 CPU로 Qwen을 로딩합니다 — 응답이 느릴 수 있습니다.")
        # CPU 메모리가 넉넉하지 않은 환경(예: 8GB 이하)에서는 float32(약 12GB)로
        # 로딩하다가 RAM이 부족해지고, device_map="auto"가 이를 디스크로 오프로드
        # 하면서 생성 시점에 meta 텐서에 접근해 세그폴트가 났었다. bfloat16으로
        # 메모리 사용량을 절반으로 줄이고, device_map을 "auto" 대신 고정된
        # {"": "cpu"}로 못박아 accelerate의 자동 오프로드 로직 자체를 끈다.
        compute_dtype = torch.bfloat16
        device_map = {"": "cpu"}
    else:
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        device_map = "auto"

    print(f"⏳ 모델 로딩: {QWEN_MODEL_ID}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    ) if torch.cuda.is_available() else None

    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_ID,
        device_map=device_map,
        quantization_config=quantization_config,
        dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()

    print("✅ Qwen 모델 로딩 완료")
    return tokenizer, model


# v6: LLM_BACKEND가 "qwen"일 때만 실제로 다운로드/로딩
# openai/anthropic 백엔드에서는 TOKENIZER/MODEL을 아예 안 쓴다 (None으로 둠)
TOKENIZER = None
MODEL = None

if LLM_BACKEND == "qwen":
    TOKENIZER, MODEL = load_qwen_model()


# ============================================================
# 2. 게임 정보
# ============================================================

import json as _json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
GAME_JSON_PATH = DATA_DIR / "game.json"
CHARACTER_NAMES = ["김준이", "홍지연", "이찬형", "김시은"]


def load_game_info() -> dict:
    with open(GAME_JSON_PATH, encoding="utf-8") as f:
        return _json.load(f)


def load_all_profiles() -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    for name in CHARACTER_NAMES:
        path = DATA_DIR / f"{name}.json"
        with open(path, encoding="utf-8") as f:
            profiles[name] = _json.load(f)
    return profiles


def get_culprit_name() -> str | None:
    """PROFILES 중 is_culprit=true인 캐릭터의 이름을 반환한다."""
    for name, profile in PROFILES.items():
        if profile.get("is_culprit"):
            return name
    return None


def contains_any(text: str, keywords: list[str]) -> bool:
    normalized = text.replace(" ", "").lower()
    return any(keyword.replace(" ", "").lower() in normalized for keyword in keywords)




def _write_default_json_files_if_missing() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if not GAME_JSON_PATH.exists():
        with open(GAME_JSON_PATH, "w", encoding="utf-8") as f:
            _json.dump(_DEFAULT_GAME_INFO, f, ensure_ascii=False, indent=2)
    for name, data in _DEFAULT_PROFILES.items():
        path = DATA_DIR / f"{name}.json"
        if not path.exists():
            with open(path, "w", encoding="utf-8") as f:
                _json.dump(data, f, ensure_ascii=False, indent=2)


_write_default_json_files_if_missing()

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
# 2-1. 장소 / 이동 가능한 공간
# ============================================================
LOCATIONS = ["강의실", "인쇄실", "복도", "상담실", "원장실"]
DEFAULT_LOCATION = "복도"

LOCATION_NPC: dict[str, str | None] = {
    "강의실": "홍지연",
    "인쇄실": None,
    "복도": "김준이",
    "상담실": "이찬형",
    "원장실": "김시은",
}

LOCATION_CLUES: dict[str, list[str]] = {
    "강의실": ["강의 시간표", "분필통"],
    "인쇄실": ["복사기", "계좌 번호가 적힌 메모장"],
    "복도": ["등원 기록"],
    "상담실": ["상담실 모니터"],
    "원장실": ["상담 예약표", "성과급 평가표", "금고 키패드 지문"],
}

assert sorted(sum(LOCATION_CLUES.values(), [])) == sorted(EVIDENCE_DESCRIPTIONS.keys()), (
    "LOCATION_CLUES에 EVIDENCE_DESCRIPTIONS의 모든 단서가 정확히 한 번씩 배치되어야 합니다."
)


# ============================================================
# 2-2. 세션 관리
# ============================================================
class GameSession:
    """
    - 전역(게임) 상태: 현재 위치, 장소별 탐색 진행도, 발견한 단서, 남은 질문권(30회 공유).
    - 캐릭터별 상태: 그 NPC와 나눈 대화 이력, 직접 제시했던 단서.
    """

    def __init__(self, question_limit: int = QUESTION_LIMIT):
        self._npc_state: dict[str, dict[str, Any]] = {}
        self._question_limit = question_limit
        self._global: dict[str, Any] = {
            "current_location": DEFAULT_LOCATION,
            "discovered_clues": [],
            "explored": {loc: [] for loc in LOCATIONS},
            "questions_used": 0,
            "notes" : [],
        }

    def get_location(self) -> str:
        return self._global["current_location"]

    def move_to(self, location: str) -> bool:
        if location not in LOCATIONS:
            return False
        self._global["current_location"] = location
        return True

    def explore_location(self, location: str) -> tuple[str | None, str | None]:
        candidates = LOCATION_CLUES.get(location, [])
        explored = self._global["explored"].setdefault(location, [])
        remaining = [c for c in candidates if c not in explored]
        if not remaining:
            return None, None
        clue = remaining[0]
        explored.append(clue)
        if clue not in self._global["discovered_clues"]:
            self._global["discovered_clues"].append(clue)
        return clue, EVIDENCE_DESCRIPTIONS.get(clue, "")

    def location_progress(self, location: str) -> tuple[int, int]:
        total = len(LOCATION_CLUES.get(location, []))
        done = len(self._global["explored"].get(location, []))
        return done, total

    def total_clue_progress(self) -> tuple[int, int]:
        return len(self._global["discovered_clues"]), len(EVIDENCE_DESCRIPTIONS)

    def _ensure(self, character_id: str) -> dict[str, Any]:
        if character_id not in self._npc_state:
            self._npc_state[character_id] = {
                "presented_clues": [],
                "dialogue_history": [],
                "turn_count": 0,
            }
        return self._npc_state[character_id]

    def get_discovered_clues(self, character_id: str | None = None) -> list[str]:
        return list(self._global["discovered_clues"])

    def add_discovered_clue(self, clue_label: str, character_id: str | None = None) -> None:
        if clue_label not in self._global["discovered_clues"]:
            self._global["discovered_clues"].append(clue_label)

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

    def questions_remaining(self, character_id: str | None = None) -> int:
        return max(0, self._question_limit - self._global["questions_used"])

    def has_questions_left(self, character_id: str | None = None) -> bool:
        return self.questions_remaining() > 0

    def use_question(self, character_id: str | None = None) -> None:
        self._global["questions_used"] = min(
            self._question_limit, self._global["questions_used"] + 1,
        )

    def snapshot(self, character_id: str) -> dict[str, Any]:
        return copy.deepcopy(self._ensure(character_id))

    def reset(self, character_id: str) -> None:
        self._npc_state[character_id] = {
            "presented_clues": [],
            "dialogue_history": [],
            "turn_count": 0,
        }

    def add_note(self, text: str) -> None:
        self._global["notes"].append(text)

    def get_notes(self) -> list[str]:
        return list(self._global["notes"])

    def delete_note(self, index: int) -> bool:
        notes = self._global["notes"]
        if 0 <= index < len(notes):
            notes.pop(index)
            return True
        return False


# ============================================================
# 3. 프롬프트 빌더
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
# 4. 응답 후처리 가드레일
# ============================================================


UNAWARE_MARKERS = ["몰라", "모르", "처음 듣", "글쎄", "잘 모르겠", "본 적 없", "안 가"]

_HANJA_PATTERN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


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
# 5. LLM 호출 — Qwen / OpenAI / Anthropic 멀티 백엔드
# ============================================================
# call_llm()은 겉에서 보면 백엔드가 뭐든 항상 똑같은 함수다.
# LLM_BACKEND 값에 따라 내부에서 _call_qwen / _call_openai / _call_claude로
# 나뉘어 갈 뿐, 프롬프트 빌더·가드레일·오케스트레이션은 이 함수를 그대로 쓰면 된다.

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
    # 캐릭터가 4명이라 이름표 제거 정규식도 PROFILES에 등록된 모든 이름을 대상으로 한다.
    name_pattern = "|".join(re.escape(name) for name in PROFILES)
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
    """세 백엔드 모두 동일하게 쓰는 '이번 질문' 메시지 포맷.
    매 턴마다 [아는 사실]만 근거로 답하라는 지시를 한 번 더 얹는다."""
    return (
        f"플레이어 질문:\n\n{player_message}\n\n"
        "위 질문에 대해\n"
        "반드시 시스템 프롬프트의 [아는 사실]만 근거로 답하세요.\n"
        "모르는 내용은 절대 추측하지 마세요."
    )


def call_llm(system_prompt: str, dialogue_history: list[dict[str, Any]], player_message: str) -> str:
    if LLM_BACKEND == "qwen":
        return _call_qwen(system_prompt, dialogue_history, player_message)
    if LLM_BACKEND == "openai":
        return _call_openai(system_prompt, dialogue_history, player_message)
    if LLM_BACKEND == "anthropic":
        return _call_claude(system_prompt, dialogue_history, player_message)
    raise ValueError(f"알 수 없는 LLM_BACKEND: {LLM_BACKEND!r}")


# ------------------------------------------------------------
# Qwen (로컬 4-bit)
# ------------------------------------------------------------
@torch.inference_mode()
def _call_qwen(system_prompt: str, dialogue_history: list[dict[str, Any]], player_message: str) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(normalize_history(dialogue_history))
    messages.append({"role": "user", "content": build_user_turn(player_message)})

    model_inputs = TOKENIZER.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    model_inputs = {k: v.to(MODEL.device) for k, v in model_inputs.items()}

    terminators = [TOKENIZER.eos_token_id]
    im_end_id = TOKENIZER.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in terminators:
        terminators.append(im_end_id)

    generated = MODEL.generate(
        **model_inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.65,
        top_p=0.80,
        top_k=35,
        repetition_penalty=1.15,
        eos_token_id=terminators,
        pad_token_id=TOKENIZER.pad_token_id,
    )

    input_length = model_inputs["input_ids"].shape[-1]
    new_tokens = generated[0][input_length:]
    reply = TOKENIZER.decode(new_tokens, skip_special_tokens=True)
    return clean_reply(reply)


# ------------------------------------------------------------
# OpenAI (GPT)
# ------------------------------------------------------------
def _call_openai(system_prompt: str, dialogue_history: list[dict[str, Any]], player_message: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. 1번 셀에서 설정하세요.")

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


# ------------------------------------------------------------
# Anthropic (Claude)
# ------------------------------------------------------------
def _call_claude(system_prompt: str, dialogue_history: list[dict[str, Any]], player_message: str) -> str:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY가 설정되지 않았습니다. 1번 셀에서 설정하세요.")

    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    # Claude API는 system을 messages 배열이 아니라 별도 파라미터로 받는다.
    messages = normalize_history(dialogue_history)
    messages.append({"role": "user", "content": build_user_turn(player_message)})

    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        system=system_prompt,
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature = 0.65,
    )
    text = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")
    return clean_reply(text)


# ============================================================
# 6. 오케스트레이션
# ============================================================

QUESTIONS_EXHAUSTED_MESSAGE = (
    "질문권을 모두 사용하셨어요... 더 이상 심문할 수 없습니다. "
    "('초기화' 후 다시 시작해 주세요.)"
)

# 4명 전원이 PROFILES(LLM 캐릭터)라서 규칙 기반 분기가 따로 없다.
# handle_player_message는 단일 LLM 경로만 탄다.
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


# ============================================================
# 7. 파이프라인 인터페이스 — pygame이 직접 호출하는 함수들
# ============================================================
# 이 아래 함수들은 input()/print()를 전혀 쓰지 않는다. 매개변수로 값을 받고
# 결과를 값으로 돌려줄 뿐이다. pygame 이벤트 루프와 콘솔 모드(다음 셀)가
# 이 함수들을 똑같이 호출한다 — 로직이 두 군데로 갈라지지 않는다.
# QUESTIONS_EXHAUSTED_MESSAGE는 오케스트레이션 셀(handle_player_message)에서
# 이미 정의했으므로 여기서는 그대로 재사용한다.


def ask_npc(session: GameSession, player_message: str) -> str:
    """현재 위치의 NPC에게 자연어로 질문한다."""
    npc_id = LOCATION_NPC.get(session.get_location())
    if not npc_id:
        return "지금 이 장소에는 심문할 사람이 없습니다."
    return handle_player_message(npc_id, player_message, session)


def move_to(session: GameSession, location: str) -> dict:
    """다른 장소로 이동한다."""
    if not session.move_to(location):
        return {"ok": False, "error": f"'{location}'은(는) 존재하지 않는 장소입니다."}
    npc_id = LOCATION_NPC.get(location)
    return {
        "ok": True,
        "location": location,
        "npc_id": npc_id,
        "npc_name": npc_id,
    }


def explore_here(session: GameSession) -> dict:
    """현재 장소를 탐색해 단서를 하나 발견한다 (더 없으면 found=False)."""
    loc = session.get_location()
    clue, desc = session.explore_location(loc)
    return {
        "found": clue is not None,
        "clue": clue,
        "description": desc,
        "clue_progress": session.total_clue_progress(),
        "location_progress": session.location_progress(loc),
    }


def get_state(session: GameSession) -> dict:
    """pygame이 화면에 그릴 현재 상태 스냅샷."""
    loc = session.get_location()
    npc_id = LOCATION_NPC.get(loc)
    return {
        "location": loc,
        "npc_id": npc_id,
        "npc_name": npc_id,
        "discovered_clues": session.get_discovered_clues(),
        "clue_progress": session.total_clue_progress(),
        "location_progress": session.location_progress(loc),
        "questions_remaining": session.questions_remaining(),
        "question_limit": QUESTION_LIMIT,
        "llm_backend": LLM_BACKEND,
    }










# ------------------------------------------------------------
# 사건 재구성(storyboard) — 캐릭터가 실제로 걸어다니며 사건을 보여주는 장면들.
# true_storyboard는 CASE_SUMMARY_TEXT를 장면 단위로 고정해 둔 것이고,
# player_storyboard는 플레이어가 제출한 추리를 바탕으로 매번 새로 생성한다.
# ------------------------------------------------------------
TRUE_STORYBOARD: list[dict[str, str]] = [
    {
        "character": "이찬형",
        "location": "원장실",
        "action": "이른 아침 원장실 금고를 열어 시험지 봉투를 개봉하고, 지문을 닦아 흔적을 지운다.",
    },
    {
        "character": "이찬형",
        "location": "인쇄실",
        "action": "인쇄실 복사기로 개봉한 시험지를 몰래 복사한다.",
    },
    {
        "character": "이찬형",
        "location": "상담실",
        "action": "상담실 모니터로 학부모와 대화하며 대가로 받을 계좌번호를 주고받는다.",
    },
    {
        "character": "이찬형",
        "location": "복도",
        "action": "서두르는 걸음으로 복도를 지나가다 등원한 김준이와 마주친다.",
    },
]

STORYBOARD_MAX_TOKENS = 500


def _storyboard_llm_call(system_prompt: str, user_prompt: str) -> str:
    """스토리보드 생성 전용 LLM 호출.
    대화용 call_llm과 달리 clean_reply의 개행 절단 등 대사용 후처리를 하지 않고
    모델 원문을 그대로 반환한다 (JSON 파싱은 호출부에서 처리)."""
    if LLM_BACKEND == "qwen":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        model_inputs = TOKENIZER.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        model_inputs = {k: v.to(MODEL.device) for k, v in model_inputs.items()}
        terminators = [TOKENIZER.eos_token_id]
        im_end_id = TOKENIZER.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in terminators:
            terminators.append(im_end_id)
        with torch.inference_mode():
            generated = MODEL.generate(
                **model_inputs,
                max_new_tokens=STORYBOARD_MAX_TOKENS,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=terminators,
                pad_token_id=TOKENIZER.pad_token_id,
            )
        input_length = model_inputs["input_ids"].shape[-1]
        new_tokens = generated[0][input_length:]
        return TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()

    if LLM_BACKEND == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=STORYBOARD_MAX_TOKENS,
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()

    if LLM_BACKEND == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY가 설정되지 않았습니다.")
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=ANTHROPIC_MODEL,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=STORYBOARD_MAX_TOKENS,
            temperature=0.3,
        )
        return "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        ).strip()

    raise ValueError(f"알 수 없는 LLM_BACKEND: {LLM_BACKEND!r}")


def _parse_storyboard_json(raw_text: str) -> list | None:
    match = re.search(r"\[.*\]", raw_text, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = _json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, list) else None


def _validate_storyboard(steps: list | None) -> list[dict[str, str]]:
    """존재하지 않는 캐릭터/장소를 가리키는 단계는 걸러낸다."""
    if not steps:
        return []
    valid: list[dict[str, str]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        character = step.get("character")
        location = step.get("location")
        action = step.get("action")
        if character not in CHARACTER_NAMES:
            continue
        if location not in LOCATIONS:
            continue
        if not isinstance(action, str) or not action.strip():
            continue
        valid.append({"character": character, "location": location, "action": action.strip()[:60]})
    return valid


def _rule_based_player_storyboard(
    culprit: str, evidence: str, discovered_clues: list[str],
) -> list[dict[str, str]]:
    """LLM 결과가 없거나 파싱/검증에 실패했을 때 쓰는 최소한의 규칙 기반 폴백.
    플레이어가 지목한 이름이 실제 등장인물과 일치하면 그 인물을, 아니면 정답 범인을 주인공으로 삼는다.
    가능하면 플레이어가 제시한 증거/발견된 단서와 그 단서가 있던 장소를 엮어 최소한의 정황을 보여준다."""
    accused = next((name for name in CHARACTER_NAMES if name and name in culprit), None)
    if accused is None:
        accused = CASE_CONFIG["gold"]["culprit"]["aliases"][0]

    mentioned = [
        clue for clue in discovered_clues if contains_any(evidence, CLUE_KEYWORDS.get(clue, [clue]))
    ]
    steps: list[dict[str, str]] = []
    for clue in (mentioned or discovered_clues[:2]):
        location = next((loc for loc, clues in LOCATION_CLUES.items() if clue in clues), None)
        if location:
            steps.append({"character": accused, "location": location, "action": f"{clue}과(와) 관련된 정황."})

    if not steps:
        steps = [{"character": accused, "location": "복도", "action": "혐의를 받고 있다."}]
    return steps


def build_player_storyboard(
    culprit: str, motive: str, evidence: str, explanation: str, discovered_clues: list[str],
) -> list[dict[str, str]]:
    system_prompt = (
        "당신은 추리 게임 '봉인된 모의고사'의 내레이터입니다. "
        "플레이어가 제출한 최종 추리를 바탕으로, '플레이어의 추리가 맞다면 사건이 어떻게 "
        "벌어졌을지'를 인물이 장소를 옮겨 다니는 3~5개의 장면으로 재구성하세요.\n\n"
        f"등장 가능한 인물: {', '.join(CHARACTER_NAMES)}\n"
        f"등장 가능한 장소: {', '.join(LOCATIONS)}\n\n"
        "반드시 아래 형식의 JSON 배열만 출력하세요. 다른 설명, 인사말, 코드블록 기호 없이 "
        "JSON 배열 하나만 출력해야 합니다:\n"
        '[{"character": "인물 이름", "location": "장소 이름", "action": "그 장소에서 하는 행동(한 문장)"}, ...]'
    )
    user_prompt = (
        f"[플레이어가 지목한 범인]\n{culprit}\n\n"
        f"[플레이어가 생각하는 동기]\n{motive}\n\n"
        f"[플레이어가 제시한 증거]\n{evidence}\n\n"
        f"[플레이어의 설명]\n{explanation}\n\n"
        "위 추리가 사실이라고 가정하고, 시간 순서대로 장면을 JSON 배열로 작성하세요."
    )

    steps: list[dict[str, str]] = []
    if not FORCE_FALLBACK_STORYBOARD:
        try:
            raw = _storyboard_llm_call(system_prompt, user_prompt)
            steps = _validate_storyboard(_parse_storyboard_json(raw))
        except Exception:
            steps = []

    if not steps:
        steps = _rule_based_player_storyboard(culprit, evidence, discovered_clues)

    return steps


def _fixed_grade_a_result() -> dict[str, Any]:
    """FORCE_GRADE_A 테스트 모드에서 쓰는 고정 채점 결과.
    total(85)이 grade_for()에서 실제로 "A"가 되는 값이라 결과 화면 표시가 어긋나지 않는다.
    LLM을 전혀 호출하지 않으므로 score_report(세부 판정)는 없다."""
    total = 85.0
    return {
        "total": total,
        "max_total": 100.0,
        "percentage": 85.0,
        "grade": grade_for(total),
        "feedback": (
            "[FORCE_GRADE_A 테스트 모드] 실제 채점 파이프라인(LLM 호출) 없이 "
            "고정 85점(A등급) 결과를 반환합니다."
        ),
        "breakdown": [],
        "score_report": None,
    }


def _build_score_breakdown(score_report):
    """ui.draw_result()가 그대로 순회해서 그릴 수 있도록 6개 채점 항목을
    고정된 순서(범인 → 시간 → 장소 → 동기 → 범행 과정 → 단서 연결)로 정리한다."""
    if not score_report:
        return []
    explicit = score_report.get("explicit", {})
    semantic = score_report.get("semantic", {})
    order = (
        ("explicit", "culprit"), ("explicit", "time"), ("explicit", "location"),
        ("semantic", "motive"), ("semantic", "method"), ("semantic", "evidence_links"),
    )
    breakdown = []
    for section, key in order:
        item = (explicit if section == "explicit" else semantic).get(key)
        if not item:
            continue
        breakdown.append({
            "label": item["label"],
            "earned": item["earned"],
            "max_points": item["max_points"],
        })
    return breakdown


def _build_user_deduction_text(culprit: str, motive: str, evidence: str, explanation: str) -> str:
    """main.py(pygame UI)는 motive/evidence/explanation을 이미 explanation 하나로 합쳐서
    넘기고, 콘솔 모드는 네 항목을 따로 입력받는다. 두 경우 모두 지원하기 위해 중복되는
    문자열은 한 번만 남기고 이어 붙여 evaluate_deduction()이 기대하는 자유 서술
    user_text 하나로 만든다."""
    parts = []
    seen = set()
    for text in (culprit, motive, evidence, explanation):
        cleaned = (text or "").strip()
        if cleaned and cleaned not in seen:
            parts.append(cleaned)
            seen.add(cleaned)
    return "\n".join(parts)


def submit_reasoning(
    session: GameSession, culprit: str, motive: str, evidence: str, explanation: str,
) -> dict:
    """플레이어의 최종 추리를 채점한다."""
    discovered_clues = session.get_discovered_clues()
    if FORCE_GRADE_A:
        result = _fixed_grade_a_result()
    else:
        user_text = _build_user_deduction_text(culprit, motive, evidence, explanation)
        # debug=True로 호출해 explicit/semantic 세부 판정(details)까지 받는다 —
        # 결과 화면에 6개 항목별 점수를 보여주려면 이 세부 정보가 필요하다.
        eval_result = evaluate_deduction(user_text, llm_backend=GRADING_LLM, debug=True)
        score_report = (eval_result.details or {}).get("score_report", {})
        result = {
            "total": eval_result.score,
            "max_total": eval_result.max_score,
            "percentage": eval_result.percentage,
            "grade": grade_for(eval_result.score),
            "feedback": eval_result.feedback,
            "breakdown": _build_score_breakdown(score_report),
            "score_report": eval_result.details,
        }
    result["case_summary"] = CASE_SUMMARY_TEXT
    result["player_storyboard"] = build_player_storyboard(
        culprit, motive, evidence, explanation, discovered_clues,
    )
    result["true_storyboard"] = TRUE_STORYBOARD
    return result


