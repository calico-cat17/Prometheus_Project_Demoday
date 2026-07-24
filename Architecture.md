## 시스템 아키텍처 구상

[게임 클라이언트 — 2D 공간 탐색]
        │
        │ 플레이어가 특정 NPC 위치로 이동해 말을 걺
        │ 
        ▼
[게임 서버]
        │
        ├─ 공유 게임 상태 조회
        │    - 지금까지 발견된 단서 목록
        │    - 이 캐릭터와의 대화 이력
        │    - 압박 질문 누적 횟수 (감정 상태용)
        ▼
[캐릭터 Agent 직접 호출 — 김준이] (LLM 호출 1회)
        │  시스템 프롬프트 = 고정 페르소나
        │                  + 아는 사실 / 모르는 사실 경계
        │                  + 현재 발견된 단서에 대응하는 반응 규칙
        │                  + 대화 이력
        ▼
[응답 후처리 가드레일]
        │  - unknown_facts 노출 여부 체크
        │  - 범인/정답 직접 노출 여부 체크
        │  - 캐릭터 이탈(메타 발언) 체크
        │  실패 시 → 강화된 지시로 1회 재생성
        ▼
[대화 이력 · 감정 상태 갱신] → [플레이어에게 응답 반환]
        │
        │ 플레이어가 특정 NPC 위치로 이동해 말을 걺
        │ 
        ▼
[최종 추리]
          - 플레이어가 제출한 범인, 범행 동기, 과정 등을 종합하여 점수화
          - 플레이어가 제출한 답안과 정답 각각 2d로 재구성하여 범행 당시 상황 영상을 재생


## 수도코드

def handle_player_message(character_id, player_message, session):
    profile = load_profile(character_id)  # kim_junyi.json
    discovered_clues = session.get_discovered_clues()
    dialogue_history = session.get_dialogue_history(character_id)
    pressure_count = session.get_pressure_count(character_id)

    # 1. 발견된 단서 중 이 캐릭터에게 해당하는 반응만 필터링
    active_reactions = [
        profile.clue_reactions[clue]
        for clue in discovered_clues
        if clue in profile.clue_reactions
    ]

    # 2. 다중 단서 조합 상관관계 체크
    for rule in profile.correlation_rules:
        if any(c in discovered_clues for c in rule["trigger_clues"]):
            active_reactions.append(rule["unlocked_reaction"])

    # 3. 감정 상태 계산 (반복 압박 질문 대응)
    emotional_tone = (
        profile.emotional_state_rules["at_or_above_threshold"]
        if pressure_count >= profile.emotional_state_rules["pressure_threshold"]
        else profile.emotional_state_rules["below_threshold"]
    )

    # 4. 시스템 프롬프트 조립
    system_prompt = build_prompt(
        persona=profile.personality,
        known_facts=profile.known_facts,
        unknown_facts=profile.unknown_facts,   # 해당 주제 질문 시 "모른다"로 답하도록 강제
        active_reactions=active_reactions,
        emotional_tone=emotional_tone,
        restricted_topics=profile.restricted_topics
    )

    # 5. LLM 호출 (캐릭터당 1회)
    response = call_llm(system_prompt, dialogue_history, player_message)

    # 6. 가드레일 검증 (실패 시 1회 재생성)
    if violates_guardrail(response, profile.unknown_facts, culprit="이찬형"):
        response = call_llm(system_prompt + STRICT_REMINDER, dialogue_history, player_message)

    # 7. 상태 업데이트
    session.update_dialogue_history(character_id, player_message, response)
    session.update_pressure_count(character_id, player_message)

    return response


## 설계 선정 이유
| 결정 지점 | 채택한 방식 | 검토했지만 채택하지 않은 대안 | 채택 이유 |
| --- | --- | --- | --- |
| NPC 대상 선택 | 직접 호출 (2D 위치 기반) | 오케스트레이터가 매 질문마다 대상 NPC 판단 | 플레이어 위치가 이미 대상을 확정함. 오케스트레이터를 씌우면 호출이 2배로 늘고 지연만 커짐 |
| 발견 단서 반영 | 공유 게임 상태 + 프롬프트 주입 | 진행도 판단 전용 서브에이전트 | 단서 목록 대조는 규칙 기반 매칭으로 충분한 저추론 작업. LLM 왕복 비용 대비 이득이 없음 |
| 거짓말 처리 | 범인 캐릭터에 한해 사전 작성된 위장 진술 데이터 (김준이는 결백이라 해당 없음) | 거짓말 생성 서브에이전트 | 위장 진술은 이미 사건 대본상 고정된 사실. 즉석 생성은 오히려 다른 단서와 모순될 위험을 키움 |
| 사실 노출 방지 | 응답 후 단일 가드레일 검증 | 캐릭터 간 교차검증용 다중 에이전트 | 지금 규모(4인, 텍스트 대화)에서는 단일 검증으로 스포일러/이탈을 충분히 잡아낼 수 있음 |
| 캐릭터 지식 관리 | 캐릭터별 JSON 데이터 계층 분리 | 프롬프트에 사실 하드코딩 | 사건·캐릭터가 늘어나도 같은 템플릿 재사용 가능, 유지보수 용이 |

-> 오케스트레이터 및 서브에이전트는 여러 선택지 중 AI가 골라야 하는 상황에서 값어치를 하는데, 이 게임은 공간 UI와 미리 정의된 규칙 데이터가 그 역할을 대신하고 있어 굳이 필요않다는 생각