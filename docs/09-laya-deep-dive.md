# 09. Laya 깊이 파기 — 벤치마크, 체크포인트, 판단의 내부

> 2026-10-08, [08](08-laya-vs-llm.md) 실험 후 이어진 스터디.
> 오늘의 질문 셋: "공식 성적은?", "우리 실험은 공정했나?", "함수 안에서 판단은
> 어떻게 나오나?" — 전부 코드와 공식 수치로 답한다.

## 1. 공식 벤치마크 — Laya vs Jev (Convai 실측, T4)

| 과제 | Laya | Jev | 읽기 |
| --- | --- | --- | --- |
| AG News 분류 | **0.950** | 0.910 | 라틴 문자 과제에선 이김 |
| typed-decisions 2,000개 (파인튜닝 후) | **0.766** | 0.727 | 학습하면 역전 |
| typed-decisions (제로샷) | 0.362 | — | 다수클래스 베이스라인(0.461)보다 낮음 |
| Banking77 (옵션 77개) | 0.425 | **0.870** | 옵션 많으면 무너짐 |
| ECE (낮을수록 정직) | **0.081** | 0.246 | 보정 3배 우위 |
| 지연 (T4) | **32.8ms** | 236~276ms | 7.8배 빠름 |

모델 카드의 자기 평가 원문: *"Laya is a fast base to specialise, **not a
zero-shot decision engine**"* — [08](08-laya-vs-llm.md)에서 우리가 직접
발견한 결론(제로샷 64.7% < 도메인 학습 88.2%)과 정확히 일치한다.

**정리**: 제로샷 범용성은 Jev/GPT가, 도메인 학습 + 대량 + 보정은 Laya가.
정확도 경쟁은 "제로샷이냐 학습이냐"로 갈린다.

## 2. 우리 실험의 결함 — 영어 체크포인트로 한국어를 돌렸다

Laya는 체크포인트가 3종이다:

| 체크포인트 | 인코더 | 언어 |
| --- | --- | --- |
| 루트 (`@receptron/laya` 기본값) | ModernBERT-large | **영어만** |
| `multilingual` 서브폴더 | mmBERT-base | 100+ 언어 |
| `typed-decisions` | 파인튜닝용 | — |

[08](08-laya-vs-llm.md)과 UI 실험은 **기본값(영어)**으로 한국어 티켓을
돌렸다. 모델 카드에 따르면 영어 체크포인트는 비라틴 문자에서
"정확도 0.000에 confidence 0.952"로 붕괴하는 사례가 있다 — 64.7%가
한국어에서 낮게 나온 원인 후보 1순위. `Laya.load({ subfolder:
"multilingual" })`로 재실측하면 이 숫자는 갱신되어야 한다. **미숙제.**

교훈: 벤치마크 숫자를 비교하기 전에 **어떤 체크포인트인지**부터 확인한다.

## 3. 엔드포인트는 없다 — 함수 호출의 전체 경로

Laya는 네트워크를 다니지 않는다. 유일한 다운로드는 첫 실행 시 HF에서
1.7GB 가중치(`~/.cache/receptron-laya`)를 받은 것뿐, 이후 완전 오프라인.

```
브라우저 ──POST /api/ask──▶ localhost:3000 (express)
                               │ laya.systemOne(state, questions)   ← 함수 호출
                               │ buildSequence: state+질문+옵션 → 토큰 시퀀스 1개
                               │ session.run({ input_ids, attention_mask,
                               │               marker_pos, marker_mask, qtype })
                               ▼
                     onnxruntime-node가 CPU에서 forward 1회
                               │ logits → softmax(logits/온도)
                               ▼
                     { choice, probabilities, confidence } 반환
```

포인트: 엔드포인트는 **클라이언트-서버 경계가 있을 때만** 존재한다.
모델이 내 프로세스 RAM에 있으면 경계 자체가 없다. `laya.systemOne()`은
`JSON.parse()`와 같은 종류의 호출이다 — 안의 로직이 손으로 쓴 코드 대신
신경망 가중치일 뿐.

## 4. 판단은 "작문이 아니라 채점"

`systemOne` 내부 (소스: `@receptron/laya/dist/laya.js`):

1. **합치기** — 티켓 + 질문 + 각 옵션 설명을 하나의 토큰 시퀀스로.
   옵션 위치는 `marker_pos` 텐서로 표시
2. **벡터화** — ModernBERT 인코더가 전 토큰을 벡터로. 문맥과 옵션이
   서로를 바라봄 (어텐션)
3. **점수** — 옵션 벡터 × 문맥 표현 내적 → 옵션당 logit 1개
4. **확률화** — `softmax(logits / temp)`, temp는 옵션별 캘리브레이션 온도

응답 타입(`types.d.ts`)에서 눈여겨볼 한 줄:

```ts
confidence: number;   // 1 - normalized entropy
```

confidence가 확률 분포에서 **계산된** 값이다. GPT의 self-confidence
(틀린 답에 86~90%를 붙이던)는 모델이 쓴 감이고, 이건 수학이다.

그리고 "케이스가 무한한데 어떻게 판단하나"의 답: 모델 파일은 사례집이
아니라 **함수의 파라미터**다. 학습 사례 수백만 개는 버려지고, 패턴만
수억 파라미터에 굳는다. 새 입력은 그 함수를 통과할 뿐 — 저장된 케이스와
비교하는 게 아니다. 그래서 도메인이 어긋나면 (우리 제로샷처럼) 점수가
플랫해진다.

## 5. 대조 — jevlike 학습은 이 판단 함수를 우리 손으로 굳히는 것

`jevlike/model.py`와 `train.py`의 전부 (학습 95줄, 모델 65줄):

```python
# 판단: 옵션 벡터(query)가 문맥(key/value)에서 관련 부분을 읽고 내적 점수
query  = self.query(options)
scores = torch.einsum("bnr,blr->bnl", query, self.key(context)) / sqrt(rank)
logits = (query * attended).sum(-1)

# 학습: 정답 옵션의 점수를 올리는 방향으로 가중치 조정
loss = F.cross_entropy(model(batch), batch["labels"])
loss.backward(); optimiser.step()
```

72행 × 60에폭 → top-1 88.2%. Laya의 Convai가 수백만 사례로 한 일과
정확히 같은 연산을, 우리 도메인 72행으로 직접 돌린 것이 [07](07-hands-on-local-training.md)이다.

## 6. 세 배포 형태 — 오늘의 지도

| | 형태 | 비용 | 강점 |
| --- | --- | --- | --- |
| **Jev** | 벤더 API (호스팅) | 호출당 | 프런티어 제로샷, 유지보수 0 |
| **Laya** | 가중치 다운로드 → 로컬 | $0 | 오프라인·프라이버시·무제한, 보정 우위 |
| **jevlike** | 직접 학습 | $0 | 도메인 특화 (이 과제에선 최강) |

## 남은 숙제

- [ ] multilingual 체크포인트로 08 재실측 → 숫자 갱신
- [ ] Laya에 우리 72행 파인튜닝이 가능한지 (오픈소스니까 이론상 가능) → jevlike 88.2%와 비교

→ [08. Laya vs GPT-5 실측으로 돌아가기](08-laya-vs-llm.md)
