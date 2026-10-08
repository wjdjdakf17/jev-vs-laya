# 08. Laya — 오픈소스 Jev 대안 실측

> 2026-10-08 실시. [Convai Innovations의 Laya](https://huggingface.co/convaiinnovations/laya)는
> Jev와 같은 계약(상태 + 타입화된 질문 → 확률이 붙은 답)을 무료 로컬로 돌리는
> 오픈소스 모델이다. 이 레포의 한국어 티켓 17개로 **Laya vs GPT-5** 실측 비교.

## Laya란

- ModernBERT 인코더 + 결정 헤드, **비자기회귀(non-autoregressive)** — 텍스트를 생성하지 않는다
- `choice` / `score` / `noul` — 우리 `src/types.ts`의 3종과 1:1
- ONNX 1.7GB, 로컬 CPU 실행 (Apple Silicon에서 질문 배치당 ~140ms), 런타임에 PyTorch 불필요
- 실행은 [@receptron/laya](https://github.com/receptron/laya)(Node.js)로 — 이 레포 언어 그대로

## 실험 설계

`data/tickets/test.jsonl` 17개(애매한 티켓 5개 포함)를 **둘 다 제로샷**으로:

- Laya: `systemOne(context, { route: choice })` — criteria에 팀 설명 한 줄씩
- GPT-5: JSON 모드로 `{team, confidence}` 반환 요청 (OpenAI API)
- 워밍업 1회 후 측정. 스크립트: [examples/laya-vs-gpt.ts](../examples/laya-vs-gpt.ts)

## 결과 (2026-10-08 실측, Apple Silicon CPU)

| | **Laya** (로컬) | **GPT-5** (API) | 참고: jevlike 학습모델 |
| --- | --- | --- | --- |
| top-1 정확도 | 11/17 (**64.7%**) | 13/17 (**76.5%**) | 15/17 (**88.2%**)¹ |
| 지연 (평균) | **119ms** | 3,118ms | 수십 ms |
| 지연 (최대) | 137ms | 5,112ms | — |
| 토큰 (17티켓) | in 3,736 · out **0** | in 1,404 · out **3,287** | — |
| 비용 (17티켓) | **$0** | **$0.035**² | $0 |
| 비용 (100만 티켓 환산) | **$0** | ~$2,037 | $0 |

¹ [07](07-hands-on-local-training.md)의 같은 시험지, 72행 학습 후.
² gpt-5 $1.25/M 입력 · $10/M 출력 ([OpenAI 가격](https://developers.openai.com), 2026-10 기준). 출력 3,287토큰은 JSON 17개 치고 많다 — 추론(reasoning) 토큰이 포함된 생성 비용. **결정 하나를 내리는 데 생성을 돈 주고 사는** 구조가 수치로 드러난다.

## 읽을 거리

**→ 속도·비용은 압도적, 정확도는 제로샷 한계.** Laya는 26배 빠르고 공짜지만,
이 한국어 티켓 분류를 본 적 없는 제로샷이라 64.7%. GPT도 제로샷이라 76.5%.
**도메인 72행을 학습한 0.6B짜리 jevlike가 둘 다 이긴다(88.2%)** —
"결정 태스크는 프롬프트보다 학습"이라는 [07](07-hands-on-local-training.md)의
결론이 프런티어 LLM에도 그대로 성립한다.

**→ 보정의 두 얼굴.** Laya의 분포는 솔직하지만 플랫하다(1위 40% · 2위 29% 수준) —
확신이 없다는 걸 확률로 그대로 말한다. GPT의 self-confidence는 틀린 답에도
80~90%를 붙인다(오답 4개 중 3개가 confidence 86~90%) — **자기 보고 confidence는
보정이 아니라 감**이다. 이게 [02](02-how-it-works.md)에서 공부한
confidence vs calibration의 실측 예시.

**→ 그래서 confidence 게이트가 있다.** Laya의 플랫한 분포는
`workflow.ts`의 게이트(임계값 미만 → 사람)에 걸리는 게 정상 동작이다.
틀리면서 90%를 부르짖는 쪽이 아니라, 망설이는 쪽이 시스템에 더 안전하다.

## 언제 Laya를 쓰고 안 쓰나

| 상황 | 선택 |
| --- | --- |
| 고정 옵션 반복 결정, 대량 처리, 지연/비용 민감 | **Laya** (또는 파인튜닝 소형) |
| 도메인 데이터가 있고 최고 정확도 필요 | **파인튜닝** ([07](07-hands-on-local-training.md)) |
| 열린 질문, 설명 필요, 처음 보는 상황 | **LLM** |
| 둘 다 필요 (설명하는 결정) | LLM 요약 + 결정은 별도 — [04](04-use-cases-limits.md)의 분업 |

## 재현

```bash
npm install                      # @receptron/laya 포함
python3 scripts/make_ticket_data.py   # data/tickets/ 생성 (seed 고정)
cp ~/Desktop/saju/.env.local .   # 또는 OPENAI_API_KEY 직접 — .env.local은 git 제외
npx tsx examples/laya-vs-gpt.ts  # 첫 실행 시 ONNX 1.7GB 자동 다운로드
```

→ [07. 로컬 학습으로 돌아가기](07-hands-on-local-training.md)
