/**
 * examples/laya-vs-gpt.ts — Laya(로컬, 무료) vs GPT-5(OpenAPI API) 실측 비교.
 *
 * 같은 한국어 티켓 17개(test.jsonl)를 두 모델에 던져 측정한다:
 *   - 정확도 (top-1)
 *   - 지연시간 (평균/최대, ms)
 *   - 토큰 사용량 (입력/출력)
 *   - 확률 분포의 모양 (보정 관점 — Laya는 모델 확률, GPT는 자기 보고 confidence)
 *
 * 실행: npx tsx examples/laya-vs-gpt.ts
 * 사전: .env.local에 OPENAI_API_KEY, data/tickets/test.jsonl (scripts/make_ticket_data.py)
 * 첫 실행 시 Laya ONNX 가중치(~1.7GB)를 HF에서 내려받는다.
 */
import { readFileSync } from "node:fs";
import { Laya } from "@receptron/laya";

const TEAMS = ["billing", "technical", "account"] as const;
type Row = { context: string; label: number };

const rows: Row[] = readFileSync("data/tickets/test.jsonl", "utf8")
  .trim()
  .split("\n")
  .map((line) => JSON.parse(line) as Row);

// .env.local 수동 로드 (dotenv 의존성 없이)
const env = Object.fromEntries(
  readFileSync(".env.local", "utf8")
    .split("\n")
    .filter((l) => l.includes("=") && !l.startsWith("#"))
    .map((l) => [l.slice(0, l.indexOf("=")), l.slice(l.indexOf("=") + 1).trim()]),
);
const OPENAI_KEY = env["OPENAI_API_KEY"];
const OPENAI_MODEL = env["OPENAI_MODEL"] ?? "gpt-5";
if (!OPENAI_KEY) throw new Error(".env.local에 OPENAI_API_KEY가 없다");

const CRITERIA = {
  billing: "결제, 환불, 청구, 구독 요금 문제",
  technical: "버그, 오류, 앱/기능 작동 문제",
  account: "계정, 로그인, 탈퇴, 개인정보 문제",
} as const;

type Result = { pick: string; latencyMs: number; inTok: number; outTok: number; detail: string };

/* ───────── 1) Laya ───────── */
async function runLaya(): Promise<Result[]> {
  const laya = await Laya.load();
  const out: Result[] = [];
  // 워밍업 1회 (첫 추론의 로드 비용 제거)
  await laya.systemOne("워밍업", { w: { type: "choice", instructions: "팀?", criteria: CRITERIA } });

  for (const row of rows) {
    const t0 = performance.now();
    const r = await laya.systemOne(row.context, {
      route: { type: "choice", instructions: "이 지원 티켓을 처리할 팀은?", criteria: CRITERIA },
    });
    const latencyMs = performance.now() - t0;
    const a = r.answers.route;
    if (a.type !== "choice") throw new Error("choice 응답 아님");
    const probs = Object.entries(a.probabilities)
      .map(([k, v]) => `${k} ${(v * 100).toFixed(1)}%`)
      .join(" · ");
    out.push({
      pick: a.choice,
      latencyMs,
      inTok: r.usage.input_tokens,
      outTok: 0, // Laya는 생성이 없다 — 출력 토큰 개념 없음
      detail: probs,
    });
  }
  await laya.close();
  return out;
}

/* ───────── 2) GPT-5 ───────── */
async function runGpt(): Promise<Result[]> {
  const out: Result[] = [];
  for (const row of rows) {
    const t0 = performance.now();
    const res = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${OPENAI_KEY}` },
      body: JSON.stringify({
        model: OPENAI_MODEL,
        messages: [
          {
            role: "system",
            content:
              "지원 티켓을 분류하는 분류기다. 반드시 JSON만 출력한다: {\"team\": \"billing|technical|account\", \"confidence\": 0.0~1.0}",
          },
          { role: "user", content: row.context },
        ],
        response_format: { type: "json_object" },
      }),
    });
    if (!res.ok) throw new Error(`OpenAI ${res.status}: ${(await res.text()).slice(0, 200)}`);
    const j = (await res.json()) as {
      choices: { message: { content: string } }[];
      usage: { prompt_tokens: number; completion_tokens: number };
    };
    const latencyMs = performance.now() - t0;
    const parsed = JSON.parse(j.choices[0].message.content) as { team: string; confidence: number };
    out.push({
      pick: parsed.team,
      latencyMs,
      inTok: j.usage.prompt_tokens,
      outTok: j.usage.completion_tokens,
      detail: `self-confidence ${(parsed.confidence * 100).toFixed(0)}%`,
    });
  }
  return out;
}

/* ───────── 3) 리포트 ───────── */
function report(name: string, results: Result[]): void {
  const correct = results.filter((r, i) => r.pick === TEAMS[rows[i].label]).length;
  const lat = results.map((r) => r.latencyMs);
  const inTok = results.reduce((s, r) => s + r.inTok, 0);
  const outTok = results.reduce((s, r) => s + r.outTok, 0);
  console.log(`\n===== ${name} =====`);
  console.log(`정확도: ${correct}/${results.length} (${((correct / results.length) * 100).toFixed(1)}%)`);
  console.log(`지연: 평균 ${(lat.reduce((a, b) => a + b, 0) / lat.length).toFixed(0)}ms · 최대 ${Math.max(...lat).toFixed(0)}ms · 최소 ${Math.min(...lat).toFixed(0)}ms`);
  console.log(`토큰: 입력 ${inTok} · 출력 ${outTok} (17티켓 합계)`);
  results.forEach((r, i) => {
    const ok = r.pick === TEAMS[rows[i].label] ? "O" : "X";
    console.log(`  [${ok}] ${r.pick.padEnd(10)} ${r.latencyMs.toFixed(0).padStart(5)}ms  ${r.detail}  | ${rows[i].context.split("\n")[0].slice(0, 28)}`);
  });
}

const layaResults = await runLaya();
report("Laya (로컬, 무료, 1.7GB ONNX)", layaResults);

const gptResults = await runGpt();
report(`GPT (OpenAI API, ${OPENAI_MODEL})`, gptResults);
