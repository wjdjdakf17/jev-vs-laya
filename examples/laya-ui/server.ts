/**
 * examples/laya-ui/server.ts — Laya vs GPT-5 인터랙티브 데모 서버 (포트 3000).
 *
 * 실행: npx tsx examples/laya-ui/server.ts
 * 브라우저: http://localhost:3000
 */
import express from "express";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { Laya } from "@receptron/laya";

const HERE = dirname(fileURLToPath(import.meta.url));

const env = Object.fromEntries(
  readFileSync(join(HERE, "../../.env.local"), "utf8")
    .split("\n")
    .filter((l) => l.includes("=") && !l.startsWith("#"))
    .map((l) => [l.slice(0, l.indexOf("=")), l.slice(l.indexOf("=") + 1).trim()]),
);
const OPENAI_KEY = env["OPENAI_API_KEY"];
const OPENAI_MODEL = env["OPENAI_MODEL"] ?? "gpt-5";

const CRITERIA = {
  billing: "결제, 환불, 청구, 구독 요금 문제",
  technical: "버그, 오류, 앱/기능 작동 문제",
  account: "계정, 로그인, 탈퇴, 개인정보 문제",
} as const;

console.log("Laya 로딩 중 (첫 실행이면 1.7GB 다운로드)...");
const laya = await Laya.load();
console.log("Laya 준비 완료. 워밍업...");
await laya.systemOne("워밍업", { w: { type: "choice", instructions: "팀?", criteria: CRITERIA } });
console.log("http://localhost:3000 에서 실행 중");

const app = express();
app.use(express.json());
app.get("/", (_req, res) => res.sendFile(join(HERE, "index.html")));

app.post("/api/ask", async (req, res) => {
  const context = String(req.body?.context ?? "").trim();
  if (!context) return res.status(400).json({ error: "티켓 내용이 비었다" });

  // 1) Laya — 로컬 순전파
  const t0 = performance.now();
  const r = await laya.systemOne(context, {
    route: { type: "choice", instructions: "이 지원 티켓을 처리할 팀은?", criteria: CRITERIA },
  });
  const layaMs = performance.now() - t0;
  const a = r.answers.route;
  if (a.type !== "choice") return res.status(500).json({ error: "choice 아님" });

  // 2) GPT — 실제 OpenAI API 호출
  const t1 = performance.now();
  let gpt: object;
  try {
    const rr = await fetch("https://api.openai.com/v1/chat/completions", {
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
          { role: "user", content: context },
        ],
        response_format: { type: "json_object" },
      }),
    });
    if (!rr.ok) throw new Error(`OpenAI ${rr.status}`);
    const j = (await rr.json()) as {
      choices: { message: { content: string } }[];
      usage: { prompt_tokens: number; completion_tokens: number };
    };
    const parsed = JSON.parse(j.choices[0].message.content) as { team: string; confidence: number };
    gpt = {
      team: parsed.team,
      confidence: parsed.confidence,
      latencyMs: performance.now() - t1,
      promptTokens: j.usage.prompt_tokens,
      completionTokens: j.usage.completion_tokens,
    };
  } catch (e) {
    gpt = { error: String(e) };
  }

  res.json({
    laya: {
      choice: a.choice,
      probabilities: a.probabilities,
      latencyMs: layaMs,
      inputTokens: r.usage.input_tokens,
    },
    gpt,
  });
});

app.listen(3000);
