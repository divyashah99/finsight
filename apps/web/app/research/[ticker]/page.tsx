"use client";

import { AgentTimeline } from "@/components/agent-timeline";
import { MemoViewer } from "@/components/memo-viewer";
import type { AgentEvent, AgentName, Memo } from "@/lib/types";
import Link from "next/link";
import { use, useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const ORDER: AgentName[] = ["market", "quant", "news", "sec", "writer"];

export default function ResearchPage({ params }: { params: Promise<{ ticker: string }> }) {
  const { ticker: rawTicker } = use(params);
  const ticker = rawTicker.toUpperCase();
  const [memo, setMemo] = useState("");
  const [structuredMemo, setStructuredMemo] = useState<Memo | null>(null);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [finalId, setFinalId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const url = `${API_URL}/research/${ticker}`;
    console.log("[SSE] opening EventSource:", url);
    const es = new EventSource(url);

    es.addEventListener("open", () => console.log("[SSE] open"));

    es.addEventListener("agent_start", (e: MessageEvent) => {
      console.log("[SSE] agent_start", e.data);
      const p = JSON.parse(e.data);
      setEvents((prev) => upsert(prev, p.agent, { status: "running" }));
    });

    es.addEventListener("agent_done", (e: MessageEvent) => {
      console.log("[SSE] agent_done", e.data);
      const p = JSON.parse(e.data);
      setEvents((prev) => {
        const updated = upsert(prev, p.agent, { status: "done", summary: p.summary });
        const idx = ORDER.indexOf(p.agent);
        const next = ORDER[idx + 1];
        if (next) return upsert(updated, next, { status: "running" });
        return updated;
      });
    });

    es.addEventListener("final", (e: MessageEvent) => {
      console.log("[SSE] final");
      const p = JSON.parse(e.data);
      setFinalId(p.report_id);
      if (p.memo) {
        setStructuredMemo(p.memo as Memo);
        if (p.memo.markdown) setMemo(p.memo.markdown);
      }
      es.close();
    });

    es.addEventListener("error", (e: Event) => {
      // EventSource fires "error" on normal close too — only flag if not closed.
      if (es.readyState === EventSource.CLOSED) {
        console.log("[SSE] closed");
      } else {
        console.error("[SSE] error event", e);
        setError("Connection error");
      }
    });

    return () => {
      console.log("[SSE] cleanup: closing");
      es.close();
    };
  }, [ticker]);

  return (
    <main className="mx-auto max-w-6xl px-4 py-8">
      <header className="mb-6 flex items-center justify-between">
        <Link href="/" className="text-sm text-neutral-500 hover:text-neutral-200">
          ← FinSight
        </Link>
        <div className="text-right">
          <div className="font-mono text-3xl font-semibold">{ticker}</div>
          {finalId && (
            <div className="text-xs text-neutral-600">report {finalId.slice(0, 8)}</div>
          )}
        </div>
      </header>

      {error && (
        <div className="mb-4 rounded-md border border-rose-900 bg-rose-950/40 p-3 text-sm text-rose-300">
          {error}
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
        <aside>
          <AgentTimeline events={events} />
        </aside>
        <section>
          <MemoViewer markdown={memo} memo={structuredMemo} />
        </section>
      </div>
    </main>
  );
}

function upsert(events: AgentEvent[], agent: AgentName, patch: Partial<AgentEvent>): AgentEvent[] {
  const existing = events.find((e) => e.agent === agent);
  if (existing) {
    return events.map((e) => (e.agent === agent ? { ...e, ...patch } : e));
  }
  return [...events, { agent, status: "running", ...patch } as AgentEvent];
}
