"use client";

import { AgentTimeline } from "@/components/agent-timeline";
import { MemoChat } from "@/components/memo-chat";
import { MemoViewer } from "@/components/memo-viewer";
import type { Memo, TimelineStep } from "@/lib/types";
import Link from "next/link";
import { use, useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function ResearchPage({ params }: { params: Promise<{ ticker: string }> }) {
  const { ticker: rawTicker } = use(params);
  const ticker = rawTicker.toUpperCase();
  const [structuredMemo, setStructuredMemo] = useState<Memo | null>(null);
  const [steps, setSteps] = useState<TimelineStep[]>([]);
  const [finalId, setFinalId] = useState<string | null>(null);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const es = new EventSource(`${API_URL}/research/${ticker}`);

    es.addEventListener("supervisor_decision", (e: MessageEvent) => {
      const p = JSON.parse(e.data);
      setSteps((prev) => [...prev, { kind: "decision", next: p.next, reason: p.reason }]);
    });

    es.addEventListener("specialist_done", (e: MessageEvent) => {
      const p = JSON.parse(e.data);
      setSteps((prev) => [
        ...prev,
        { kind: "specialist", agent: p.agent, summary: p.summary, citations: p.citations ?? 0 },
      ]);
    });

    es.addEventListener("final", (e: MessageEvent) => {
      const p = JSON.parse(e.data);
      setFinalId(p.report_id);
      if (p.thread_id) setThreadId(p.thread_id);
      if (p.memo) setStructuredMemo(p.memo as Memo);
      setDone(true);
      es.close();
    });

    es.addEventListener("error", (e: MessageEvent) => {
      if (es.readyState === EventSource.CLOSED) return;
      try {
        if ((e as MessageEvent).data) setError(JSON.parse((e as MessageEvent).data).error);
      } catch {
        setError("Connection error");
      }
    });

    return () => es.close();
  }, [ticker]);

  return (
    <main className="mx-auto max-w-6xl px-4 py-8">
      <header className="mb-6 flex items-center justify-between">
        <Link href="/" className="text-sm text-neutral-500 hover:text-neutral-200">
          ← FinSight
        </Link>
        <div className="text-right">
          <div className="font-mono text-3xl font-semibold">{ticker}</div>
          {finalId && <div className="text-xs text-neutral-600">report {finalId.slice(0, 8)}</div>}
        </div>
      </header>

      {error && (
        <div className="mb-4 rounded-md border border-rose-900 bg-rose-950/40 p-3 text-sm text-rose-300">
          {error}
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
        <aside>
          <AgentTimeline steps={steps} done={done} />
        </aside>
        <section>
          {structuredMemo ? (
            <MemoViewer markdown="" memo={structuredMemo} />
          ) : (
            <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-8 text-center text-sm text-neutral-500">
              The research agent is working — the memo appears when it finishes.
            </div>
          )}
          {threadId && structuredMemo && <MemoChat ticker={ticker} threadId={threadId} />}
        </section>
      </div>
    </main>
  );
}
