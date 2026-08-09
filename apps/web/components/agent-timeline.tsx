"use client";

import type { TimelineStep } from "@/lib/types";

const SPEC_LABEL: Record<string, string> = {
  fundamentals: "Fundamentals",
  technicals: "Technicals",
  news: "News",
  filings: "Filings (RAG)",
};

/**
 * Dynamic agent timeline. Unlike the old fixed pipeline, the sequence here is
 * decided at runtime by the supervisor — each `decision` shows where it routed
 * (and why), each `specialist` shows what that sub-agent found.
 */
export function AgentTimeline({ steps, done }: { steps: TimelineStep[]; done: boolean }) {
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-4">
      <h3 className="mb-3 text-xs uppercase tracking-widest text-neutral-500">
        Agent timeline
      </h3>

      {steps.length === 0 && (
        <p className="text-sm text-neutral-600">Supervisor is planning…</p>
      )}

      <ol className="space-y-2">
        {steps.map((s, i) =>
          s.kind === "decision" ? (
            <li key={i} className="flex items-start gap-2 text-xs text-neutral-500">
              <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-sky-500" />
              <span>
                <span className="text-sky-400">supervisor →</span>{" "}
                {s.next === "synthesize" ? "synthesize memo" : SPEC_LABEL[s.next] ?? s.next}
                {s.reason ? <span className="text-neutral-600"> · {s.reason}</span> : null}
              </span>
            </li>
          ) : (
            <li key={i} className="flex items-start gap-2 text-sm">
              <span className="mt-1.5 h-2 w-2 shrink-0 rounded-full bg-emerald-500" />
              <span className="flex-1">
                <span className="text-neutral-300">
                  {SPEC_LABEL[s.agent] ?? s.agent}
                  {s.citations > 0 && (
                    <span className="ml-2 rounded bg-amber-950/40 px-1.5 py-0.5 text-[10px] text-amber-400">
                      {s.citations} citations
                    </span>
                  )}
                </span>
                <span className="mt-0.5 block text-xs text-neutral-500">{s.summary}</span>
              </span>
            </li>
          ),
        )}
      </ol>

      {!done && steps.length > 0 && (
        <p className="mt-3 flex items-center gap-2 text-xs text-amber-400">
          <span className="h-2 w-2 animate-pulse rounded-full bg-amber-400" />
          working…
        </p>
      )}
    </div>
  );
}
