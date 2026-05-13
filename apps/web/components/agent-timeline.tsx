"use client";

import type { AgentEvent, AgentName } from "@/lib/types";

const ORDER: AgentName[] = ["market", "quant", "news", "sec", "writer", "critic"];

const LABEL: Record<AgentName, string> = {
  market: "Market Agent",
  quant: "Quant Agent",
  news: "News Agent",
  sec: "SEC Agent (RAG)",
  writer: "Writer",
  critic: "Critic",
};

function dot(status: string) {
  if (status === "done") return "bg-emerald-500";
  if (status === "running") return "bg-amber-400 animate-pulse";
  if (status === "error") return "bg-rose-500";
  return "bg-neutral-700";
}

export function AgentTimeline({ events }: { events: AgentEvent[] }) {
  const byAgent = new Map<AgentName, AgentEvent>();
  for (const e of events) byAgent.set(e.agent, e);

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-4">
      <h3 className="mb-3 text-xs uppercase tracking-widest text-neutral-500">
        Agent timeline
      </h3>
      <ol className="space-y-2">
        {ORDER.map((agent) => {
          const e = byAgent.get(agent);
          const status = e?.status ?? "pending";
          return (
            <li key={agent} className="flex items-center gap-3 text-sm">
              <span className={`h-2 w-2 rounded-full ${dot(status)}`} />
              <span className="w-32 text-neutral-300">{LABEL[agent]}</span>
              <span className="flex-1 truncate text-neutral-500">
                {e?.summary ?? (status === "running" ? "working…" : "")}
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
