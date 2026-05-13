"use client";

import type { Argument, Memo, Risk } from "@/lib/types";

/**
 * Two-pane memo viewer:
 *  - left:  streaming markdown narrative (visible while the writer streams)
 *  - right: structured memo (bull/bear columns, metrics, risks) — populated
 *           after the structurer + critic finalize the run
 */

function renderInline(text: string) {
  let html = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
  html = html.replace(/`([^`]+)`/g, "<code class='rounded bg-neutral-800 px-1 text-xs'>$1</code>");
  html = html.replace(/\[(\d+)\]/g, '<sup class="text-amber-400">[$1]</sup>');
  return html;
}

function renderMarkdown(md: string): string {
  const lines = md.split("\n");
  const out: string[] = [];
  let inList = false;
  for (const raw of lines) {
    const line = raw.trimEnd();
    if (line.startsWith("# ")) {
      if (inList) { out.push("</ul>"); inList = false; }
      out.push(`<h1 class="mt-4 text-2xl font-semibold">${renderInline(line.slice(2))}</h1>`);
    } else if (line.startsWith("## ")) {
      if (inList) { out.push("</ul>"); inList = false; }
      out.push(`<h2 class="mt-4 text-xl font-semibold text-neutral-200">${renderInline(line.slice(3))}</h2>`);
    } else if (line.startsWith("### ")) {
      if (inList) { out.push("</ul>"); inList = false; }
      out.push(`<h3 class="mt-3 text-base font-semibold text-neutral-300">${renderInline(line.slice(4))}</h3>`);
    } else if (line.startsWith("- ") || line.startsWith("* ")) {
      if (!inList) { out.push("<ul class='list-disc pl-6 space-y-1 text-neutral-300'>"); inList = true; }
      out.push(`<li>${renderInline(line.slice(2))}</li>`);
    } else if (line === "") {
      if (inList) { out.push("</ul>"); inList = false; }
      out.push("<br />");
    } else {
      if (inList) { out.push("</ul>"); inList = false; }
      out.push(`<p class="text-neutral-300 leading-relaxed">${renderInline(line)}</p>`);
    }
  }
  if (inList) out.push("</ul>");
  return out.join("");
}

function ArgList({ args, side }: { args: Argument[]; side: "bull" | "bear" }) {
  const tone = side === "bull" ? "text-emerald-400" : "text-rose-400";
  return (
    <ul className="space-y-3">
      {args.map((a, i) => (
        <li key={i} className="rounded-md border border-neutral-800 bg-neutral-950 p-3">
          <div className={`text-sm font-medium ${tone}`}>{a.claim}</div>
          <div className="mt-1 text-xs text-neutral-400">{a.evidence}</div>
          {a.citation_ids.length > 0 && (
            <div className="mt-2 flex gap-1">
              {a.citation_ids.map((cid) => (
                <span
                  key={cid}
                  className="rounded bg-amber-950/50 px-1.5 py-0.5 text-[10px] text-amber-400"
                >
                  [{cid}]
                </span>
              ))}
            </div>
          )}
        </li>
      ))}
    </ul>
  );
}

function RiskItem({ r }: { r: Risk }) {
  const sevColor =
    r.severity === "high"
      ? "border-rose-800 bg-rose-950/30 text-rose-300"
      : r.severity === "medium"
        ? "border-amber-800 bg-amber-950/30 text-amber-300"
        : "border-neutral-800 bg-neutral-950 text-neutral-300";
  return (
    <li className={`rounded-md border p-2 text-xs ${sevColor}`}>
      <div className="font-medium">{r.title}</div>
      <div className="mt-1 opacity-80">{r.detail}</div>
    </li>
  );
}

function StructuredMemo({ memo }: { memo: Memo }) {
  const rec = memo.recommendation ?? "no_opinion";
  const recColor =
    rec === "buy"
      ? "bg-emerald-500 text-black"
      : rec === "sell"
        ? "bg-rose-500 text-white"
        : "bg-neutral-700 text-neutral-100";

  return (
    <div className="space-y-5">
      <header className="flex items-start justify-between gap-3">
        <div>
          <div className="text-xs uppercase tracking-widest text-neutral-500">Headline</div>
          <h2 className="mt-1 text-lg font-medium text-neutral-100">{memo.headline}</h2>
        </div>
        <div className="flex flex-col items-end gap-1">
          <span className={`rounded-md px-2 py-1 text-xs font-bold uppercase ${recColor}`}>
            {rec}
          </span>
          {typeof memo.conviction === "number" && (
            <span className="text-xs text-neutral-500">
              conviction {"●".repeat(memo.conviction)}
              <span className="text-neutral-800">{"●".repeat(5 - memo.conviction)}</span>
            </span>
          )}
        </div>
      </header>

      {memo.key_metrics && memo.key_metrics.length > 0 && (
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {memo.key_metrics.map((m) => (
            <div key={m.name} className="rounded-md border border-neutral-800 bg-neutral-950 p-2">
              <div className="text-[10px] uppercase text-neutral-500">{m.name}</div>
              <div className="font-mono text-sm text-neutral-100">{m.value}</div>
            </div>
          ))}
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-2">
        <section>
          <h3 className="mb-2 text-xs font-medium uppercase tracking-widest text-emerald-500">
            Bull case
          </h3>
          {memo.thesis_bull?.length ? <ArgList args={memo.thesis_bull} side="bull" /> : null}
        </section>
        <section>
          <h3 className="mb-2 text-xs font-medium uppercase tracking-widest text-rose-500">
            Bear case
          </h3>
          {memo.thesis_bear?.length ? <ArgList args={memo.thesis_bear} side="bear" /> : null}
        </section>
      </div>

      {memo.catalysts?.length ? (
        <section>
          <h3 className="mb-2 text-xs font-medium uppercase tracking-widest text-neutral-400">
            Catalysts
          </h3>
          <ul className="list-disc space-y-1 pl-5 text-sm text-neutral-300">
            {memo.catalysts.map((c, i) => <li key={i}>{c}</li>)}
          </ul>
        </section>
      ) : null}

      {memo.risks?.length ? (
        <section>
          <h3 className="mb-2 text-xs font-medium uppercase tracking-widest text-neutral-400">
            Risks
          </h3>
          <ul className="space-y-2">
            {memo.risks.map((r, i) => <RiskItem key={i} r={r} />)}
          </ul>
        </section>
      ) : null}
    </div>
  );
}

export function MemoViewer({
  markdown,
  memo,
}: {
  markdown: string;
  memo: Memo | null;
}) {
  if (memo && (memo.thesis_bull?.length || memo.thesis_bear?.length)) {
    return (
      <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-6">
        <StructuredMemo memo={memo} />
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-6">
      {markdown ? (
        <div
          className="space-y-2"
          dangerouslySetInnerHTML={{ __html: renderMarkdown(markdown) }}
        />
      ) : (
        <div className="flex h-32 items-center justify-center text-sm text-neutral-600">
          Waiting for agents…
        </div>
      )}
    </div>
  );
}
