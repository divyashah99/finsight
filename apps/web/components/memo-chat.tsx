"use client";

import { readSSE } from "@/lib/sse";
import type { ChatMessage } from "@/lib/types";
import { useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/**
 * Follow-up chat over a generated memo. Streams the analyst agent's answer
 * (token by token) and surfaces which tools it invoked. The backend agent
 * remembers the conversation via the LangGraph checkpointer keyed by threadId.
 */
export function MemoChat({ ticker, threadId }: { ticker: string; threadId: string }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  async function send() {
    const q = input.trim();
    if (!q || busy) return;
    setInput("");
    setBusy(true);
    setMessages((prev) => [
      ...prev,
      { role: "user", content: q },
      { role: "assistant", content: "", tools: [] },
    ]);

    const patchLast = (patch: (m: ChatMessage) => ChatMessage) =>
      setMessages((prev) => prev.map((m, i) => (i === prev.length - 1 ? patch(m) : m)));

    try {
      const stream = readSSE(`${API_URL}/research/${ticker}/chat`, {
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ thread_id: threadId, message: q }),
      });
      for await (const evt of stream) {
        if (evt.event === "token") {
          const { delta } = JSON.parse(evt.data);
          patchLast((m) => ({ ...m, content: m.content + delta }));
        } else if (evt.event === "tool") {
          const { name } = JSON.parse(evt.data);
          patchLast((m) => ({ ...m, tools: [...(m.tools ?? []), name] }));
        } else if (evt.event === "error") {
          const { error } = JSON.parse(evt.data);
          patchLast((m) => ({ ...m, content: m.content || `⚠️ ${error}` }));
        } else if (evt.event === "done") {
          break;
        }
      }
    } catch (e) {
      patchLast((m) => ({ ...m, content: m.content || `⚠️ ${(e as Error).message}` }));
    } finally {
      setBusy(false);
      requestAnimationFrame(() =>
        listRef.current?.scrollTo({ top: listRef.current.scrollHeight }),
      );
    }
  }

  return (
    <div className="mt-6 rounded-lg border border-neutral-800 bg-neutral-950/50">
      <div className="border-b border-neutral-800 px-4 py-2 text-sm font-medium text-neutral-300">
        Ask a follow-up
      </div>

      <div ref={listRef} className="max-h-96 space-y-3 overflow-y-auto p-4">
        {messages.length === 0 && (
          <p className="text-sm text-neutral-600">
            e.g. “What did their latest income statement show?” or “Summarize the
            supply-chain risks from the filings.”
          </p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={m.role === "user" ? "text-right" : "text-left"}>
            {m.tools && m.tools.length > 0 && (
              <div className="mb-1 flex flex-wrap gap-1 text-[10px] text-amber-500/80">
                {m.tools.map((t, j) => (
                  <span key={j} className="rounded bg-amber-950/40 px-1.5 py-0.5 font-mono">
                    🔧 {t}
                  </span>
                ))}
              </div>
            )}
            <span
              className={
                m.role === "user"
                  ? "inline-block rounded-lg bg-neutral-800 px-3 py-2 text-sm text-neutral-100"
                  : "inline-block whitespace-pre-wrap rounded-lg bg-neutral-900 px-3 py-2 text-sm text-neutral-200"
              }
            >
              {m.content || (busy && i === messages.length - 1 ? "…" : "")}
            </span>
          </div>
        ))}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          void send();
        }}
        className="flex gap-2 border-t border-neutral-800 p-3"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={busy}
          placeholder={`Ask about ${ticker}…`}
          className="flex-1 rounded-md border border-neutral-800 bg-neutral-900 px-3 py-2 text-sm text-neutral-100 placeholder:text-neutral-600 focus:border-neutral-600 focus:outline-none disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={busy || !input.trim()}
          className="rounded-md bg-neutral-100 px-4 py-2 text-sm font-medium text-neutral-900 hover:bg-white disabled:opacity-40"
        >
          {busy ? "…" : "Send"}
        </button>
      </form>
    </div>
  );
}
