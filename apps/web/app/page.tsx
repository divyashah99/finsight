"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

export default function Home() {
  const router = useRouter();
  const [ticker, setTicker] = useState("");

  return (
    <main className="flex min-h-screen items-center justify-center p-8">
      <div className="w-full max-w-xl space-y-8 text-center">
        <div>
          <h1 className="text-5xl font-semibold tracking-tight">FinSight</h1>
          <p className="mt-3 text-neutral-400">
            Autonomous equity research — multi-agent AI investment memo
          </p>
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            const t = ticker.trim().toUpperCase();
            if (t) router.push(`/research/${t}`);
          }}
          className="flex gap-2"
        >
          <input
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            placeholder="Enter ticker (e.g. AAPL)"
            className="flex-1 rounded-md border border-neutral-800 bg-neutral-950 px-4 py-3 text-lg uppercase placeholder:normal-case placeholder:text-neutral-500 focus:border-neutral-600 focus:outline-none"
            autoFocus
          />
          <button
            type="submit"
            className="rounded-md bg-white px-6 py-3 font-medium text-black hover:bg-neutral-200"
          >
            Research
          </button>
        </form>

        <p className="text-xs text-neutral-600">
          Powered by LangGraph + GPT-4o-mini · Alpha Vantage · SEC EDGAR
        </p>
      </div>
    </main>
  );
}
