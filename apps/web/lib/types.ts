// Agentic timeline: the supervisor decides the sequence at runtime, so the
// timeline is a dynamic list of decisions + specialist completions.
export type SpecialistName = "fundamentals" | "technicals" | "news" | "filings";

export type TimelineStep =
  | { kind: "decision"; next: string; reason: string }
  | { kind: "specialist"; agent: string; summary: string; citations: number };

export type Recommendation = "buy" | "hold" | "sell" | "no_opinion";

export type Argument = {
  claim: string;
  evidence: string;
  citation_ids: number[];
};

export type Risk = {
  title: string;
  detail: string;
  severity: "low" | "medium" | "high";
  citation_ids: number[];
};

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  tools?: string[];
};

export type Memo = {
  ticker: string;
  as_of?: string;
  recommendation?: Recommendation;
  conviction?: number;
  headline?: string;
  thesis_bull?: Argument[];
  thesis_bear?: Argument[];
  key_metrics?: { name: string; value: string }[];
  catalysts?: string[];
  risks?: Risk[];
  markdown?: string;
};
