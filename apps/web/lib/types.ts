export type AgentName = "market" | "quant" | "news" | "sec" | "writer" | "critic";

export type AgentStatus = "pending" | "running" | "done" | "error";

export type AgentEvent = {
  agent: AgentName;
  status: AgentStatus;
  summary?: string;
};

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
