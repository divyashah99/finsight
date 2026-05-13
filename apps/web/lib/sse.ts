/**
 * Minimal SSE client using fetch + ReadableStream.
 *
 * Why not EventSource? We POST to `/research/{ticker}` and EventSource only
 * supports GET. This implementation parses `event:`/`data:` frames manually
 * out of the chunked response body.
 */

export type SseEvent = {
  event: string;
  data: string;
};

export async function* readSSE(
  url: string,
  init?: RequestInit,
): AsyncGenerator<SseEvent> {
  const res = await fetch(url, {
    method: "POST",
    headers: { Accept: "text/event-stream", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok || !res.body) {
    throw new Error(`SSE failed: ${res.status} ${res.statusText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);

      let event = "message";
      const dataLines: string[] = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
      }
      if (dataLines.length === 0) continue;
      yield { event, data: dataLines.join("\n") };
    }
  }
}
