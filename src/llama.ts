// Cliente de chat para o llama-server (API compativel com OpenAI),
// com streaming SSE e captura das metricas (tok/s) do llama.cpp.

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface SamplingParams {
  temperature: number;
  top_p: number;
  top_k: number;
  min_p: number;
  repeat_penalty: number;
  max_tokens: number;
}

export interface Timings {
  prompt_n?: number;
  prompt_per_second?: number;
  predicted_n?: number;
  predicted_per_second?: number;
}

export interface StreamChunk {
  delta?: string;
  /** canal de "pensamento" de modelos de reasoning (ex.: Qwen3.5) */
  reasoning?: string;
  timings?: Timings;
  done?: boolean;
}

export async function* streamChat(
  port: number,
  messages: ChatMessage[],
  params: SamplingParams,
  signal: AbortSignal,
): AsyncGenerator<StreamChunk> {
  const body = {
    model: "local",
    messages,
    stream: true,
    cache_prompt: true,
    temperature: params.temperature,
    top_p: params.top_p,
    top_k: params.top_k,
    min_p: params.min_p,
    repeat_penalty: params.repeat_penalty,
    max_tokens: params.max_tokens,
  };

  const resp = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!resp.ok || !resp.body) {
    const txt = await resp.text().catch(() => "");
    throw new Error(`Servidor respondeu ${resp.status}: ${txt}`);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const raw of lines) {
      const line = raw.trim();
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") {
        yield { done: true };
        return;
      }
      try {
        const obj = JSON.parse(data);
        const d = obj?.choices?.[0]?.delta;
        const delta: string | undefined = d?.content;
        // llama.cpp expoe o pensamento em reasoning_content (ou variantes)
        const reasoning: string | undefined =
          d?.reasoning_content ?? d?.reasoning;
        const timings: Timings | undefined = obj?.timings;
        if (delta || reasoning || timings) {
          yield { delta, reasoning, timings };
        }
      } catch {
        // chunk parcial/ruido — ignora
      }
    }
  }
  yield { done: true };
}

/// Verifica /health e retorna o id do modelo carregado, se disponivel.
export async function fetchModelId(port: number): Promise<string | null> {
  try {
    const r = await fetch(`http://127.0.0.1:${port}/v1/models`);
    if (!r.ok) return null;
    const j = await r.json();
    return j?.data?.[0]?.id ?? null;
  } catch {
    return null;
  }
}
