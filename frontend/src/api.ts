const BASE = '/api'
const API_KEY = import.meta.env.VITE_API_KEY as string

// Per-session secret issued once by createSession(). It proves ownership of the
// current session (the API key is shared by every browser, so it cannot). Lives
// in module state to match the lifetime of the React-state session id (neither
// is persisted to localStorage). Cleared on 410 so a stale token isn't reused.
let sessionToken: string | null = null

function sessionHeaders(extra?: Record<string, string>): Record<string, string> {
  return { 'X-API-Key': API_KEY, 'X-Session-Token': sessionToken ?? '', ...extra }
}

export class SessionExpiredError extends Error {
  constructor() {
    super('Your session has expired. Starting a new one…')
    this.name = 'SessionExpiredError'
  }
}

export interface Source {
  drug_name: string
  section: string
}

export type StreamEvent =
  | { type: 'token'; text: string }
  | { type: 'reset' } // discard tokens received so far (mid-stream provider failover)
  | { type: 'sources'; sources: Source[] }
  | { type: 'done' }

export async function createSession(): Promise<string> {
  const res = await fetch(`${BASE}/session`, { method: 'POST', headers: { 'X-API-Key': API_KEY } })
  if (!res.ok) throw new Error('Failed to create session')
  const data = await res.json()
  sessionToken = data.session_token as string
  return data.session_id as string
}

interface JobStatus {
  job_id: string
  session_id: string
  status: 'processing' | 'done' | 'failed'
  drugs_found: string[]
  missing_leaflets: string[]
  error?: string | null
}

async function pollJobStatus(jobId: string, sessionId: string, maxWaitMs = 120_000, signal?: AbortSignal): Promise<JobStatus> {
  const deadline = Date.now() + maxWaitMs
  while (Date.now() < deadline) {
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError')
    await new Promise((resolve) => setTimeout(resolve, 1000))
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError')
    const res = await fetch(
      `${BASE}/upload/status/${jobId}?session_id=${encodeURIComponent(sessionId)}`,
      { signal, headers: sessionHeaders() },
    )
    if (res.status === 410) { sessionToken = null; throw new SessionExpiredError() }
    if (!res.ok) throw new Error('Could not check upload status.')
    const data: JobStatus = await res.json()
    if (data.status === 'done') return data
    if (data.status === 'failed') throw new Error(data.error ?? 'Prescription processing failed.')
  }
  throw new Error('Upload timed out. The server may be busy — please try again.')
}

export async function uploadPrescription(
  sessionId: string,
  file: File,
  signal?: AbortSignal,
): Promise<{ drugs_found: string[]; missing_leaflets: string[]; status: string }> {
  const form = new FormData()
  form.append('session_id', sessionId)
  form.append('file', file)
  const res = await fetch(`${BASE}/upload`, { method: 'POST', body: form, signal, headers: sessionHeaders() })
  if (res.status === 410) { sessionToken = null; throw new SessionExpiredError() }
  if (res.status === 429) {
    throw new Error("You're sending requests too quickly. Please wait a moment and try again.")
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Upload failed' }))
    throw new Error(err.detail ?? 'Upload failed')
  }
  const { job_id } = await res.json()
  return pollJobStatus(job_id, sessionId, 120_000, signal)
}


export interface SampleInfo {
  id: string
  label: string
  description: string
  drugs: string[]
}

export async function listSamples(): Promise<SampleInfo[]> {
  const res = await fetch(`${BASE}/samples`, { headers: { 'X-API-Key': API_KEY } })
  if (!res.ok) throw new Error('Failed to load samples')
  const data = await res.json()
  return data.samples as SampleInfo[]
}

export async function loadSample(
  sessionId: string,
  sampleId: string,
  signal?: AbortSignal,
): Promise<{ drugs_found: string[]; missing_leaflets: string[]; status: string }> {
  const res = await fetch(`${BASE}/samples/${encodeURIComponent(sampleId)}`, {
    method: 'POST',
    headers: sessionHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ session_id: sessionId }),
    signal,
  })
  if (res.status === 410) { sessionToken = null; throw new SessionExpiredError() }
  if (res.status === 429) {
    throw new Error("You're sending requests too quickly. Please wait a moment and try again.")
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Sample failed to load' }))
    throw new Error(err.detail ?? 'Sample failed to load')
  }
  const { job_id } = await res.json()
  return pollJobStatus(job_id, sessionId, 120_000, signal)
}

export interface InteractionFlag {
  drug_a: string
  drug_b: string
  excerpt: string
}

export interface InteractionsResult {
  pairs_checked: number
  interactions: InteractionFlag[]
}

export async function checkInteractions(sessionId: string): Promise<InteractionsResult> {
  const res = await fetch(`${BASE}/interactions`, {
    method: 'POST',
    headers: sessionHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ session_id: sessionId }),
  })
  if (res.status === 410) { sessionToken = null; throw new SessionExpiredError() }
  if (!res.ok) throw new Error('Interaction check failed')
  return res.json() as Promise<InteractionsResult>
}

export async function* streamQuestion(
  sessionId: string,
  question: string,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const res = await fetch(`${BASE}/chat/stream`, {
    method: 'POST',
    headers: sessionHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ session_id: sessionId, question }),
    signal,
  })
  if (res.status === 410) { sessionToken = null; throw new SessionExpiredError() }
  if (res.status === 429) {
    throw new Error("You're sending requests too quickly. Please wait a moment and try again.")
  }
  if (!res.ok) throw new Error('Stream request failed')

  if (!res.body) throw new Error('No response body from server.')
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    if (signal?.aborted) break
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    // SSE events are separated by double newlines
    const events = buffer.split('\n\n')
    buffer = events.pop() ?? ''

    for (const event of events) {
      // Parse the event type and data from the SSE block.
      // Each block has one or more lines; we look for "event:" and "data:" lines.
      let eventType = 'token'
      let dataLine = ''
      for (const line of event.split('\n')) {
        if (line.startsWith('event: ')) {
          eventType = line.slice(7).trim()
        } else if (line.startsWith('data: ')) {
          dataLine = line.slice(6)
        }
      }

      if (!dataLine && eventType !== 'done' && eventType !== 'reset') continue

      if (eventType === 'done') {
        yield { type: 'done' }
        return
      } else if (eventType === 'reset') {
        yield { type: 'reset' }
      } else if (eventType === 'sources') {
        const parsed = JSON.parse(dataLine)
        yield { type: 'sources', sources: parsed.sources }
      } else {
        // token data is JSON-encoded to safely transport newlines
        const text = JSON.parse(dataLine) as string
        yield { type: 'token', text }
      }
    }
  }
}
