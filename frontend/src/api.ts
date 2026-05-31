const BASE = '/api'

export interface Source {
  drug_name: string
  section: string
}

export interface ChatTurn {
  role: 'user' | 'assistant'
  content: string
}

export type StreamEvent =
  | { type: 'token'; text: string }
  | { type: 'sources'; sources: Source[] }
  | { type: 'done' }

export async function createSession(): Promise<string> {
  const res = await fetch(`${BASE}/session`, { method: 'POST' })
  if (!res.ok) throw new Error('Failed to create session')
  const data = await res.json()
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

async function pollJobStatus(jobId: string, maxWaitMs = 120_000): Promise<JobStatus> {
  const deadline = Date.now() + maxWaitMs
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 1000))
    const res = await fetch(`${BASE}/upload/status/${jobId}`)
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
): Promise<{ drugs_found: string[]; missing_leaflets: string[]; status: string }> {
  const form = new FormData()
  form.append('session_id', sessionId)
  form.append('file', file)
  const res = await fetch(`${BASE}/upload`, { method: 'POST', body: form })
  if (res.status === 429) {
    throw new Error("You're sending requests too quickly. Please wait a moment and try again.")
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Upload failed' }))
    throw new Error(err.detail ?? 'Upload failed')
  }
  const { job_id } = await res.json()
  return pollJobStatus(job_id)
}

export async function askQuestion(
  sessionId: string,
  question: string,
  history: ChatTurn[] = [],
): Promise<{ answer: string; sources: Source[] }> {
  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, question, history }),
  })
  if (!res.ok) throw new Error('Chat request failed')
  return res.json()
}

export async function* streamQuestion(
  sessionId: string,
  question: string,
  history: ChatTurn[] = [],
): AsyncGenerator<StreamEvent> {
  const res = await fetch(`${BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, question, history }),
  })
  if (res.status === 429) {
    throw new Error("You're sending requests too quickly. Please wait a moment and try again.")
  }
  if (!res.ok) throw new Error('Stream request failed')

  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
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

      if (!dataLine && eventType !== 'done') continue

      if (eventType === 'done') {
        yield { type: 'done' }
        return
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
