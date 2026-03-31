const BASE = '/api'

export async function createSession(): Promise<string> {
  const res = await fetch(`${BASE}/session`, { method: 'POST' })
  if (!res.ok) throw new Error('Failed to create session')
  const data = await res.json()
  return data.session_id as string
}

export async function uploadPrescription(
  sessionId: string,
  file: File,
): Promise<{ drugs_found: string[]; status: string }> {
  const form = new FormData()
  form.append('session_id', sessionId)
  form.append('file', file)
  const res = await fetch(`${BASE}/upload`, { method: 'POST', body: form })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Upload failed' }))
    throw new Error(err.detail ?? 'Upload failed')
  }
  return res.json()
}

export async function askQuestion(
  sessionId: string,
  question: string,
): Promise<{ answer: string; sources: Array<{ drug_name: string; section: string }> }> {
  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, question }),
  })
  if (!res.ok) throw new Error('Chat request failed')
  return res.json()
}
