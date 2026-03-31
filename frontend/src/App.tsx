import { useEffect, useRef, useState } from 'react'
import { askQuestion, createSession, uploadPrescription } from './api'
import ChatPanel from './components/ChatPanel'
import Mascot from './components/Mascot'
import UploadZone from './components/UploadZone'

export type Phase = 'idle' | 'uploading' | 'ready' | 'asking'

export interface Message {
  id: string
  role: 'user' | 'lara'
  content: string
  sources?: Array<{ drug_name: string; section: string }>
}

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [phase, setPhase] = useState<Phase>('idle')
  const [drugs, setDrugs] = useState<string[]>([])
  const [messages, setMessages] = useState<Message[]>([])
  const [error, setError] = useState<string | null>(null)
  const [mascotHappy, setMascotHappy] = useState(false)
  const happyTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    createSession()
      .then(setSessionId)
      .catch(() => setError('Could not connect to the backend. Is it running?'))
  }, [])

  function triggerHappy() {
    setMascotHappy(true)
    if (happyTimer.current) clearTimeout(happyTimer.current)
    happyTimer.current = setTimeout(() => setMascotHappy(false), 800)
  }

  async function handleUpload(file: File) {
    if (!sessionId) return
    setPhase('uploading')
    setError(null)
    try {
      const result = await uploadPrescription(sessionId, file)
      setDrugs(result.drugs_found)
      setPhase('ready')
      triggerHappy()
      const drugsText =
        result.drugs_found.length > 0
          ? `I've read the official FDA leaflets for **${result.drugs_found.join(', ')}**. What would you like to know? Ask me about dosage, side effects, warnings, or drug interactions.`
          : "I processed your prescription, but couldn't find matching leaflets in DailyMed. This sometimes happens with brand names. Try asking anyway — I'll do my best!"
      setMessages([{ id: 'welcome', role: 'lara', content: drugsText }])
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Something went wrong during upload.')
      setPhase('idle')
    }
  }

  async function handleQuestion(question: string) {
    if (!sessionId || phase === 'asking') return
    const userMsg: Message = { id: `u-${Date.now()}`, role: 'user', content: question }
    setMessages((prev) => [...prev, userMsg])
    setPhase('asking')
    try {
      const result = await askQuestion(sessionId, question)
      setMessages((prev) => [
        ...prev,
        {
          id: `l-${Date.now()}`,
          role: 'lara',
          content: result.answer,
          sources: result.sources,
        },
      ])
      triggerHappy()
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          role: 'lara',
          content: "Sorry, something went wrong on my end. Please try again.",
        },
      ])
    } finally {
      setPhase('ready')
    }
  }

  function handleReset() {
    setPhase('idle')
    setDrugs([])
    setMessages([])
    setError(null)
    createSession().then(setSessionId).catch(() => null)
  }

  const mascotState = mascotHappy
    ? 'happy'
    : phase === 'uploading'
      ? 'uploading'
      : phase === 'asking'
        ? 'thinking'
        : 'idle'

  const inChat = phase === 'ready' || phase === 'asking'

  return (
    <div className="min-h-screen bg-surface font-sans flex flex-col">
      {/* ── Header ── */}
      <header className="sticky top-0 z-20 bg-surface/80 backdrop-blur-sm px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-lg font-semibold text-navy tracking-tight">LARA</span>
          <span className="text-xs font-medium text-secondary bg-secondary-container px-2 py-0.5 rounded-full">
            medical assistant
          </span>
        </div>

        {inChat && (
          <div className="flex items-center gap-3">
            <div className="flex gap-1.5 flex-wrap">
              {drugs.map((d) => (
                <span
                  key={d}
                  className="text-xs font-medium bg-primary-container text-primary-dark px-2.5 py-1 rounded-full capitalize"
                >
                  {d}
                </span>
              ))}
            </div>
            <button
              onClick={handleReset}
              className="text-xs font-medium text-secondary hover:text-navy transition-colors px-3 py-1.5 rounded-full hover:bg-surface-low"
            >
              New prescription
            </button>
          </div>
        )}
      </header>

      {/* ── Error banner ── */}
      {error && (
        <div className="mx-6 mt-2 px-4 py-3 bg-red-50 border-l-4 border-red-400 rounded-2xl text-sm text-red-700">
          {error}
        </div>
      )}

      {/* ── Main content ── */}
      <main className="flex-1 flex overflow-hidden">
        {!inChat ? (
          /* ── Upload screen (bento layout) ── */
          <div className="flex-1 grid md:grid-cols-[1fr_1fr] gap-5 p-6 max-w-4xl mx-auto w-full content-center">
            {/* Mascot card */}
            <div className="bg-surface-low rounded-5xl p-8 flex flex-col items-center justify-center gap-5 min-h-[320px]">
              <Mascot size={200} state={mascotState} />
              <div className="text-center">
                <p className="text-xl font-semibold text-navy leading-snug">
                  Hi! I'm LARA.
                </p>
                <p className="text-sm text-secondary mt-1.5 leading-relaxed max-w-[220px]">
                  Upload a prescription and I'll look up the official drug leaflets to answer your
                  questions.
                </p>
              </div>
              {phase === 'uploading' && (
                <div className="flex gap-2 items-center">
                  <div className="typing-dot" />
                  <div className="typing-dot" />
                  <div className="typing-dot" />
                  <span className="text-xs text-secondary ml-1">Reading leaflets…</span>
                </div>
              )}
            </div>

            {/* Upload card */}
            <div className="bg-surface-lowest rounded-5xl shadow-ambient flex flex-col justify-center min-h-[320px]">
              <UploadZone onUpload={handleUpload} loading={phase === 'uploading'} />
            </div>
          </div>
        ) : (
          /* ── Chat screen ── */
          <div className="flex-1 flex overflow-hidden max-w-5xl mx-auto w-full p-4 gap-4">
            {/* Mascot sidebar */}
            <div className="hidden md:flex flex-col items-center gap-4 w-44 flex-shrink-0 pt-4">
              <div className="bg-surface-low rounded-4xl p-5 flex flex-col items-center gap-3 w-full">
                <Mascot size={130} state={mascotState} />
                {phase === 'asking' && (
                  <div className="flex gap-1.5 justify-center">
                    <div className="typing-dot" />
                    <div className="typing-dot" />
                    <div className="typing-dot" />
                  </div>
                )}
                {phase !== 'asking' && (
                  <p className="text-xs text-center text-secondary leading-relaxed">
                    Ask me anything about your prescription
                  </p>
                )}
              </div>
            </div>

            {/* Chat messages + input */}
            <div className="flex-1 flex flex-col overflow-hidden">
              <ChatPanel
                messages={messages}
                onSend={handleQuestion}
                disabled={phase === 'asking'}
              />
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
