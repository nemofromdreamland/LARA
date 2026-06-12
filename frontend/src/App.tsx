import { useCallback, useEffect, useRef, useState } from 'react'
import { SessionExpiredError, createSession, streamQuestion, uploadPrescription } from './api'
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

function SunIcon() {
  return (
    <svg viewBox="0 0 24 24" width={16} height={16} fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="5" />
      <line x1="12" y1="1" x2="12" y2="3" />
      <line x1="12" y1="21" x2="12" y2="23" />
      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
      <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
      <line x1="1" y1="12" x2="3" y2="12" />
      <line x1="21" y1="12" x2="23" y2="12" />
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
      <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
    </svg>
  )
}

function MoonIcon() {
  return (
    <svg viewBox="0 0 24 24" width={16} height={16} fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  )
}

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [sessionReady, setSessionReady] = useState(false)
  const [phase, setPhase] = useState<Phase>('idle')
  const [drugs, setDrugs] = useState<string[]>([])
  const [messages, setMessages] = useState<Message[]>([])
  const [error, setError] = useState<string | null>(null)
  const [mascotHappy, setMascotHappy] = useState(false)
  const [mascotError, setMascotError] = useState(false)
  const [dark, setDark] = useState(() => {
    if (typeof window === 'undefined') return false
    const stored = localStorage.getItem('theme')
    return stored === 'dark' || (!stored && window.matchMedia('(prefers-color-scheme: dark)').matches)
  })
  const happyTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const errorTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const abortControllerRef = useRef<AbortController | null>(null)
  const chatInputRef = useRef<HTMLTextAreaElement>(null)

  // Keep <html class="dark"> in sync with state
  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
  }, [dark])

  // Follow OS preference changes when the user has no stored override
  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    function handleChange(e: MediaQueryListEvent) {
      if (!localStorage.getItem('theme')) setDark(e.matches)
    }
    mq.addEventListener('change', handleChange)
    return () => mq.removeEventListener('change', handleChange)
  }, [])

  useEffect(() => {
    createSession()
      .then((id) => { setSessionId(id); setSessionReady(true) })
      .catch(() => setError('Unable to reach the server. Try refreshing the page.'))
  }, [])

  useEffect(() => {
    if (phase === 'ready') {
      chatInputRef.current?.focus()
    }
  }, [phase])

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort()
      if (happyTimer.current) clearTimeout(happyTimer.current)
      if (errorTimer.current) clearTimeout(errorTimer.current)
    }
  }, [])

  const triggerHappy = useCallback(() => {
    setMascotHappy(true)
    if (happyTimer.current) clearTimeout(happyTimer.current)
    happyTimer.current = setTimeout(() => setMascotHappy(false), 800)
  }, [])

  const triggerMascotError = useCallback(() => {
    setMascotError(true)
    if (errorTimer.current) clearTimeout(errorTimer.current)
    errorTimer.current = setTimeout(() => setMascotError(false), 2000)
  }, [])

  const handleReset = useCallback(() => {
    if (messages.length > 1 && !window.confirm('Start over? Your conversation will be cleared.')) return
    abortControllerRef.current?.abort()
    abortControllerRef.current = null
    setPhase('idle')
    setDrugs([])
    setMessages([])
    setError(null)
    setSessionReady(false)
    createSession()
      .then((id) => { setSessionId(id); setSessionReady(true) })
      .catch(() => setError('Unable to reach the server. Try refreshing the page.'))
  }, [messages])

  const handleUpload = useCallback(async (file: File) => {
    if (!sessionId) return
    abortControllerRef.current?.abort()
    const controller = new AbortController()
    abortControllerRef.current = controller
    setPhase('uploading')
    setError(null)
    try {
      const result = await uploadPrescription(sessionId, file, controller.signal)
      setDrugs(result.drugs_found)
      setPhase('ready')
      triggerHappy()
      let drugsText: string
      if (result.drugs_found.length > 0) {
        const warning =
          result.missing_leaflets.length > 0
            ? `⚠ I couldn't find official leaflets for: **${result.missing_leaflets.join(', ')}**. Results for those drugs may be limited. Try using the generic name.\n\n`
            : ''
        drugsText = `${warning}I've read the official FDA leaflets for **${result.drugs_found.join(', ')}**. What would you like to know? Ask me about dosage, side effects, warnings, or drug interactions.`
      } else {
        drugsText =
          "I processed your prescription, but couldn't find matching leaflets in DailyMed. This sometimes happens with brand names. Try asking anyway — I'll do my best!"
      }
      setMessages([{ id: 'welcome', role: 'lara', content: drugsText }])
    } catch (e: unknown) {
      if (e instanceof DOMException && e.name === 'AbortError') return
      if (e instanceof SessionExpiredError) { handleReset(); return }
      setError(e instanceof Error ? e.message : 'Something went wrong during upload.')
      setPhase('idle')
      triggerMascotError()
    }
  }, [sessionId, triggerHappy, triggerMascotError, handleReset])

  const handleQuestion = useCallback(async (question: string) => {
    if (!sessionId || phase === 'asking') return

    abortControllerRef.current?.abort()
    const controller = new AbortController()
    abortControllerRef.current = controller

    const userMsg: Message = { id: `u-${Date.now()}`, role: 'user', content: question }
    const laraId = `l-${Date.now()}`

    setMessages((prev) => [...prev, userMsg])
    setPhase('asking')

    try {
      for await (const event of streamQuestion(sessionId, question, controller.signal)) {
        if (event.type === 'token') {
          setMessages((prev) => {
            if (!prev.some((m) => m.id === laraId)) {
              return [...prev, { id: laraId, role: 'lara' as const, content: event.text }]
            }
            return prev.map((m) =>
              m.id === laraId ? { ...m, content: m.content + event.text } : m,
            )
          })
        } else if (event.type === 'reset') {
          // Mid-stream provider failover: the answer is being regenerated from
          // scratch, so the partial text shown so far must be discarded.
          setMessages((prev) =>
            prev.map((m) => (m.id === laraId ? { ...m, content: '' } : m)),
          )
        } else if (event.type === 'sources') {
          setMessages((prev) =>
            prev.map((m) => {
              if (m.id !== laraId) return m
              // Strip any trailing CITED: line the LLM appended (citation metadata)
              const cleanContent = m.content.replace(/\nCITED:\s*.+$/i, '').trimEnd()
              return { ...m, content: cleanContent, sources: event.sources }
            }),
          )
        } else if (event.type === 'done') {
          triggerHappy()
        }
      }
      if (!controller.signal.aborted) {
        setPhase('ready')
      }
    } catch (e: unknown) {
      if (e instanceof DOMException && e.name === 'AbortError') return
      if (e instanceof SessionExpiredError) { handleReset(); return }
      setMessages((prev) => {
        const exists = prev.some((m) => m.id === laraId)
        if (exists) {
          return prev.map((m) =>
            m.id === laraId
              ? { ...m, content: 'Sorry, something went wrong on my end. Please try again.' }
              : m,
          )
        }
        return [...prev, { id: laraId, role: 'lara' as const, content: 'Sorry, something went wrong on my end. Please try again.' }]
      })
      triggerMascotError()
      setPhase('ready')
    }
  }, [sessionId, phase, messages, triggerHappy, triggerMascotError, handleReset])

  const mascotState = mascotError
    ? 'error'
    : mascotHappy
      ? 'happy'
      : phase === 'uploading'
        ? 'uploading'
        : phase === 'asking'
          ? 'thinking'
          : 'idle'

  const inChat = phase === 'ready' || phase === 'asking'

  return (
    <div className="min-h-screen bg-surface dark:bg-surface-d font-sans flex flex-col">
      {/* ── Screen-reader phase announcer ── */}
      <div role="status" aria-live="polite" className="sr-only">
        {phase === 'uploading' ? 'Fetching drug leaflets…'
          : phase === 'ready' ? 'Ready. Ask LARA about your prescription.'
          : phase === 'asking' ? 'LARA is thinking…'
          : ''}
      </div>

      {/* ── Header ── */}
      <header className="sticky top-0 z-20 bg-surface/80 dark:bg-surface-d/80 backdrop-blur-sm px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          {inChat && <Mascot size={28} state={mascotState} className="md:hidden flex-shrink-0" />}
          <span className="text-lg font-semibold text-navy dark:text-navy-d tracking-tight">LARA</span>
          <span className="text-xs font-medium text-secondary dark:text-secondary-d bg-secondary-container dark:bg-secondary-container-d px-2 py-0.5 rounded-full">
            medical information assistant
          </span>
        </div>

        <div className="flex items-center gap-3">
          {inChat && (
            <>
              <div className="flex gap-1.5 flex-wrap min-w-0 overflow-hidden">
                {drugs.map((d) => (
                  <span
                    key={d}
                    className="text-xs font-medium bg-primary-container dark:bg-primary-container-d text-primary-dark dark:text-primary-text-d px-2.5 py-1 rounded-full capitalize"
                  >
                    {d}
                  </span>
                ))}
              </div>
              <button
                onClick={handleReset}
                className="text-xs font-medium text-secondary dark:text-secondary-d hover:text-navy dark:hover:text-navy-d transition-colors px-3 py-2.5 min-h-[44px] flex items-center rounded-full hover:bg-surface-low dark:hover:bg-surface-low-d"
              >
                New prescription
              </button>
            </>
          )}

          <button
            aria-label="Toggle dark mode"
            onClick={() => setDark((d) => {
              const next = !d
              localStorage.setItem('theme', next ? 'dark' : 'light')
              return next
            })}
            className="w-11 h-11 flex items-center justify-center rounded-full text-secondary dark:text-secondary-d hover:bg-surface-low dark:hover:bg-surface-low-d transition-colors"
          >
            {dark ? <SunIcon /> : <MoonIcon />}
          </button>
        </div>
      </header>

      {/* ── Error banner ── */}
      {error && (
        <div role="alert" className="mx-6 mt-2 px-4 py-3 bg-red-50 dark:bg-red-950 border-l-4 border-red-400 dark:border-red-700 rounded-2xl text-sm text-red-700 dark:text-red-300 flex items-start justify-between gap-3">
          <span>{error}</span>
          <button
            aria-label="Dismiss error"
            onClick={() => setError(null)}
            className="flex-shrink-0 text-red-400 hover:text-red-600 dark:hover:text-red-200 transition-colors leading-none text-base font-medium"
          >
            ×
          </button>
        </div>
      )}

      {/* ── Main content ── */}
      <main className="flex-1 flex overflow-hidden">
        {!inChat ? (
          /* ── Upload screen (bento layout) ── */
          <div key="upload" className="flex-1 grid md:grid-cols-[1fr_1fr] gap-5 p-6 max-w-4xl mx-auto w-full content-center fade-up">
            {/* Mascot card */}
            <div className="bg-surface-low dark:bg-surface-low-d rounded-5xl p-8 flex flex-col items-center justify-center gap-5 min-h-[320px]">
              <Mascot size={200} state={mascotState} />
              <div className="text-center">
                <p className="text-xl font-semibold text-navy dark:text-navy-d leading-snug">
                  Hi! I'm LARA.
                </p>
                <p className="text-sm text-secondary dark:text-secondary-d mt-1.5 leading-relaxed max-w-[220px]">
                  Upload a prescription and I'll look up the official drug leaflets to answer your
                  questions.
                </p>
              </div>
            </div>

            {/* Upload card */}
            <div className="bg-surface-lowest dark:bg-surface-lowest-d rounded-5xl shadow-ambient flex flex-col justify-center min-h-[320px]">
              <UploadZone onUpload={handleUpload} loading={phase === 'uploading'} sessionReady={sessionReady} />
            </div>
          </div>
        ) : (
          /* ── Chat screen ── */
          <div key="chat" className="flex-1 flex overflow-hidden max-w-5xl mx-auto w-full p-4 gap-4 fade-up">
            {/* Mascot sidebar */}
            <div className="hidden md:flex flex-col items-center gap-4 w-44 flex-shrink-0 pt-4">
              <div className="bg-surface-low dark:bg-surface-low-d rounded-4xl p-5 flex flex-col items-center gap-3 w-full">
                <Mascot size={130} state={mascotState} />
                {phase === 'asking' && (
                  <div className="flex gap-1.5 justify-center">
                    <div className="typing-dot" />
                    <div className="typing-dot" />
                    <div className="typing-dot" />
                  </div>
                )}
                {phase !== 'asking' && (
                  <p className="text-xs text-center text-secondary dark:text-secondary-d leading-relaxed">
                    I only answer from your uploaded drug leaflets.
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
                textareaRef={chatInputRef}
              />
            </div>
          </div>
        )}
      </main>
      {/* ── Disclaimer footer ── */}
      <footer className="px-6 py-2 text-center text-xs text-secondary/50 dark:text-secondary-d/50 leading-relaxed">
        LARA provides information from official drug leaflets only. Not a substitute for professional medical advice — always consult your doctor or pharmacist.
      </footer>
    </div>
  )
}
