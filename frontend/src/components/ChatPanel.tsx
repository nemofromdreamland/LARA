import { useEffect, useRef, useState, type RefObject } from 'react'
import type { Message } from '../App'
import MessageBubble from './MessageBubble'

interface ChatPanelProps {
  messages: Message[]
  onSend: (question: string) => void
  disabled: boolean
  textareaRef?: RefObject<HTMLTextAreaElement>
}

const SUGGESTIONS = [
  'What are the side effects?',
  'Are there any drug interactions?',
  'What is the recommended dosage?',
  'What warnings should I know about?',
]

export default function ChatPanel({ messages, onSend, disabled, textareaRef }: ChatPanelProps) {
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const localInputRef = useRef<HTMLTextAreaElement>(null)
  const inputRef = textareaRef ?? localInputRef

  // Auto-scroll only when already near the bottom so the user can scroll up during streaming
  useEffect(() => {
    const el = scrollContainerRef.current
    if (!el) return
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120
    if (nearBottom) bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Cross-browser textarea auto-grow
  useEffect(() => {
    const el = inputRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = el.scrollHeight + 'px'
  }, [input])

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const q = input.trim()
    if (!q || disabled) return
    setInput('')
    onSend(q)
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e as unknown as React.FormEvent)
    }
  }

  function useSuggestion(s: string) {
    if (disabled) return
    onSend(s)
    // Return focus to input so it lands correctly once re-enabled
    inputRef.current?.focus()
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Screen-reader status announcer — fires once when LARA finishes, not per token */}
      <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {!disabled && messages.length > 0 && messages[messages.length - 1].role === 'lara'
          ? 'LARA has responded.'
          : ''}
      </div>

      {/* Messages */}
      <div ref={scrollContainerRef} className="flex-1 overflow-y-auto py-4 px-2 flex flex-col gap-4" aria-live="polite" aria-atomic="false" aria-relevant="additions">
        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}

        {/* Thinking indicator — shown as a LARA bubble while disabled */}
        {disabled && (
          <div className="flex items-start gap-2 fade-up" aria-label="LARA is thinking">
            <div className="w-2 flex-shrink-0" />
            <div className="bg-surface-lowest dark:bg-surface-lowest-d rounded-4xl rounded-tl-lg shadow-ambient px-5 py-4">
              <div className="flex gap-2">
                <div className="typing-dot" aria-hidden="true" />
                <div className="typing-dot" aria-hidden="true" />
                <div className="typing-dot" aria-hidden="true" />
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Suggestion chips — shown only after the first LARA welcome message */}
      {messages.length === 1 && messages[0].role === 'lara' && !input && !disabled && (
        <div className="px-2 pb-2 flex gap-2 flex-wrap">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => useSuggestion(s)}
              className="text-xs font-medium px-3 py-1.5 min-h-[44px] flex items-center rounded-full bg-surface-low dark:bg-surface-low-d text-secondary dark:text-secondary-d hover:bg-secondary-container dark:hover:bg-secondary-container-d hover:text-navy dark:hover:text-navy-d transition-colors"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* Mobile scope reminder — hidden on desktop where the mascot sidebar says this */}
      <p className="md:hidden text-xs text-center text-secondary/50 dark:text-secondary-d/50 pb-1">
        Answering from your prescription's drug leaflets only
      </p>

      {/* Input bar */}
      <form
        onSubmit={handleSubmit}
        className="flex items-end gap-3 p-3 bg-surface-low dark:bg-surface-low-d rounded-4xl"
      >
        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          rows={1}
          placeholder="Ask LARA anything about your prescription…"
          className="
            flex-1 bg-transparent resize-none outline-none text-sm text-on-surface dark:text-on-surface-d
            placeholder-secondary/60 dark:placeholder-secondary-d/60 leading-relaxed max-h-32 py-1.5
            disabled:opacity-50
          "
        />
        <button
          type="submit"
          aria-label="Send message"
          disabled={!input.trim() || disabled}
          className="
            flex-shrink-0 w-11 h-11 rounded-full flex items-center justify-center
            bg-primary-dark text-white transition-all
            hover:bg-navy active:scale-95
            disabled:opacity-40 disabled:cursor-not-allowed
          "
        >
          <svg viewBox="0 0 24 24" width={18} height={18} fill="none" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </form>
    </div>
  )
}
