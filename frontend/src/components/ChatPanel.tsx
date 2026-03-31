import { useEffect, useRef, useState } from 'react'
import type { Message } from '../App'
import MessageBubble from './MessageBubble'

interface ChatPanelProps {
  messages: Message[]
  onSend: (question: string) => void
  disabled: boolean
}

const SUGGESTIONS = [
  'What are the side effects?',
  'Are there any drug interactions?',
  'What is the recommended dosage?',
  'What warnings should I know about?',
]

export default function ChatPanel({ messages, onSend, disabled }: ChatPanelProps) {
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

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
    setInput(s)
    inputRef.current?.focus()
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto py-4 px-2 flex flex-col gap-4">
        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}

        {/* Thinking indicator — shown as a LARA bubble while disabled */}
        {disabled && (
          <div className="flex items-start gap-2 fade-up">
            <div className="w-2 flex-shrink-0" />
            <div className="bg-surface-lowest rounded-4xl rounded-tl-lg shadow-ambient px-5 py-4">
              <div className="flex gap-2">
                <div className="typing-dot" />
                <div className="typing-dot" />
                <div className="typing-dot" />
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Suggestion chips — shown only when no questions asked yet */}
      {messages.length === 1 && !disabled && (
        <div className="px-2 pb-2 flex gap-2 flex-wrap">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => useSuggestion(s)}
              className="text-xs font-medium px-3 py-1.5 rounded-full bg-surface-low text-secondary hover:bg-secondary-container hover:text-navy transition-colors"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* Input bar */}
      <form
        onSubmit={handleSubmit}
        className="flex items-end gap-3 p-3 bg-surface-low rounded-4xl"
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
            flex-1 bg-transparent resize-none outline-none text-sm text-on-surface
            placeholder-secondary/60 leading-relaxed max-h-32 py-1.5
            disabled:opacity-50
          "
          style={{ fieldSizing: 'content' } as React.CSSProperties}
        />
        <button
          type="submit"
          disabled={!input.trim() || disabled}
          className="
            flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center
            bg-primary text-white transition-all
            hover:bg-primary-dark active:scale-95
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
