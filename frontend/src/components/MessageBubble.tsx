import React, { useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import type { Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Message } from '../App'
import { drugColorClass } from '../utils/drugColors'

interface Props {
  message: Message
  drugs: string[]
}

const SECTION_LABELS: Record<string, string> = {
  boxed_warnings: 'Black Box Warning',
  indications: 'Indications',
  dosage: 'Dosage',
  contraindications: 'Contraindications',
  drug_interactions: 'Drug Interactions',
  adverse_reactions: 'Adverse Reactions',
  warnings: 'Warnings',
  warnings_and_precautions: 'Warnings & Precautions',
  pregnancy: 'Pregnancy',
  teratogenic_effects: 'Teratogenic Effects',
  nonteratogenic_effects: 'Non-teratogenic Effects',
  nursing_mothers: 'Nursing Mothers',
  pediatric_use: 'Pediatric Use',
  geriatric_use: 'Geriatric Use',
}

function escapeRe(s: string) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function walkDrugs(node: React.ReactNode, drugs: string[], pattern: RegExp): React.ReactNode {
  if (typeof node === 'string') {
    const parts = node.split(pattern)
    if (parts.length === 1) return node
    return parts.map((part, i) => {
      const matched = drugs.find((d) => d.toLowerCase() === part.toLowerCase())
      if (matched) {
        return (
          <span key={i} className={`font-semibold rounded-sm px-0.5 ${drugColorClass(matched, drugs)}`}>
            {part}
          </span>
        )
      }
      return part || undefined
    })
  }
  if (React.isValidElement(node)) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const el = node as React.ReactElement<any>
    if (el.type === 'code' || el.type === 'pre') return el
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const children = (el.props as any).children as React.ReactNode | undefined
    if (children === undefined) return el
    return React.cloneElement(el, { children: walkDrugs(children, drugs, pattern) })
  }
  if (Array.isArray(node)) {
    return node.map((child, i) => (
      <React.Fragment key={i}>{walkDrugs(child, drugs, pattern)}</React.Fragment>
    ))
  }
  return node
}

function makeComponents(drugs: string[], isComplete: boolean): Components {
  function highlight(children: React.ReactNode): React.ReactNode {
    if (!isComplete || drugs.length === 0) return children
    const pattern = new RegExp(`(${drugs.map(escapeRe).join('|')})`, 'gi')
    return walkDrugs(children, drugs, pattern)
  }

  return {
    p: ({ children }) => (
      <p className="mb-2 last:mb-0 leading-relaxed">{highlight(children)}</p>
    ),
    ul: ({ children }) => <ul className="list-disc pl-4 mb-2 space-y-0.5">{children}</ul>,
    ol: ({ children }) => <ol className="list-decimal pl-4 mb-2 space-y-0.5">{children}</ol>,
    li: ({ children }) => <li className="leading-relaxed">{highlight(children)}</li>,
    h1: ({ children }) => <h1 className="text-base font-bold mb-1 mt-3 first:mt-0">{children}</h1>,
    h2: ({ children }) => <h2 className="text-sm font-bold mb-1 mt-3 first:mt-0">{children}</h2>,
    h3: ({ children }) => <h3 className="text-sm font-semibold mb-1 mt-2 first:mt-0">{children}</h3>,
    strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
    em: ({ children }) => <em className="italic">{children}</em>,
    code: ({ children }) => (
      <code className="bg-surface-low dark:bg-surface-low-d rounded px-1 py-0.5 text-xs font-mono">
        {children}
      </code>
    ),
  }
}

function MessageBubble({ message, drugs }: Props) {
  const [copied, setCopied] = useState(false)
  const isLara = message.role === 'lara'
  const isComplete = !!message.sources

  const markdownComponents = useMemo(
    () => makeComponents(drugs, isComplete),
    [drugs, isComplete],
  )

  if (isLara) {
    return (
      <div className="flex items-start gap-0 fade-up">
        {/* Spacer that aligns with the mascot panel arrow */}
        <div className="w-2 flex-shrink-0" />
        <div className="group relative lara-bubble bg-surface-lowest dark:bg-surface-lowest-d rounded-4xl rounded-tl-lg shadow-ambient px-5 py-4 max-w-[85%]">
          {/* Copy button — visible on hover */}
          <button
            onClick={() => {
              navigator.clipboard.writeText(message.content)
              setCopied(true)
              setTimeout(() => setCopied(false), 2000)
            }}
            aria-label="Copy answer"
            className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity w-7 h-7 flex items-center justify-center rounded-full text-secondary dark:text-secondary-d hover:bg-surface-low dark:hover:bg-surface-low-d"
          >
            {copied ? (
              <svg viewBox="0 0 24 24" width={14} height={14} fill="none" stroke="currentColor" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round">
                <polyline points="20 6 9 17 4 12" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" width={14} height={14} fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
              </svg>
            )}
          </button>

          <div className="text-sm leading-relaxed text-on-surface dark:text-on-surface-d pr-5">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
              {message.content}
            </ReactMarkdown>
          </div>

          {/* Source chips */}
          {message.sources && message.sources.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mt-3 pt-3 border-t border-surface-low dark:border-surface-low-d">
              {message.sources.map((s, i) => {
                const isBlackBox = s.section === 'boxed_warnings'
                return (
                  <span
                    key={i}
                    className={`text-xs font-medium px-2.5 py-1 rounded-full capitalize ${
                      isBlackBox
                        ? 'bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300'
                        : drugColorClass(s.drug_name, drugs)
                    }`}
                  >
                    {isBlackBox ? '⚠ ' : ''}{s.drug_name} · {SECTION_LABELS[s.section] ?? s.section}
                  </span>
                )
              })}
            </div>
          )}
        </div>
      </div>
    )
  }

  // User message
  return (
    <div className="flex justify-end fade-up">
      <div className="bg-navy dark:bg-user-bubble-d text-white dark:text-on-surface-d rounded-4xl rounded-tr-lg px-5 py-3 max-w-[75%]">
        <p className="text-sm leading-relaxed">{message.content}</p>
      </div>
    </div>
  )
}

export default React.memo(MessageBubble)
