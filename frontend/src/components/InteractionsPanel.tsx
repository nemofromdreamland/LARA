import { useState } from 'react'
import type { InteractionsResult } from '../api'
import { drugColorClass } from '../utils/drugColors'

interface Props {
  result: InteractionsResult
  drugs: string[]
  onDismiss: () => void
}

export default function InteractionsPanel({ result, drugs, onDismiss }: Props) {
  const [expanded, setExpanded] = useState<number | null>(null)

  return (
    <div className="bg-surface-lowest dark:bg-surface-lowest-d rounded-3xl shadow-ambient px-4 py-3 flex-shrink-0 max-h-56 overflow-y-auto">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-on-surface dark:text-on-surface-d">Drug Interactions</span>
          <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
            result.interactions.length > 0
              ? 'bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300'
              : 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200'
          }`}>
            {result.interactions.length > 0
              ? `${result.interactions.length} found`
              : 'None found'}
          </span>
          <span className="text-xs text-secondary dark:text-secondary-d">
            {result.pairs_checked} pair{result.pairs_checked !== 1 ? 's' : ''} checked
          </span>
        </div>
        <button
          onClick={onDismiss}
          aria-label="Dismiss interactions panel"
          className="w-7 h-7 flex items-center justify-center rounded-full text-secondary dark:text-secondary-d hover:bg-surface-low dark:hover:bg-surface-low-d transition-colors text-base leading-none"
        >
          ×
        </button>
      </div>

      {result.interactions.length === 0 ? (
        <p className="text-xs text-secondary dark:text-secondary-d leading-relaxed">
          No interactions were detected between these drugs based on their FDA leaflets. Always consult your pharmacist for a complete review.
        </p>
      ) : (
        <div className="flex flex-col gap-1.5">
          {result.interactions.map((interaction, i) => (
            <div key={i} className="rounded-2xl bg-surface-low dark:bg-surface-low-d px-3 py-2">
              <button
                onClick={() => setExpanded(expanded === i ? null : i)}
                className="w-full flex items-center gap-2 text-left"
              >
                <span className={`text-xs font-medium px-2 py-0.5 rounded-full capitalize ${drugColorClass(interaction.drug_a, drugs)}`}>
                  {interaction.drug_a}
                </span>
                <span className="text-xs text-secondary dark:text-secondary-d">mentions</span>
                <span className={`text-xs font-medium px-2 py-0.5 rounded-full capitalize ${drugColorClass(interaction.drug_b, drugs)}`}>
                  {interaction.drug_b}
                </span>
                <svg
                  className={`ml-auto w-3.5 h-3.5 text-secondary dark:text-secondary-d transition-transform ${expanded === i ? 'rotate-180' : ''}`}
                  viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5}
                >
                  <polyline points="6 9 12 15 18 9" />
                </svg>
              </button>
              {expanded === i && (
                <p className="mt-1.5 text-xs text-secondary dark:text-secondary-d leading-relaxed italic border-t border-surface dark:border-surface-d pt-1.5">
                  "{interaction.excerpt}"
                </p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
