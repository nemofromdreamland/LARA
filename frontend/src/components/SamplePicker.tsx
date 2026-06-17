import type { SampleInfo } from '../api'

interface SamplePickerProps {
  samples: SampleInfo[]
  onSelect: (sampleId: string) => void
  loadingId: string | null
  disabled: boolean
}

export default function SamplePicker({ samples, onSelect, loadingId, disabled }: SamplePickerProps) {
  if (samples.length === 0) return null

  return (
    <div className="w-full flex flex-col items-center gap-3 px-8 pb-8">
      <div className="w-full flex items-center gap-3">
        <div className="flex-1 h-px bg-secondary-container dark:bg-secondary-container-d" />
        <span className="text-xs font-medium text-secondary dark:text-secondary-d whitespace-nowrap">
          or try a sample prescription
        </span>
        <div className="flex-1 h-px bg-secondary-container dark:bg-secondary-container-d" />
      </div>

      <div className="flex gap-2 flex-wrap justify-center">
        {samples.map((s) => (
          <button
            key={s.id}
            onClick={() => onSelect(s.id)}
            className={`
              text-xs font-medium px-4 py-2 min-h-[44px] flex flex-col items-center justify-center gap-0.5 rounded-2xl
              bg-surface-low dark:bg-surface-low-d text-secondary dark:text-secondary-d
              hover:bg-secondary-container dark:hover:bg-secondary-container-d
              hover:text-navy dark:hover:text-navy-d transition-colors
              ${disabled ? 'opacity-60 pointer-events-none' : ''}
            `}
          >
            <span className="flex items-center gap-1.5">
              {loadingId === s.id && (
                <svg className="animate-spin w-3.5 h-3.5" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
                </svg>
              )}
              {s.label}
            </span>
            {s.description && (
              <span className="text-xs font-normal text-secondary/70 dark:text-secondary-d/70 leading-tight max-w-[180px] text-center">
                {s.description}
              </span>
            )}
          </button>
        ))}
      </div>
    </div>
  )
}
