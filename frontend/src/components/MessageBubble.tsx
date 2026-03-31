import type { Message } from '../App'

interface Props {
  message: Message
}

// Render **bold** markdown in message content
function renderContent(text: string) {
  const parts = text.split(/(\*\*[^*]+\*\*)/)
  return parts.map((part, i) =>
    part.startsWith('**') && part.endsWith('**') ? (
      <strong key={i}>{part.slice(2, -2)}</strong>
    ) : (
      <span key={i}>{part}</span>
    ),
  )
}

const SECTION_LABELS: Record<string, string> = {
  boxed_warnings: 'Black Box Warning',
  indications: 'Indications',
  dosage: 'Dosage',
  contraindications: 'Contraindications',
  drug_interactions: 'Drug Interactions',
  adverse_reactions: 'Adverse Reactions',
  warnings: 'Warnings',
}

export default function MessageBubble({ message }: Props) {
  const isLara = message.role === 'lara'

  if (isLara) {
    return (
      <div className="flex items-start gap-0 fade-up">
        {/* Spacer that aligns with the mascot panel arrow */}
        <div className="w-2 flex-shrink-0" />
        <div className="lara-bubble bg-surface-lowest rounded-4xl rounded-tl-lg shadow-ambient px-5 py-4 max-w-[85%]">
          <p className="text-sm leading-relaxed text-on-surface whitespace-pre-wrap">
            {renderContent(message.content)}
          </p>

          {/* Source chips */}
          {message.sources && message.sources.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mt-3 pt-3 border-t border-surface-low">
              {message.sources.map((s, i) => (
                <span
                  key={i}
                  className="text-xs font-medium bg-primary-container text-primary-dark px-2.5 py-1 rounded-full capitalize"
                >
                  {s.drug_name} · {SECTION_LABELS[s.section] ?? s.section}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    )
  }

  // User message
  return (
    <div className="flex justify-end fade-up">
      <div className="bg-navy text-white rounded-4xl rounded-tr-lg px-5 py-3 max-w-[75%]">
        <p className="text-sm leading-relaxed">{message.content}</p>
      </div>
    </div>
  )
}
