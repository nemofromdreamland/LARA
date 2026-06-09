interface MascotProps {
  size?: number
  state?: 'idle' | 'uploading' | 'thinking' | 'happy' | 'error'
  className?: string
}

const STATE_CLASS: Record<NonNullable<MascotProps['state']>, string> = {
  idle:      'mascot-idle',
  uploading: 'mascot-uploading',
  thinking:  'mascot-thinking',
  happy:     'mascot-happy',
  error:     'mascot-error',
}

const STATE_ALT: Record<NonNullable<MascotProps['state']>, string> = {
  idle:      'LARA mascot',
  uploading: 'LARA is fetching your drug leaflets',
  thinking:  'LARA is thinking',
  happy:     'LARA is happy',
  error:     'LARA encountered an error',
}

export default function Mascot({ size = 160, state = 'idle', className }: MascotProps) {
  return (
    <img
      src="/mascot.png"
      alt={STATE_ALT[state]}
      width={size}
      height={size}
      className={[STATE_CLASS[state], className].filter(Boolean).join(' ')}
      style={{ imageRendering: 'auto', objectFit: 'contain' }}
      draggable={false}
    />
  )
}
