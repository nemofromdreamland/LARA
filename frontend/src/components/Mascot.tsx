interface MascotProps {
  size?: number
  state?: 'idle' | 'uploading' | 'thinking' | 'happy'
}

export default function Mascot({ size = 160, state = 'idle' }: MascotProps) {
  const cls =
    state === 'uploading'
      ? 'mascot-uploading'
      : state === 'thinking'
        ? 'mascot-thinking'
        : state === 'happy'
          ? 'mascot-happy'
          : 'mascot-idle'

  return (
    <img
      src="/mascot.png"
      alt="LARA mascot"
      width={size}
      height={size}
      className={cls}
      style={{ imageRendering: 'pixelated', objectFit: 'contain' }}
      draggable={false}
    />
  )
}
