// Pixel-art recreation of the LARA mascot (orange mushroom with white cap, navy outline)
// Grid: 13×13 pixels. Each cell is 1 SVG unit.

const N = '#1a2744' // navy
const O = '#f39237' // orange
const W = '#ffffff' // white
const _ = null     // transparent

const PIXELS_NORMAL: (string | null)[][] = [
  [_,_,_,_,N,N,N,N,N,_,_,_,_],
  [_,_,_,N,W,W,W,W,W,N,_,_,_],
  [_,_,N,W,W,W,N,W,W,W,N,_,_],
  [_,N,N,N,W,N,W,W,W,N,N,N,_],
  [N,N,N,N,N,N,N,N,N,N,N,N,N],
  [N,O,O,O,O,O,O,O,O,O,O,O,N],
  [N,W,O,O,O,O,O,O,O,O,O,O,N],
  [N,O,O,O,O,O,O,O,O,O,O,O,N],
  [N,O,O,N,O,O,O,O,O,N,O,O,N], // eyes
  [N,O,O,O,O,O,O,O,O,O,O,O,N],
  [N,O,O,O,N,O,O,N,O,O,O,O,N], // smile
  [N,O,O,O,O,N,N,O,O,O,O,O,N], // smile bottom
  [_,N,N,N,N,N,N,N,N,N,N,N,_],
]

// Squinted eyes for "thinking" state
const PIXELS_THINKING: (string | null)[][] = PIXELS_NORMAL.map((row, y) => {
  if (y === 8) {
    // Replace full eye squares with single pixels (squint)
    const r = [...row]
    r[3] = N; r[9] = N
    return r
  }
  return row
})

interface MascotProps {
  size?: number
  state?: 'idle' | 'uploading' | 'thinking' | 'happy'
}

export default function Mascot({ size = 160, state = 'idle' }: MascotProps) {
  const pixels = state === 'thinking' ? PIXELS_THINKING : PIXELS_NORMAL
  const cls =
    state === 'uploading'
      ? 'mascot-uploading'
      : state === 'thinking'
        ? 'mascot-thinking'
        : state === 'happy'
          ? 'mascot-happy'
          : 'mascot-idle'

  return (
    <svg
      viewBox="0 0 13 13"
      width={size}
      height={size}
      style={{ imageRendering: 'pixelated' }}
      className={cls}
    >
      {pixels.map((row, y) =>
        row.map((color, x) =>
          color ? (
            <rect
              key={`${x}-${y}`}
              x={x}
              y={y}
              width={1}
              height={1}
              fill={color}
            />
          ) : null,
        ),
      )}
    </svg>
  )
}
