import { useRef, useState } from 'react'

interface UploadZoneProps {
  onUpload: (file: File) => void
  loading: boolean
}

export default function UploadZone({ onUpload, loading }: UploadZoneProps) {
  const [dragging, setDragging] = useState(false)
  const [fileError, setFileError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  function validate(file: File): boolean {
    if (file.type !== 'application/pdf') {
      setFileError('Only PDF files are accepted.')
      return false
    }
    setFileError(null)
    return true
  }

  function handleFile(file: File) {
    if (validate(file)) onUpload(file)
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }

  return (
    <div className="h-full flex flex-col items-center justify-center p-8 gap-5">
      <div
        className={`
          w-full flex-1 flex flex-col items-center justify-center gap-4
          rounded-4xl border-2 border-dashed cursor-pointer transition-all duration-200
          ${dragging ? 'dropzone-active border-primary' : 'border-secondary-container hover:border-primary'}
          ${loading ? 'opacity-60 pointer-events-none' : ''}
        `}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          className="hidden"
          onChange={handleChange}
        />

        {/* Icon */}
        <div className={`
          w-16 h-16 rounded-3xl flex items-center justify-center transition-colors
          ${dragging ? 'bg-primary' : 'bg-primary-container'}
        `}>
          <svg
            viewBox="0 0 24 24"
            width={28}
            height={28}
            fill="none"
            stroke={dragging ? '#ffffff' : '#904d00'}
            strokeWidth={1.8}
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="12" y1="18" x2="12" y2="12" />
            <polyline points="9 15 12 12 15 15" />
          </svg>
        </div>

        <div className="text-center px-4">
          <p className="font-semibold text-navy">
            {dragging ? 'Drop it here!' : 'Drop your prescription'}
          </p>
          <p className="text-sm text-secondary mt-1">
            PDF only · or{' '}
            <span className="text-primary font-medium underline underline-offset-2">
              browse files
            </span>
          </p>
        </div>
      </div>

      {fileError && (
        <p className="text-sm text-red-500 font-medium">{fileError}</p>
      )}

      {loading && (
        <div className="flex items-center gap-2 text-sm text-secondary">
          <svg className="animate-spin w-4 h-4 text-primary" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
          </svg>
          Fetching leaflets from DailyMed…
        </div>
      )}
    </div>
  )
}
