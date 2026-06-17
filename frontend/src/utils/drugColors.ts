const DRUG_PALETTE = [
  'bg-teal-100 dark:bg-teal-900 text-teal-800 dark:text-teal-200',
  'bg-violet-100 dark:bg-violet-900 text-violet-800 dark:text-violet-200',
  'bg-emerald-100 dark:bg-emerald-900 text-emerald-800 dark:text-emerald-200',
  'bg-pink-100 dark:bg-pink-900 text-pink-800 dark:text-pink-200',
  'bg-sky-100 dark:bg-sky-900 text-sky-800 dark:text-sky-200',
] as const

export function drugColorClass(drugName: string, drugs: string[]): string {
  const idx = drugs.findIndex((d) => d.toLowerCase() === drugName.toLowerCase())
  return DRUG_PALETTE[Math.max(0, idx) % DRUG_PALETTE.length]
}
