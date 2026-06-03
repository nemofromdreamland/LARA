/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // ── Light mode ──────────────────────────────────────────────────
        primary: '#f39237',
        'primary-dark': '#904d00',
        'primary-container': '#ffe0b2',
        secondary: '#525f71',
        'secondary-container': '#d3e1f6',
        surface: '#fcf9f4',
        'surface-low': '#f6f3ee',
        'surface-lowest': '#ffffff',
        'on-surface': '#1c1c19',
        navy: '#1a2744',
        // ── Dark mode ───────────────────────────────────────────────────
        // Warm near-black backgrounds (mirrors cream warmth from light mode)
        'surface-d': '#1a1a17',
        'surface-low-d': '#222220',
        'surface-lowest-d': '#2a2a27',
        'on-surface-d': '#f0ede8',
        'secondary-d': '#8a97a8',
        'navy-d': '#e8edf5',
        'secondary-container-d': '#1e2d3d',
        'primary-container-d': '#3d2000',
        'primary-text-d': '#ffb870',
        'user-bubble-d': '#162040',
      },
      fontFamily: {
        sans: ['Lexend', 'ui-sans-serif', 'system-ui'],
      },
      borderRadius: {
        '4xl': '2rem',
        '5xl': '3rem',
      },
      boxShadow: {
        ambient: '0 8px 40px 0 rgba(28, 28, 25, 0.06)',
        'ambient-lg': '0 20px 60px 0 rgba(28, 28, 25, 0.08)',
      },
    },
  },
  plugins: [],
}
