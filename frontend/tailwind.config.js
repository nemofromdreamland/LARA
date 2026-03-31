/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
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
