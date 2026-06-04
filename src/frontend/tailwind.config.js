/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        brand:  { DEFAULT: '#6366f1', dark: '#4f46e5' },
        bull:   '#22c55e',
        bear:   '#ef4444',
        hold:   '#f59e0b',
        surface:'#0f172a',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}

