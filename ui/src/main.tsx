import { StrictMode } from 'react'
import { MotionConfig } from 'motion/react'
import { createRoot } from 'react-dom/client'
import './theme.css'
import App from './App'

// "reduce motion" in the operating system is honoured by the CSS animations and,
// through this, by every entrance and transition the motion library runs too.
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <MotionConfig reducedMotion="user"><App /></MotionConfig>
  </StrictMode>,
)
