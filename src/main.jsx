import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './common.css';
import './index.css'
import App from './App.jsx'
import { InvitationProvider } from './contexts/InvitationContext.jsx';

createRoot(document.getElementById('root')).render(
  <InvitationProvider>
      <App />
  </InvitationProvider>
)
