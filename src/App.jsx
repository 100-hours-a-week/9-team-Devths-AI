import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { BasicInfoPage } from './pages/BasicInfoPage'
import { routes } from './router'

function App() {
  const [count, setCount] = useState(0)

  return (
    <div className="app">
      <BrowserRouter>
          <main className="main-page">
              <Routes>
                {routes.map((r) => (
                  <Route key={r.path} path={r.path} element={r.element} />
                ))}
              </Routes>
          </main>
      </BrowserRouter>
    </div>
  );
}

export default App
