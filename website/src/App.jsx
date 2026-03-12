import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import Features from './components/Features'
import CodeShowcase from './components/CodeShowcase'
import Performance from './components/Performance'
import Comparison from './components/Comparison'
import CTA from './components/CTA'
import Footer from './components/Footer'
import DocsPage from './components/DocsPage'

function Landing() {
  return (
    <>
      <Hero />
      <div className="divider" />
      <Features />
      <div className="divider" />
      <CodeShowcase />
      <div className="divider" />
      <Comparison />
      <div className="divider" />
      <Performance />
      <CTA />
    </>
  )
}

export default function App() {
  return (
    <>
      <Navbar />
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/docs" element={<DocsPage />} />
        <Route path="/docs/:slug" element={<DocsPage />} />
      </Routes>
      <Footer />
    </>
  )
}
