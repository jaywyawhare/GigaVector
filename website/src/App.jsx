import { Routes, Route, Outlet, Link } from 'react-router-dom'
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

function Layout() {
  return (
    <>
      <Navbar />
      <Outlet />
      <Footer />
    </>
  )
}

function NotFound() {
  return (
    <section style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
      <div>
        <img src="/404.webp" alt="404" style={{ maxWidth: '600px', marginBottom: '24px' }} />
        <h2 className="section-heading" style={{ marginBottom: '24px' }}>Page not found.</h2>
        <Link to="/" className="btn primary">
          Go back to Home
        </Link>
      </div>
    </section>
  )
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Landing />} />
        <Route path="/docs" element={<DocsPage />} />
        <Route path="/docs/:slug" element={<DocsPage />} />
      </Route>
      <Route path="*" element={<NotFound />} />
    </Routes>
  )
}
