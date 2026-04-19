import { useEffect, useRef, useState, useCallback } from 'react'
import gsap from 'gsap'
import { useVectorScene } from '../hooks/useVectorScene'

export default function Hero() {
  const canvasRef = useVectorScene()
  const inner = useRef(null)
  const [copied, setCopied] = useState(false)
  const [copyFailed, setCopyFailed] = useState(false)
  const isReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches

  const copy = useCallback(() => {
    navigator.clipboard.writeText('pip install gigavector')
      .then(() => {
        setCopyFailed(false)
        setCopied(true)
        setTimeout(() => setCopied(false), 1500)
      })
      .catch(() => {
        setCopyFailed(true)
        setTimeout(() => setCopyFailed(false), 1500)
      })
  }, [])

  useEffect(() => {
    if (isReduced) return
    const ctx = gsap.context(() => {
      const tl = gsap.timeline({ delay: 0.3, defaults: { ease: 'power3.out' } })
      tl.fromTo('.hero-badge', { y: 16, opacity: 0 }, { y: 0, opacity: 1, duration: 0.5 })
        .fromTo('.hero-h1', { y: 24, opacity: 0 }, { y: 0, opacity: 1, duration: 0.6 }, '-=0.3')
        .fromTo('.hero-desc', { y: 20, opacity: 0 }, { y: 0, opacity: 1, duration: 0.5 }, '-=0.35')
        .fromTo('.hero-actions', { y: 16, opacity: 0 }, { y: 0, opacity: 1, duration: 0.5 }, '-=0.3')
        .fromTo('.hero-install', { y: 12, opacity: 0 }, { y: 0, opacity: 1, duration: 0.4 }, '-=0.25')
    }, inner)
    return () => ctx.revert()
  }, [])

  return (
    <section className="hero">
      <canvas ref={canvasRef} className="hero-canvas" />
      <div className="hero-fade" />

      <div ref={inner} className="hero-inner">
        <div className="hero-badge" style={{ opacity: isReduced ? 1 : 0 }}>
          <span className="hero-dot" />
          v0.8.2 shipped
        </div>

        <h1 className="hero-h1" style={{ opacity: isReduced ? 1 : 0 }}>
          Vector search<br />
          <em>at the speed of C.</em>
        </h1>

        <p className="hero-desc" style={{ opacity: isReduced ? 1 : 0 }}>
          Production-ready vector database. Pure C core, Python bindings.
          8 index algorithms, SIMD-optimized, distributed, with a built-in knowledge graph.
        </p>

        <div className="hero-actions" style={{ opacity: isReduced ? 1 : 0 }}>
          <a className="btn primary" href="#code">Get started</a>
          <a className="btn" href="https://github.com/jaywyawhare/GigaVector" target="_blank" rel="noopener">
            View on GitHub
          </a>
        </div>

        <button className="hero-install" style={{ opacity: isReduced ? 1 : 0 }} onClick={copy}>
          {copied ? 'copied!' : copyFailed ? 'copy failed' : '$ pip install gigavector'}
        </button>
      </div>
    </section>
  )
}
