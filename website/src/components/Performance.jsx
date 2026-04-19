import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

export default function Performance() {
  const ref = useRef(null)
  const isReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches

  useEffect(() => {
    if (isReduced) return
    const ctx = gsap.context(() => {
      gsap.fromTo(ref.current.children,
        { y: 20, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.5, stagger: 0.08, ease: 'power2.out',
          scrollTrigger: { trigger: ref.current, start: 'top 80%' } }
      )
    })
    return () => ctx.revert()
  }, [])

  return (
    <div id="performance" ref={ref} className="numbers">
      <div className="num-item" style={{ opacity: isReduced ? 1 : 0 }}>
        <div className="num-val">&lt;1ms</div>
        <div className="num-label">p99 Latency</div>
      </div>
      <div className="num-item" style={{ opacity: isReduced ? 1 : 0 }}>
        <div className="num-val">99.2%</div>
        <div className="num-label">Recall@10</div>
      </div>
      <div className="num-item" style={{ opacity: isReduced ? 1 : 0 }}>
        <div className="num-val">50K+</div>
        <div className="num-label">QPS</div>
      </div>
      <div className="num-item" style={{ opacity: isReduced ? 1 : 0 }}>
        <div className="num-val">8</div>
        <div className="num-label">Index Types</div>
      </div>
      <div className="num-item" style={{ opacity: isReduced ? 1 : 0 }}>
        <div className="num-val">0</div>
        <div className="num-label">Dependencies</div>
      </div>
    </div>
  )
}
