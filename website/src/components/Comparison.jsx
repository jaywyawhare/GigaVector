import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

const LOGO_URL = 'https://raw.githubusercontent.com/jaywyawhare/GigaVector/master/docs/gigavector-logo.png'

/* ── Real brand logos (simplified for 20×20) ── */
const GigaVectorLogo = () => (
  <img src={LOGO_URL} alt="GigaVector" width="20" height="20" style={{ borderRadius: 4 }} />
)

const QdrantLogo = () => (
  <svg viewBox="0 0 174 200" width="20" height="23">
    <polygon fill="#DC244C" points="86.6,0 0,50 0,150 86.6,200 119.08,181.25 119.08,143.75 86.6,162.5 32.48,131.25 32.48,68.75 86.6,37.5 140.73,68.75 140.73,193.75 173.21,175 173.21,50"/>
    <polygon fill="#DC244C" points="54.13,81.25 54.13,118.75 86.6,137.5 119.08,118.75 119.08,81.25 86.6,62.5"/>
    <polygon fill="#9E0D38" points="119.08,143.75 119.08,181.25 86.6,200 86.6,162.5"/>
    <polygon fill="#9E0D38" points="173.21,50 173.21,175 140.73,193.75 140.73,68.75"/>
    <polygon fill="#FF516B" points="173.21,50 140.73,68.75 86.6,37.5 32.48,68.75 0,50 86.6,0"/>
    <polygon fill="#DC244C" points="86.6,162.5 86.6,200 0,150 0,50 32.48,68.75 32.48,131.25"/>
    <polygon fill="#FF516B" points="119.08,81.25 86.6,100 54.13,81.25 86.6,62.5"/>
    <polygon fill="#DC244C" points="86.6,100 86.6,137.5 54.13,118.75 54.13,81.25"/>
    <polygon fill="#9E0D38" points="119.08,81.25 119.08,118.75 86.6,137.5 86.6,100"/>
  </svg>
)

const MilvusLogo = () => (
  <svg viewBox="0 0 360 360" width="20" height="20">
    <path fill="#33B5F1" d="M168.51,63.33c31.96.51,59.94,11.11,83.28,33.19c18.7,17.68,30.23,39.33,34.81,64.64a105.32,105.32,0,0,1,1.71,20.89c-.57,28.47-9.94,53.67-28.65,75.2-18.14,20.87-40.91,34.11-68.05,39.1-36.61,6.73-69.83-1.45-99.23-24.47-6.11-4.78-11.41-10.45-16.93-15.87L40.82,221.01,7.54,189.27c-5-4.93-5-11.76-.01-16.69l33.22-32.65c10.42-10.23,20.78-20.5,31.27-30.65,6.97-6.73,13.56-13.87,21.21-19.89a119.69,119.69,0,0,1,51.35-23.69,121.16,121.16,0,0,1,23.93-2.39Zm92.37,117.69a92,92,0,0,0-1.49-15.09c-4.66-24.42-17.49-43.5-38.24-56.99-20.01-13.01-42.01-17.06-65.47-12.61a85.62,85.62,0,0,0-44.02,22.65c-8.22,7.85-16.27,15.87-24.38,23.82L55.72,174.82c-3.75,3.7-3.73,8.53-.01,12.21,5.33,5.27,10.69,10.5,16.04,15.75l39.08,38.42c18.89,18.52,45.28,28.43,67.98,25.94a83.03,83.03,0,0,0,39.17-11.53c28.24-16.82,42.66-41.88,43.93-74.6Zm58.24-.08a171.38,171.38,0,0,0-4.09-38.11,11.74,11.74,0,0,1-.2-1.16,2.63,2.63,0,0,1,1.27-2.75,2.67,2.67,0,0,1,3.16.02,9.14,9.14,0,0,1,1.19,1.09l32.31,32.3c5.31,5.31,5.3,12.07-.02,17.38l-32.2,32.2c-.28.28-.55.56-.83.83a2.73,2.73,0,0,1-3.5.44,2.68,2.68,0,0,1-1.35-3.12c1.13-5.21,2.1-10.44,2.8-15.73a164.22,164.22,0,0,0,1.47-18.86c.02-1.51,0-3.03,0-4.55Z"/>
    <circle fill="#33B5F1" cx="232.21" cy="180.97" r="70.49"/>
  </svg>
)

const PineconeLogo = () => (
  <svg viewBox="0 0 256 288" width="18" height="20">
    <path fill="#fafafa" d="M108.63,254.44c9.08,0,16.44,7.36,16.44,16.44s-7.36,16.44-16.44,16.44-16.44-7.36-16.44-16.44,7.36-16.44,16.44-16.44m91.22-30l16.25,4.81L203.2,272.78a8.47,8.47,0,0,1-8.7,6.05l-3.98-.27-.1.08-41.39-2.9,1.15-16.91,27.81,1.89-18.21-26.26,13.93-9.66,18.23,26.3Zm-176.84-30.09,16.9,1.2-1.98,27.8L64.15,205.12l9.68,13.91L47.58,237.28l26.79,7.9-4.79,16.25-43.73-12.89a8.47,8.47,0,0,1-6.06-8.73ZM132.15,170.67l30.51,36.83-13.75,11.39-18.16-21.92-5.89,33.7-17.59-3.07,5.89-33.76-24.44,14.41-9.06-15.38,41.08-24.2a8.93,8.93,0,0,1,11.41,2m85.35-24.71,15.24-8.29,22.2,40.81a8.68,8.68,0,0,1-1.93,10.69l-3.14,2.71-32.05,27.89-11.39-13.09,21.55-18.75-32.1-5.78,3.08-17.07,32.07,5.78ZM37.78,103.3l11.48,13.01-21.25,18.74,32.16,5.61-2.98,17.09-32.19-5.62,13.83,25-15.18,8.4L1.07,144.76a8.68,8.68,0,0,1,1.85-10.7Zm108.69-13.42,30.4,36.73-13.75,11.38-18.15-21.93-5.89,33.71-17.59-3.07,5.87-33.62-24.35,14.27-9.03-15.4,37.4-21.93.04-.14.17.02,3.49-2.03a8.93,8.93,0,0,1,11.39,2.01m39.18-18.07,6.65-16.02,43.01,17.85a8.68,8.68,0,0,1,5.22,9.52l-.72,3.98-7.35,41.78-17.09-3.01,4.92-27.97-28.54,15.77-8.39-15.19,28.59-15.78Zm-81.94-31.58.74,17.33-28.41,1.21,21.43,24.49-13.06,11.42L62.95,70.17l-5,28-17.08-3.05,8.18-45.76a8.67,8.67,0,0,1,8.17-7.14l4.02-.18.09-.07Zm58.12-36.97,30.27,36.97-13.81,11.31-17.96-21.94-6.06,33.67-17.57-3.16,6.07-33.74-24.53,14.34-9.01-15.42L150.43,1.22a8.93,8.93,0,0,1,11.41,2.05"/>
  </svg>
)

const WeaviateLogo = () => (
  <svg viewBox="0 0 88 42" width="24" height="12">
    <defs>
      <linearGradient id="wvGrad" x1="0%" x2="100%" y1="50%" y2="50%">
        <stop offset="0%" stopColor="#364A68" />
        <stop offset="100%" stopColor="#38D611" />
      </linearGradient>
    </defs>
    <path fill="url(#wvGrad)" d="M68.08.67v19.71l-16-9.26L36,20.38V.67l-19,11v20.85l18,10.42 17-9.84 17,9.84 18-10.42V11.69L68.08.67zM19,31.37V12.85l15-8.67V40l-15-8.63zM68.08,40l-16-9.25L36,40V22.68l16-9.24 16,9.25.08,17.31zm17-8.67l-15,8.67V4.18l15,8.67v18.48z" />
  </svg>
)

const providers = [
  { key: 'gv', name: 'GigaVector', logo: <GigaVectorLogo /> },
  { key: 'qd', name: 'Qdrant', logo: <QdrantLogo /> },
  { key: 'mv', name: 'Milvus', logo: <MilvusLogo /> },
  { key: 'pc', name: 'Pinecone', logo: <PineconeLogo /> },
  { key: 'wv', name: 'Weaviate', logo: <WeaviateLogo /> },
]

const rows = [
  { label: 'Language', gv: 'C', qd: 'Rust', mv: 'Go + C++', pc: 'Proprietary', wv: 'Go' },
  { label: 'Index types', gv: '8', qd: '1', mv: '5+', pc: '?', wv: '1' },
  { label: 'Embedded mode', gv: true, qd: false, mv: false, pc: false, wv: true },
  { label: 'Knowledge graph', gv: true, qd: false, mv: false, pc: false, wv: false },
  { label: 'LLM integration', gv: true, qd: false, mv: false, pc: true, wv: true },
  { label: 'SIMD / GPU', gv: 'AVX-512 + CUDA', qd: 'AVX2', mv: 'AVX2', pc: '—', wv: '—' },
  { label: 'Self-hosted', gv: true, qd: true, mv: true, pc: false, wv: true },
  { label: 'Graph traversal', gv: true, qd: false, mv: false, pc: false, wv: false },
  { label: 'SQL queries', gv: true, qd: false, mv: false, pc: false, wv: true },
  { label: 'Open source', gv: true, qd: true, mv: true, pc: false, wv: true },
]

/* ── Benchmark data from competitive/results (random-128, 10K vectors, ef_search=200) ── */
const benchProviders = [
  { key: 'gv', name: 'GigaVector' },
  { key: 'faiss', name: 'FAISS' },
  { key: 'chroma', name: 'ChromaDB' },
  { key: 'wv', name: 'Weaviate' },
  { key: 'qd', name: 'Qdrant' },
]

const benchRows = [
  { label: 'Recall@10', gv: '95.5%', faiss: '97.6%', chroma: '88.4%', wv: '88.2%', qd: '100%*' },
  { label: 'QPS', gv: '5,026', faiss: '4,765', chroma: '1,267', wv: '742', qd: '238' },
  { label: 'P99 Latency', gv: '0.53ms', faiss: '0.63ms', chroma: '1.17ms', wv: '1.75ms', qd: '6.67ms' },
  { label: 'Peak QPS', gv: '44,617', faiss: '35,296', chroma: '1,106', wv: '794', qd: '244' },
  { label: 'Build Time', gv: '2.2s', faiss: '0.8s', chroma: '1.4s', wv: '5.2s', qd: '2.5s' },
]

function Cell({ val, highlight }) {
  const cls = highlight ? 'cmp-cell cmp-hl' : 'cmp-cell'
  if (val === true) return <td className={cls}><span className="cmp-yes">Yes</span></td>
  if (val === false) return <td className={cls}><span className="cmp-no">—</span></td>
  return <td className={cls}>{val}</td>
}

export default function Comparison() {
  const ref = useRef(null)
  const isReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches

  useEffect(() => {
    if (isReduced) return
    const ctx = gsap.context(() => {
      gsap.fromTo(ref.current,
        { y: 30, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.6, ease: 'power2.out',
          scrollTrigger: { trigger: ref.current, start: 'top 80%' } }
      )
    })
    return () => ctx.revert()
  }, [])

  return (
    <section id="comparison" ref={ref} style={{ opacity: isReduced ? 1 : 0 }}>
      <div style={{ marginBottom: 48 }}>
        <div className="section-label">Comparison</div>
        <h2 className="section-heading">How GigaVector stacks up.</h2>
      </div>

      <div className="cmp-scroll">
        <table className="cmp-table">
          <thead>
            <tr>
              <th className="cmp-feature-th">Feature</th>
              {providers.map(p => (
                <th key={p.key} className={p.key === 'gv' ? 'cmp-th cmp-hl-th' : 'cmp-th'}>
                  <span className="cmp-provider">
                    {p.logo}
                    {p.name}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td className="cmp-label">{r.label}</td>
                <Cell val={r.gv} highlight />
                <Cell val={r.qd} />
                <Cell val={r.mv} />
                <Cell val={r.pc} />
                <Cell val={r.wv} />
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ marginTop: 64, marginBottom: 48 }}>
        <div className="section-label">Benchmarks</div>
        <h2 className="section-heading">Real numbers. Not marketing.</h2>
        <p className="section-sub">HNSW, 10K vectors, 128D, ef_search=200. Same hardware, same dataset.</p>
      </div>

      <div className="cmp-scroll">
        <table className="cmp-table">
          <thead>
            <tr>
              <th className="cmp-feature-th">Metric</th>
              {benchProviders.map(p => (
                <th key={p.key} className={p.key === 'gv' ? 'cmp-th cmp-hl-th' : 'cmp-th'}>
                  {p.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {benchRows.map((r, i) => (
              <tr key={i}>
                <td className="cmp-label">{r.label}</td>
                <td className="cmp-cell cmp-hl cmp-bench-best">{r.gv}</td>
                <td className="cmp-cell">{r.faiss}</td>
                <td className="cmp-cell">{r.chroma}</td>
                <td className="cmp-cell">{r.wv}</td>
                <td className="cmp-cell">{r.qd}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="cmp-footnote">
        * Qdrant client uses exact (brute-force) search, not HNSW — recall is 100% but at significantly lower throughput.
      </p>
    </section>
  )
}
