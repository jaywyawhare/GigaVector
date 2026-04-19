import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

const features = [
  {
    name: 'Hybrid Search',
    text: 'Vector similarity, BM25 full-text, and geo-spatial filtering with RRF fusion. SQL query interface included.',
    tags: ['k-NN', 'BM25', 'ColBERT', 'SQL'],
  },
  {
    name: '8 Index Types',
    text: 'HNSW, IVF-PQ, IVF-Flat, DiskANN, Flat, KD-Tree, LSH, Sparse. Auto-index suggestion for your workload.',
    tags: ['HNSW', 'DiskANN', 'IVF-PQ'],
  },
  {
    name: 'SIMD Optimized',
    text: 'SSE4.2, AVX2, AVX-512F distance computation. Optional CUDA GPU acceleration for batch operations.',
    tags: ['AVX-512', 'CUDA'],
  },
  {
    name: 'Distributed',
    text: 'Hash/range sharding, leader-follower replication, quorum consistency, multi-tenant namespaces.',
    tags: ['Sharding', 'gRPC', 'TLS'],
  },
  {
    name: 'Knowledge Graph',
    text: 'Property graph with BFS/DFS/Dijkstra, PageRank, SPO triple store, entity resolution, link prediction.',
    tags: ['PageRank', 'Triples'],
  },
  {
    name: 'AI-Native',
    text: 'OpenAI, Anthropic, Gemini integrations. Auto-embedding, semantic memory, ONNX model serving.',
    tags: ['LLM', 'ONNX', 'Embeddings'],
  },
  {
    name: 'Crash-Safe',
    text: 'WAL with automatic replay, point-in-time snapshots, incremental backup, mmap I/O, background compaction.',
    tags: ['WAL', 'Snapshots', 'mmap'],
  },
  {
    name: 'Security',
    text: 'JWT/API key auth, RBAC with per-collection permissions, OIDC/SAML SSO, TLS 1.3 encryption.',
    tags: ['RBAC', 'JWT', 'SSO'],
  },
  {
    name: 'Dashboard',
    text: 'Built-in web dashboard with live stats, vector browser, search console, schema viewer, cluster monitoring.',
    tags: ['REST', 'OpenAPI'],
  },
]

export default function Features() {
  const ref = useRef(null)
  const isReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches

  useEffect(() => {
    if (isReduced) return
    const el = ref.current
    if (!el) return

    const ctx = gsap.context(() => {
      gsap.fromTo('.section-header-feat',
        { y: 24, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.6, ease: 'power2.out',
          scrollTrigger: { trigger: el, start: 'top 85%' } }
      )
      gsap.fromTo(el.querySelectorAll('.feat'),
        { y: 30, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.5, stagger: 0.04, ease: 'power2.out',
          scrollTrigger: { trigger: el, start: 'top 75%' } }
      )
    })
    return () => ctx.revert()
  }, [])

  return (
    <section id="features" ref={ref}>
      <div className="section-header-feat" style={{ opacity: isReduced ? 1 : 0, marginBottom: 48 }}>
        <div className="section-label">Capabilities</div>
        <h2 className="section-heading">Everything you need. Nothing you don't.</h2>
        <p className="section-sub">From prototyping to production clusters.</p>
      </div>

      <div className="feat-grid">
        {features.map((f, i) => (
          <div key={i} className="feat" style={{ opacity: isReduced ? 1 : 0 }}>
            <div className="feat-name">{f.name}</div>
            <div className="feat-text">{f.text}</div>
            <div className="feat-tags">
              {f.tags.map((t, j) => <span key={j} className="feat-tag">{t}</span>)}
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}
