import { useState, useCallback, useEffect, useRef } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

const tabs = [
  { id: 'quick', label: 'Quick Start' },
  { id: 'search', label: 'Search' },
  { id: 'graph', label: 'Graph' },
  { id: 'server', label: 'Server' },
]

const code = {
  quick: {
    plain: `from gigavector import Database, DistanceType, IndexType

with Database.open("vectors.db", dimension=128, index=IndexType.HNSW) as db:
    db.add_vector([0.1] * 128, metadata={"category": "science"})
    results = db.search([0.1] * 128, k=10, distance=DistanceType.COSINE)
    for hit in results:
        print(f"  distance={hit.distance:.4f}")`,
    jsx: (
      <>
        <span className="kw">from</span> <span className="cls">gigavector</span> <span className="kw">import</span> Database, DistanceType, IndexType{'\n\n'}
        <span className="kw">with</span> Database.<span className="fn">open</span>(<span className="str">"vectors.db"</span>, <span className="param">dimension</span>=<span className="num">128</span>, <span className="param">index</span>=IndexType.HNSW) <span className="kw">as</span> db:{'\n'}
        {'    '}db.<span className="fn">add_vector</span>([<span className="num">0.1</span>] * <span className="num">128</span>, <span className="param">metadata</span>={'{'}<span className="str">"category"</span>: <span className="str">"science"</span>{'}'}){'\n'}
        {'    '}results = db.<span className="fn">search</span>([<span className="num">0.1</span>] * <span className="num">128</span>, <span className="param">k</span>=<span className="num">10</span>, <span className="param">distance</span>=DistanceType.COSINE){'\n'}
        {'    '}<span className="kw">for</span> hit <span className="kw">in</span> results:{'\n'}
        {'    '}{'    '}<span className="fn">print</span>(<span className="str">f"  distance={'{'}</span>hit.distance<span className="str">:.4f{'}'}"</span>)
      </>
    ),
  },
  search: {
    plain: `from gigavector import SearchParams, HNSWConfig, GroupedSearch, GroupSearchConfig

cfg = HNSWConfig(M=32, ef_construction=200, ef_search=100)
db = Database.open("prod.db", dimension=768, index=IndexType.HNSW, hnsw_config=cfg)

results = db.search_with_params(query, k=20,
    params=SearchParams(ef_search=200))

groups = GroupedSearch(db).search(query,
    group_by="category",
    config=GroupSearchConfig(group_size=3, num_groups=5))`,
    jsx: (
      <>
        <span className="kw">from</span> <span className="cls">gigavector</span> <span className="kw">import</span> SearchParams, HNSWConfig, GroupedSearch, GroupSearchConfig{'\n\n'}
        cfg = <span className="cls">HNSWConfig</span>(<span className="param">M</span>=<span className="num">32</span>, <span className="param">ef_construction</span>=<span className="num">200</span>, <span className="param">ef_search</span>=<span className="num">100</span>){'\n'}
        db = Database.<span className="fn">open</span>(<span className="str">"prod.db"</span>, <span className="param">dimension</span>=<span className="num">768</span>, <span className="param">index</span>=IndexType.HNSW, <span className="param">hnsw_config</span>=cfg){'\n\n'}
        results = db.<span className="fn">search_with_params</span>(query, <span className="param">k</span>=<span className="num">20</span>,{'\n'}
        {'    '}<span className="param">params</span>=<span className="cls">SearchParams</span>(<span className="param">ef_search</span>=<span className="num">200</span>)){'\n\n'}
        groups = <span className="cls">GroupedSearch</span>(db).<span className="fn">search</span>(query,{'\n'}
        {'    '}<span className="param">group_by</span>=<span className="str">"category"</span>,{'\n'}
        {'    '}<span className="param">config</span>=<span className="cls">GroupSearchConfig</span>(<span className="param">group_size</span>=<span className="num">3</span>, <span className="param">num_groups</span>=<span className="num">5</span>))
      </>
    ),
  },
  graph: {
    plain: `from gigavector import KnowledgeGraph, KGConfig

kg = KnowledgeGraph(KGConfig(embedding_dimension=128))
alice = kg.add_entity("Alice", "Person", embedding=[0.1] * 128)
corp = kg.add_entity("Anthropic", "Company", embedding=[0.2] * 128)
kg.add_relation(alice, "works_at", corp, weight=1.0)

results = kg.hybrid_search(query,
    entity_type="Person", predicate_filter="works_at", k=10)`,
    jsx: (
      <>
        <span className="kw">from</span> <span className="cls">gigavector</span> <span className="kw">import</span> KnowledgeGraph, KGConfig{'\n\n'}
        kg = <span className="cls">KnowledgeGraph</span>(<span className="cls">KGConfig</span>(<span className="param">embedding_dimension</span>=<span className="num">128</span>)){'\n'}
        alice = kg.<span className="fn">add_entity</span>(<span className="str">"Alice"</span>, <span className="str">"Person"</span>, <span className="param">embedding</span>=[<span className="num">0.1</span>] * <span className="num">128</span>){'\n'}
        corp = kg.<span className="fn">add_entity</span>(<span className="str">"Anthropic"</span>, <span className="str">"Company"</span>, <span className="param">embedding</span>=[<span className="num">0.2</span>] * <span className="num">128</span>){'\n'}
        kg.<span className="fn">add_relation</span>(alice, <span className="str">"works_at"</span>, corp, <span className="param">weight</span>=<span className="num">1.0</span>){'\n\n'}
        results = kg.<span className="fn">hybrid_search</span>(query,{'\n'}
        {'    '}<span className="param">entity_type</span>=<span className="str">"Person"</span>, <span className="param">predicate_filter</span>=<span className="str">"works_at"</span>, <span className="param">k</span>=<span className="num">10</span>)
      </>
    ),
  },
  server: {
    plain: `from gigavector import Database, IndexType, serve_with_dashboard

db = Database.open("prod.db", dimension=768, index=IndexType.HNSW)
server = serve_with_dashboard(db, port=6969)
# Dashboard:  http://localhost:6969/dashboard
# REST API:   http://localhost:6969/
server.stop()`,
    jsx: (
      <>
        <span className="kw">from</span> <span className="cls">gigavector</span> <span className="kw">import</span> Database, IndexType, serve_with_dashboard{'\n\n'}
        db = Database.<span className="fn">open</span>(<span className="str">"prod.db"</span>, <span className="param">dimension</span>=<span className="num">768</span>, <span className="param">index</span>=IndexType.HNSW){'\n'}
        server = <span className="fn">serve_with_dashboard</span>(db, <span className="param">port</span>=<span className="num">6969</span>){'\n'}
        <span className="cm"># Dashboard:  http://localhost:6969/dashboard</span>{'\n'}
        <span className="cm"># REST API:   http://localhost:6969/</span>{'\n'}
        server.<span className="fn">stop</span>()
      </>
    ),
  },
}

export default function CodeShowcase() {
  const [tab, setTab] = useState('quick')
  const [copyLabel, setCopyLabel] = useState('copy')
  const ref = useRef(null)
  const isReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches

  const onCopy = useCallback(() => {
    navigator.clipboard.writeText(code[tab].plain)
      .then(() => {
        setCopyLabel('copied')
        setTimeout(() => setCopyLabel('copy'), 1200)
      })
      .catch(() => {
        setCopyLabel('failed')
        setTimeout(() => setCopyLabel('copy'), 1200)
      })
  }, [tab])

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
    <section id="code" ref={ref} style={{ opacity: isReduced ? 1 : 0 }}>
      <div className="section-center" style={{ marginBottom: 48 }}>
        <div className="section-label">Developer experience</div>
        <h2 className="section-heading">5 lines to first search.</h2>
        <p className="section-sub center">Simple API. Full control when you need it.</p>
      </div>

      <div className="code-wrap">
        <div className="code-tabs">
          {tabs.map(t => (
            <button
              key={t.id}
              className={`code-tab ${tab === t.id ? 'active' : ''}`}
              onClick={() => { setTab(t.id); setCopyLabel('copy') }}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div className="code-body">
          <button className="code-copy" onClick={onCopy}>{copyLabel}</button>
          <AnimatePresence mode="wait">
            <motion.pre
              key={tab}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
            >
              {code[tab].jsx}
            </motion.pre>
          </AnimatePresence>
        </div>
      </div>
    </section>
  )
}
