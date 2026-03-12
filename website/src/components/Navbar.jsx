import { useEffect, useRef } from 'react'
import { Link, useLocation } from 'react-router-dom'
import gsap from 'gsap'

export default function Navbar() {
  const ref = useRef(null)
  const { pathname } = useLocation()
  const isHome = pathname === '/'

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const ctx = gsap.context(() => {
      gsap.fromTo(ref.current,
        { y: -20, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.6, ease: 'power2.out', delay: 0.1 }
      )
    })
    return () => ctx.revert()
  }, [])

  return (
    <nav ref={ref} className="nav" style={{ opacity: 0 }}>
      <Link className="nav-logo" to="/">
        <img src="https://raw.githubusercontent.com/jaywyawhare/GigaVector/master/docs/gigavector-logo.png" alt="GigaVector" className="nav-logo-img" />
        GigaVector
        <span className="nav-ver">0.8.2</span>
      </Link>

      <ul className="nav-links">
        {isHome ? (
          <>
            <li><a href="#features">Features</a></li>
            <li><a href="#code">Code</a></li>
            <li><a href="#comparison">Comparison</a></li>
            <li><a href="#performance">Performance</a></li>
          </>
        ) : null}
        <li><Link to="/docs/index">Docs</Link></li>
      </ul>

      <div className="nav-right">
        <a className="btn-sm" href="https://github.com/jaywyawhare/GigaVector" target="_blank" rel="noopener">GitHub</a>
        <a className="btn-sm filled" href="https://pypi.org/project/gigavector/" target="_blank" rel="noopener">Install</a>
      </div>
    </nav>
  )
}
