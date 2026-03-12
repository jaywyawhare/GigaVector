import { useEffect, useRef } from 'react'
import * as THREE from 'three'

/**
 * Minimal particle field — subtle floating dots, no connection lines,
 * no search rings. Just atmosphere.
 */
export function useVectorScene() {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(window.innerWidth, window.innerHeight)

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000)
    camera.position.set(0, 0, 45)

    const COUNT = 600
    const positions = new Float32Array(COUNT * 3)
    const sizes = new Float32Array(COUNT)
    const phases = []

    for (let i = 0; i < COUNT; i++) {
      const r = 15 + Math.random() * 35
      const theta = Math.random() * Math.PI * 2
      const phi = Math.acos(2 * Math.random() - 1)
      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta)
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta)
      positions[i * 3 + 2] = r * Math.cos(phi)
      sizes[i] = 0.8 + Math.random() * 1.5
      phases.push(Math.random() * Math.PI * 2)
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geo.setAttribute('size', new THREE.BufferAttribute(sizes, 1))

    const mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms: { uColor: { value: new THREE.Color(0.5, 0.5, 0.55) } },
      vertexShader: `
        attribute float size;
        varying float vDist;
        void main() {
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          vDist = -mv.z;
          gl_PointSize = size * (140.0 / vDist);
          gl_Position = projectionMatrix * mv;
        }
      `,
      fragmentShader: `
        uniform vec3 uColor;
        varying float vDist;
        void main() {
          float d = length(gl_PointCoord - 0.5);
          if (d > 0.5) discard;
          float a = smoothstep(0.5, 0.15, d);
          float fog = smoothstep(80.0, 20.0, vDist);
          gl_FragColor = vec4(uColor, a * fog * 0.45);
        }
      `,
    })

    const points = new THREE.Points(geo, mat)
    scene.add(points)

    let mouseX = 0, mouseY = 0
    const onMove = (e) => {
      mouseX = (e.clientX / window.innerWidth - 0.5) * 2
      mouseY = (e.clientY / window.innerHeight - 0.5) * 2
    }
    window.addEventListener('mousemove', onMove)

    const clock = new THREE.Clock()
    let raf

    function loop() {
      raf = requestAnimationFrame(loop)
      const t = clock.getElapsedTime()

      points.rotation.y = t * 0.015 + mouseX * 0.08
      points.rotation.x = t * 0.008 + mouseY * 0.05

      const pos = geo.attributes.position.array
      for (let i = 0; i < COUNT; i++) {
        const p = phases[i]
        pos[i * 3] += Math.sin(t * 0.3 + p) * 0.008
        pos[i * 3 + 1] += Math.cos(t * 0.2 + p) * 0.008
      }
      geo.attributes.position.needsUpdate = true

      camera.position.x += (mouseX * 2 - camera.position.x) * 0.01
      camera.position.y += (-mouseY * 1.5 - camera.position.y) * 0.01
      camera.lookAt(0, 0, 0)

      renderer.render(scene, camera)
    }
    loop()

    const onResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight
      camera.updateProjectionMatrix()
      renderer.setSize(window.innerWidth, window.innerHeight)
    }
    window.addEventListener('resize', onResize)

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('resize', onResize)
      renderer.dispose()
      geo.dispose()
      mat.dispose()
    }
  }, [])

  return canvasRef
}
