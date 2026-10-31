/**
 * Audio Authenticity Lab — 3D Harmonic Signal Sculpture
 * Ambient spectro-temporal sound-wave visualization with perspective projection,
 * subtle background undulation, and restrained hover illumination.
 */

(function () {
  'use strict';

  const canvas = document.getElementById('signalSculptureCanvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  if (!ctx) {
    // Graceful fallback for environments without 2D canvas context
    const container = canvas.parentElement;
    if (container) {
      container.classList.add('canvas-unsupported');
    }
    return;
  }

  // Check reduced motion preference
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  let width = (canvas.width = canvas.parentElement.clientWidth || window.innerWidth || 1200);
  let height = (canvas.height = canvas.parentElement.clientHeight || 650);

  // Resize handler
  function onResize() {
    if (!canvas.parentElement) return;
    width = canvas.width = canvas.parentElement.clientWidth || window.innerWidth || 1200;
    height = canvas.height = canvas.parentElement.clientHeight || 650;
    if (prefersReducedMotion) {
      renderSingleFrame();
    }
  }
  window.addEventListener('resize', onResize);

  // Fixed, elegant viewing perspective
  const rotX = 0.36;
  const rotY = -0.42;

  // Animation & lighting state
  let time = 0;
  let pulseIntensity = 0;
  let hoverGlow = 0;
  let targetHoverGlow = 0;

  // Restrained hover lighting response across the hero section
  const heroSection = document.getElementById('overview') || canvas.parentElement || canvas;
  heroSection.addEventListener('mouseenter', () => {
    targetHoverGlow = 1.0;
  });
  heroSection.addEventListener('mouseleave', () => {
    targetHoverGlow = 0.0;
  });

  // Exposed trigger for analysis pulse
  window.triggerSculpturePulse = function (intensity = 2.0) {
    if (!prefersReducedMotion) {
      pulseIntensity = intensity;
    }
  };

  // 3D Projection math scaled for full-bleed width
  const ROWS = 28;
  const COLS = 46;

  function project(x, y, z) {
    const FOV = Math.max(500, width * 0.48);

    // Fixed perspective rotation around Y
    const cosY = Math.cos(rotY);
    const sinY = Math.sin(rotY);
    const x1 = x * cosY + z * sinY;
    const z1 = -x * sinY + z * cosY;

    // Fixed perspective rotation around X
    const cosX = Math.cos(rotX);
    const sinX = Math.sin(rotX);
    const y2 = y * cosX - z1 * sinX;
    const z2 = y * sinX + z1 * cosX + 410; // camera distance

    if (z2 <= 20) return null;

    const scale = FOV / z2;
    return {
      x: width / 2 + x1 * scale,
      y: height * 0.58 + y2 * scale,
      scale: scale,
      z: z2,
    };
  }

  // Viewport visibility optimization via IntersectionObserver
  let isVisible = true;
  let animationFrameId = null;

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        isVisible = entry.isIntersecting;
        if (isVisible && !animationFrameId && !prefersReducedMotion) {
          animationFrameId = requestAnimationFrame(render);
        }
      });
    }, { threshold: 0.05 });
    observer.observe(canvas);
  }

  // Draw one frame of the 3D harmonic wavefield
  function drawWavefield() {
    ctx.clearRect(0, 0, width, height);

    const SPACING = Math.max(16, Math.min(26, width / 44));

    // Compute 3D points
    const points = [];
    for (let r = 0; r < ROWS; r++) {
      const row = [];
      const z = (r - ROWS / 2) * SPACING;
      for (let c = 0; c < COLS; c++) {
        const x = (c - COLS / 2) * SPACING;

        // Subtle organic acoustic wave formula: fundamental + harmonic + pulse ripple
        const dist = Math.sqrt(x * x + z * z);
        const fundamental = Math.sin(x * 0.038 + time * 1.1) * 20;
        const harmonic = Math.cos(z * 0.046 - time * 0.9) * 14;
        const ripple = pulseIntensity > 0
          ? Math.sin(dist * 0.08 - time * 5.0) * (pulseIntensity * 12)
          : 0;

        const y = fundamental + harmonic + ripple;
        const p = project(x, y, z);
        row.push(p);
      }
      points.push(row);
    }

    // Dynamic stroke alpha with subtle hover illumination boost
    const glowBoost = hoverGlow * 0.25;

    // Draw connected ribbons
    ctx.lineWidth = 1.15;

    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const p = points[r][c];
        if (!p) continue;

        // Draw horizontal line to next col
        if (c < COLS - 1) {
          const nextP = points[r][c + 1];
          if (nextP) {
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(nextP.x, nextP.y);

            // Subtle electric-blue and soft lavender tones
            if (pulseIntensity > 0.8) {
              const alpha = Math.min(0.85, (p.scale * 0.85) + glowBoost);
              ctx.strokeStyle = `rgba(255, 54, 87, ${alpha})`;
            } else if (r % 2 === 0) {
              const alpha = Math.min(0.80, (p.scale * 0.65) + glowBoost);
              ctx.strokeStyle = `rgba(67, 139, 255, ${alpha})`;
            } else {
              const alpha = Math.min(0.60, (p.scale * 0.45) + glowBoost);
              ctx.strokeStyle = `rgba(186, 170, 240, ${alpha})`;
            }
            ctx.stroke();
          }
        }

        // Render crest particle glow on prominent grid vertices
        if (c % 3 === 0 && r % 3 === 0 && p.scale > 0.75) {
          ctx.beginPath();
          ctx.arc(p.x, p.y, Math.max(1, p.scale * 1.8), 0, Math.PI * 2);
          ctx.fillStyle = pulseIntensity > 0.8 ? '#FF3657' : (hoverGlow > 0.5 ? '#60A5FA' : '#3B82F6');
          ctx.fill();
        }
      }
    }
  }

  function renderSingleFrame() {
    time = 0.5;
    hoverGlow = 0;
    drawWavefield();
  }

  // Animation loop (ambient continuous flow)
  function render() {
    if (!isVisible || document.hidden) {
      animationFrameId = null;
      return;
    }

    // Smooth hover glow easing
    hoverGlow += (targetHoverGlow - hoverGlow) * 0.08;

    // Ambient time progression
    time += 0.014;
    pulseIntensity = Math.max(0, pulseIntensity - 0.022);

    drawWavefield();

    animationFrameId = requestAnimationFrame(render);
  }

  // Start animation loop or render static frame for reduced motion
  if (prefersReducedMotion) {
    renderSingleFrame();
  } else {
    animationFrameId = requestAnimationFrame(render);
  }
})();
