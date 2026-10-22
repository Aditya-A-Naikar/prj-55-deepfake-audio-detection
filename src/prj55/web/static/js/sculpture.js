/**
 * PRJ_55 Audio Forensics Platform — 3D Harmonic Signal Sculpture
 * Interactive perspective-projected sound-wave particle ribbon and lattice.
 */

(function () {
  'use strict';

  const canvas = document.getElementById('signalSculptureCanvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  // Check reduced motion preference
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  let width = (canvas.width = canvas.parentElement.clientWidth || 600);
  let height = (canvas.height = canvas.parentElement.clientHeight || 480);

  // Resize handler
  function onResize() {
    width = canvas.width = canvas.parentElement.clientWidth || 600;
    height = canvas.height = canvas.parentElement.clientHeight || 480;
  }
  window.addEventListener('resize', onResize);

  // State
  let rotX = 0.45;
  let rotY = -0.55;
  let targetRotX = 0.45;
  let targetRotY = -0.55;
  let isDragging = false;
  let lastMouseX = 0;
  let lastMouseY = 0;
  let time = 0;
  let pulseIntensity = 0;
  let mode = 'ribbon'; // 'ribbon' or 'lattice'

  // Mouse interaction
  canvas.addEventListener('mousedown', (e) => {
    isDragging = true;
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
  });

  window.addEventListener('mouseup', () => {
    isDragging = false;
  });

  window.addEventListener('mousemove', (e) => {
    if (isDragging) {
      const dx = e.clientX - lastMouseX;
      const dy = e.clientY - lastMouseY;
      targetRotY += dx * 0.008;
      targetRotX += dy * 0.008;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
    }
  });

  // Touch support
  canvas.addEventListener('touchstart', (e) => {
    if (e.touches.length === 1) {
      isDragging = true;
      lastMouseX = e.touches[0].clientX;
      lastMouseY = e.touches[0].clientY;
    }
  }, { passive: true });

  window.addEventListener('touchmove', (e) => {
    if (isDragging && e.touches.length === 1) {
      const dx = e.touches[0].clientX - lastMouseX;
      const dy = e.touches[0].clientY - lastMouseY;
      targetRotY += dx * 0.008;
      targetRotX += dy * 0.008;
      lastMouseX = e.touches[0].clientX;
      lastMouseY = e.touches[0].clientY;
    }
  }, { passive: true });

  window.addEventListener('touchend', () => {
    isDragging = false;
  });

  // Pulse & Mode Buttons
  const pulseBtn = document.getElementById('sculpturePulseBtn');
  if (pulseBtn) {
    pulseBtn.addEventListener('click', () => {
      pulseIntensity = 2.4;
    });
  }

  const modeBtn = document.getElementById('sculptureModeBtn');
  if (modeBtn) {
    modeBtn.addEventListener('click', () => {
      mode = mode === 'ribbon' ? 'lattice' : 'ribbon';
      modeBtn.textContent = mode === 'ribbon' ? '〰 Harmonic Mode' : '∷ Lattice Mode';
    });
  }

  // Exposed method to trigger pulse during real inference
  window.triggerSculpturePulse = function (intensity = 2.0) {
    pulseIntensity = intensity;
  };

  // 3D Projection math
  const FOV = 420;
  const ROWS = 28;
  const COLS = 36;
  const SPACING = 14;

  function project(x, y, z) {
    // Rotate around Y
    const cosY = Math.cos(rotY);
    const sinY = Math.sin(rotY);
    const x1 = x * cosY + z * sinY;
    const z1 = -x * sinY + z * cosY;

    // Rotate around X
    const cosX = Math.cos(rotX);
    const sinX = Math.sin(rotX);
    const y2 = y * cosX - z1 * sinX;
    const z2 = y * sinX + z1 * cosX + 380; // camera distance

    if (z2 <= 20) return null;

    const scale = FOV / z2;
    return {
      x: width / 2 + x1 * scale,
      y: height / 2 + y2 * scale,
      scale: scale,
      z: z2,
    };
  }

  // Animation loop
  function render() {
    // Dampen rotation towards target
    rotX += (targetRotX - rotX) * 0.06;
    rotY += (targetRotY - rotY) * 0.06;

    if (!prefersReducedMotion) {
      time += 0.024;
      pulseIntensity = Math.max(0, pulseIntensity - 0.025);
    }

    ctx.clearRect(0, 0, width, height);

    // Compute 3D points
    const points = [];
    for (let r = 0; r < ROWS; r++) {
      const row = [];
      const z = (r - ROWS / 2) * SPACING;
      for (let c = 0; c < COLS; c++) {
        const x = (c - COLS / 2) * SPACING;

        // Wave elevation formula: fundamental + harmonic + pulse ripple
        const dist = Math.sqrt(x * x + z * z);
        const fundamental = Math.sin(x * 0.04 + time * 1.5) * 22;
        const harmonic = Math.cos(z * 0.05 - time * 1.2) * 16;
        const ripple = pulseIntensity > 0
          ? Math.sin(dist * 0.08 - time * 6.0) * (pulseIntensity * 14)
          : 0;

        const y = fundamental + harmonic + ripple;
        const p = project(x, y, z);
        row.push(p);
      }
      points.push(row);
    }

    // Draw connected ribbons or lattice lines
    ctx.lineWidth = 1.2;

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

            // Color gradient based on pulse or depth
            if (pulseIntensity > 0.8) {
              ctx.strokeStyle = `rgba(255, 54, 87, ${Math.min(0.85, p.scale * 0.9)})`;
            } else if (r % 2 === 0) {
              ctx.strokeStyle = `rgba(67, 139, 255, ${Math.min(0.75, p.scale * 0.7)})`;
            } else {
              ctx.strokeStyle = `rgba(201, 185, 255, ${Math.min(0.55, p.scale * 0.5)})`;
            }
            ctx.stroke();
          }
        }

        // In lattice mode, also draw vertical lines to next row
        if (mode === 'lattice' && r < ROWS - 1) {
          const downP = points[r + 1][c];
          if (downP) {
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(downP.x, downP.y);
            ctx.strokeStyle = `rgba(131, 138, 152, ${Math.min(0.35, p.scale * 0.4)})`;
            ctx.stroke();
          }
        }

        // Render crest particle glow
        if (c % 3 === 0 && r % 3 === 0 && p.scale > 0.8) {
          ctx.beginPath();
          ctx.arc(p.x, p.y, Math.max(1, p.scale * 2.2), 0, Math.PI * 2);
          ctx.fillStyle = pulseIntensity > 0.8 ? '#FF3657' : '#438BFF';
          ctx.fill();
        }
      }
    }

    requestAnimationFrame(render);
  }

  requestAnimationFrame(render);
})();
