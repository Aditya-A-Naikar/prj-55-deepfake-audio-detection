/**
 * Audio Authenticity Lab — Public Landing Page Controller
 * Lightweight, zero-dependency controller for the public landing experience.
 * Manages model status polling, diagnostics drawer accessibility,
 * interactive product preview switching, mobile navigation, and scroll reveals.
 * Strictly decoupled from audio upload, file decoding, and live inference.
 */

(function () {
  'use strict';

  // API Endpoints
  const API = {
    status: '/api/status',
  };

  // DOM Elements
  const el = {
    statusPill: document.getElementById('systemStatusPill'),
    statusText: document.getElementById('systemStatusText'),
    statusDot: document.querySelector('.status-dot'),
    heroParams: document.getElementById('heroParams'),
    heroLatency: document.getElementById('heroLatency'),

    // Mobile Navigation
    mobileMenuBtn: document.getElementById('mobileMenuBtn'),
    mobileNavDrawer: document.getElementById('mobileNavDrawer'),
    mobileNavBackdrop: document.getElementById('mobileNavBackdrop'),
    closeMobileNavBtn: document.getElementById('closeMobileNavBtn'),

    // System Diagnostics Drawer
    diagDrawer: document.getElementById('diagDrawer'),
    diagBackdrop: document.getElementById('diagModalBackdrop'),
    closeDiagBtn: document.getElementById('closeDiagBtn'),
    linkOpenDiag: document.getElementById('linkOpenDiag'),
    diagModelBadge: document.getElementById('diagModelBadge'),
    diagParamCount: document.getElementById('diagParamCount'),
    diagDeviceVal: document.getElementById('diagDeviceVal'),
    diagSha256: document.getElementById('diagSha256'),
    diagFfmpegVal: document.getElementById('diagFfmpegVal'),
    diagOpusVal: document.getElementById('diagOpusVal'),
    diagHostDisk: document.getElementById('diagHostDisk'),
    diagDatasetStatus: document.getElementById('diagDatasetStatus'),

    // Interactive Preview Controls
    previewTabs: document.querySelectorAll('.preview-tab-btn'),
    previewVerdictPill: document.getElementById('previewVerdictPill'),
    previewVerdictText: document.getElementById('previewVerdictText'),
    previewVerdictSub: document.getElementById('previewVerdictSub'),
    previewLogitVal: document.getElementById('previewLogitVal'),
    previewLatencyVal: document.getElementById('previewLatencyVal'),
    previewWaveformCanvas: document.getElementById('previewWaveformCanvas'),
    previewMeterNeedle: document.getElementById('previewMeterNeedle'),
    previewSourceMeta: document.getElementById('previewSourceMeta'),
    previewAcousticNote: document.getElementById('previewAcousticNote'),
  };

  // 1. SYSTEM STATUS & TELEMETRY INITIALIZATION
  async function fetchSystemStatus() {
    try {
      const resp = await fetch(API.status);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      updateStatusDisplay(data);
    } catch (err) {
      if (el.statusText) el.statusText.textContent = 'STANDBY';
      if (el.statusDot) {
        el.statusDot.classList.remove('pulse', 'online');
        el.statusDot.classList.add('warning');
      }
    }
  }

  function updateStatusDisplay(data) {
    if (el.statusText) {
      el.statusText.textContent = data.status === 'online' ? 'ONLINE (MPS / CPU)' : 'STANDBY';
    }
    if (el.statusDot) {
      el.statusDot.classList.remove('warning');
      el.statusDot.classList.add('online');
    }
    if (el.heroParams && data.parameter_count) {
      el.heroParams.textContent = data.parameter_count.toLocaleString();
    }
    if (el.heroLatency) {
      el.heroLatency.textContent = data.device === 'mps' ? '~37 ms (MPS model)' : '~201 ms (CPU model)';
    }

    // Populate Diagnostics Drawer if present
    if (el.diagParamCount && data.parameter_count) {
      el.diagParamCount.textContent = `${data.parameter_count.toLocaleString()} parameters`;
    }
    if (el.diagDeviceVal && data.device) {
      el.diagDeviceVal.textContent = data.device === 'mps' ? 'Apple Silicon MPS (Metal)' : 'Host CPU';
    }
    if (el.diagSha256 && data.checkpoint_sha256) {
      el.diagSha256.textContent = data.checkpoint_sha256;
    }
    if (el.diagFfmpegVal && data.codecs) {
      el.diagFfmpegVal.textContent = data.codecs.ffmpeg_available ? 'Installed' : 'System fallback';
    }
    if (el.diagOpusVal && data.codecs) {
      el.diagOpusVal.textContent = data.codecs.opus_available ? 'Available' : 'Synthetic fallback';
    }
  }

  // 2. DIAGNOSTICS DRAWER MODAL
  function openDiagnostics() {
    if (el.diagDrawer) {
      el.diagDrawer.classList.remove('hidden');
      el.diagDrawer.setAttribute('aria-hidden', 'false');
    }
    if (el.diagBackdrop) {
      el.diagBackdrop.classList.remove('hidden');
    }
    document.body.classList.add('drawer-open');
    if (el.closeDiagBtn) el.closeDiagBtn.focus();
  }

  function closeDiagnostics() {
    if (el.diagDrawer) {
      el.diagDrawer.classList.add('hidden');
      el.diagDrawer.setAttribute('aria-hidden', 'true');
    }
    if (el.diagBackdrop) {
      el.diagBackdrop.classList.add('hidden');
    }
    document.body.classList.remove('drawer-open');
    if (el.statusPill) el.statusPill.focus();
  }

  if (el.statusPill) {
    el.statusPill.addEventListener('click', openDiagnostics);
  }
  if (el.closeDiagBtn) {
    el.closeDiagBtn.addEventListener('click', closeDiagnostics);
  }
  if (el.diagBackdrop) {
    el.diagBackdrop.addEventListener('click', closeDiagnostics);
  }
  if (el.linkOpenDiag) {
    el.linkOpenDiag.addEventListener('click', openDiagnostics);
  }

  // 3. MOBILE NAVIGATION DRAWER
  function openMobileNav() {
    if (el.mobileNavDrawer) el.mobileNavDrawer.classList.remove('hidden');
    if (el.mobileNavBackdrop) el.mobileNavBackdrop.classList.remove('hidden');
    document.body.classList.add('mobile-nav-open');
  }

  function closeMobileNav() {
    if (el.mobileNavDrawer) el.mobileNavDrawer.classList.add('hidden');
    if (el.mobileNavBackdrop) el.mobileNavBackdrop.classList.add('hidden');
    document.body.classList.remove('mobile-nav-open');
  }

  if (el.mobileMenuBtn) {
    el.mobileMenuBtn.addEventListener('click', openMobileNav);
  }
  if (el.closeMobileNavBtn) {
    el.closeMobileNavBtn.addEventListener('click', closeMobileNav);
  }
  if (el.mobileNavBackdrop) {
    el.mobileNavBackdrop.addEventListener('click', closeMobileNav);
  }

  // Global escape key handler for accessible modal dismissal
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeDiagnostics();
      closeMobileNav();
    }
  });

  // 4. INTERACTIVE PRODUCT PREVIEW (ILLUSTRATIVE DEMO SWITCHER)
  const PREVIEW_PROFILES = {
    bonafide: {
      type: 'bonafide',
      verdictText: 'BONAFIDE',
      verdictSub: 'Authentic Human Speech',
      logit: '+2.148',
      latency: '40.6 ms',
      sourceMeta: '16,000 Hz Mono • VCTK Speaker p225 • Authentic glottal resonance',
      meterPercent: 78,
      acousticNote: 'Dense, continuous glottal harmonic series across 0–4 kHz with preserved high-frequency air dispersion. SincNet graph-attention confirms natural vocal tract resonance.',
      waveSeed: 1.0,
    },
    spoof_tts: {
      type: 'spoof',
      verdictText: 'SPOOF',
      verdictSub: 'Neural TTS Speech Attack',
      logit: '-3.412',
      latency: '41.2 ms',
      sourceMeta: '16,000 Hz Mono • Tacotron2 / HiFi-GAN • Unseen synthesis family',
      meterPercent: 12,
      acousticNote: 'Phase-incoherent frame transitions detected between phoneme boundaries. Truncated high-frequency phase spectrum characteristic of neural vocoder upsampling.',
      waveSeed: 2.5,
    },
    spoof_vocoder: {
      type: 'spoof',
      verdictText: 'SPOOF',
      verdictSub: 'Vocoder Synthesis Attack',
      logit: '-1.895',
      latency: '40.8 ms',
      sourceMeta: '16,000 Hz Mono • STRAIGHT Vocoder • SS_3 synthetic attack',
      meterPercent: 26,
      acousticNote: 'Harmonic overtone smearing and artificial pitch period quantization detected by temporal attention nodes. Logit score sits deeply within spoof decision space.',
      waveSeed: 4.0,
    },
  };

  function renderPreviewWaveform(waveSeed, isSpoof) {
    const canvas = el.previewWaveformCanvas;
    if (!canvas) return;
    const parent = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    // Calculate precise inner width of parent element
    const rect = parent ? parent.getBoundingClientRect() : null;
    const paddingX = 28; // 14px padding on left and right
    const rawW = rect ? Math.floor(rect.width - paddingX) : ((parent && parent.clientWidth) ? parent.clientWidth - paddingX : 400);
    const w = Math.max(160, rawW);
    const h = 80;

    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = '100%';
    canvas.style.height = `${h}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const mid = h / 2;
    const numBars = 48;
    const gap = 2.5;
    const totalGaps = (numBars - 1) * gap;
    const barWidth = Math.max(2, (w - totalGaps) / numBars);

    for (let i = 0; i < numBars; i++) {
      const x = i * (barWidth + gap);
      const angle = (i / numBars) * Math.PI * 4 * waveSeed;
      const env = Math.sin((i / numBars) * Math.PI);
      let amp = (Math.abs(Math.sin(angle)) * 0.7 + 0.25) * env * (h * 0.42);
      if (amp < 3) amp = 3;

      ctx.beginPath();
      ctx.roundRect(x, mid - amp, barWidth, amp * 2, 2);
      if (isSpoof) {
        ctx.fillStyle = i % 2 === 0 ? 'rgba(244, 63, 94, 0.85)' : 'rgba(244, 63, 94, 0.4)';
      } else {
        ctx.fillStyle = i % 2 === 0 ? 'rgba(113, 153, 255, 0.85)' : 'rgba(45, 212, 191, 0.55)';
      }
      ctx.fill();
    }
  }

  function setPreviewProfile(profileKey) {
    const data = PREVIEW_PROFILES[profileKey];
    if (!data) return;

    // Update Tab UI
    el.previewTabs.forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.preview === profileKey);
      btn.setAttribute('aria-selected', btn.dataset.preview === profileKey ? 'true' : 'false');
    });

    // Update Verdict Display
    if (el.previewVerdictPill) {
      el.previewVerdictPill.className = `verdict-pill-large ${data.type}`;
    }
    if (el.previewVerdictText) el.previewVerdictText.textContent = data.verdictText;
    if (el.previewVerdictSub) el.previewVerdictSub.textContent = data.verdictSub;
    if (el.previewLogitVal) el.previewLogitVal.textContent = data.logit;
    if (el.previewLatencyVal) el.previewLatencyVal.textContent = data.latency;
    if (el.previewSourceMeta) el.previewSourceMeta.textContent = data.sourceMeta;
    if (el.previewAcousticNote) el.previewAcousticNote.textContent = data.acousticNote;

    // Update Score Meter Needle
    if (el.previewMeterNeedle) {
      el.previewMeterNeedle.style.left = `${data.meterPercent}%`;
    }

    // Render Preview Waveform
    renderPreviewWaveform(data.waveSeed, data.type === 'spoof');
  }

  if (el.previewTabs.length > 0) {
    el.previewTabs.forEach((tab) => {
      tab.addEventListener('click', () => {
        setPreviewProfile(tab.dataset.preview);
      });
    });
    // Initialize default profile
    setPreviewProfile('bonafide');
    window.addEventListener('resize', () => {
      const activeTab = document.querySelector('.preview-tab-btn.active');
      const key = activeTab ? activeTab.dataset.preview : 'bonafide';
      const data = PREVIEW_PROFILES[key];
      if (data) renderPreviewWaveform(data.waveSeed, data.type === 'spoof');
    });
  }

  // 5. NATURAL SCROLL REVEALS (INTERSECTION OBSERVER)
  if ('IntersectionObserver' in window) {
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const revealEls = document.querySelectorAll('.reveal-on-scroll');
    
    if (prefersReducedMotion) {
      revealEls.forEach((elem) => elem.classList.add('is-visible'));
    } else {
      const observer = new IntersectionObserver((entries, obs) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
            obs.unobserve(entry.target);
          }
        });
      }, { threshold: 0.12 });
      revealEls.forEach((elem) => observer.observe(elem));
    }
  } else {
    document.querySelectorAll('.reveal-on-scroll').forEach((elem) => elem.classList.add('is-visible'));
  }

  // Fetch initial telemetry
  fetchSystemStatus();
})();
