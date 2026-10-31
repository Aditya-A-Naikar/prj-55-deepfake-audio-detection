/**
 * Audio Authenticity Lab — Robustness & Perturbation Laboratory Controller
 * Manages channel degradation experiments, multi-condition sweeps, and A/B audio playback.
 */

(function () {
  'use strict';

  const state = {
    selectedSampleId: 'speech_human_male',
    uploadedFile: null,
    condition: 'noise_snr',
    snrDb: 10.0,
    mitigationEnabled: true,
    seed: 42,
    audioCtx: null,
    cleanAudioBuffer: null,
    procAudioBuffer: null,
    activeAudioSourceNode: null,
    isPlaying: false,
    currentPlaybackType: 'clean',
    lastResult: null,
    sweepData: null
  };

  const el = {
    statusPill: document.getElementById('systemStatusPill'),
    statusText: document.getElementById('systemStatusText'),
    diagDrawer: document.getElementById('diagDrawer'),
    diagBackdrop: document.getElementById('diagBackdrop'),
    closeDiagBtn: document.getElementById('closeDiagBtn'),
    mobileMenuBtn: document.getElementById('mobileMenuBtn'),
    mobileNavDrawer: document.getElementById('mobileNavDrawer'),
    mobileNavBackdrop: document.getElementById('mobileNavBackdrop'),
    closeMobileNavBtn: document.getElementById('closeMobileNavBtn'),

    // Source selection
    presetList: document.getElementById('robPresetList'),
    dropZone: document.getElementById('robDropZone'),
    fileInput: document.getElementById('robFileInput'),
    activeLabel: document.getElementById('robActiveAudioLabel'),
    audioTime: document.getElementById('robAudioTime'),
    playSourceBtn: document.getElementById('robPlaySourceBtn'),
    stopSourceBtn: document.getElementById('robStopSourceBtn'),

    // Degradation & mitigation controls
    conditionRadios: document.querySelectorAll('input[name="robCondition"]'),
    snrSlider: document.getElementById('robSnrSlider'),
    snrDisplay: document.getElementById('robSnrDisplay'),
    snrGroup: document.getElementById('robSnrGroup'),
    seedInput: document.getElementById('robSeedInput'),
    mitigationToggle: document.getElementById('robMitigationToggle'),
    runSingleBtn: document.getElementById('runRobExpBtn'),
    runSweepBtn: document.getElementById('runSweepMatrixBtn'),

    // Comparison cards
    expStatus: document.getElementById('robExpStatus'),
    cleanScore: document.getElementById('robCleanScore'),
    cleanPill: document.getElementById('robCleanPill'),
    cleanProb: document.getElementById('robCleanProb'),
    cleanWaveCanvas: document.getElementById('robCleanWaveCanvas'),
    procScore: document.getElementById('robProcScore'),
    procPill: document.getElementById('robProcPill'),
    procProb: document.getElementById('robProcProb'),
    procCondition: document.getElementById('robProcCondition'),
    procTag: document.getElementById('robProcTag'),
    procWaveCanvas: document.getElementById('robProcWaveCanvas'),

    // A/B playback
    playCleanBtn: document.getElementById('playCleanAudioBtn'),
    playProcBtn: document.getElementById('playProcAudioBtn'),

    // Delta synthesis
    deltaValue: document.getElementById('robDeltaValue'),
    boundaryBadge: document.getElementById('robBoundaryBadge'),
    deltaExplanation: document.getElementById('robDeltaExplanation'),

    // Sweep table & canvas
    sweepCountTag: document.getElementById('sweepCountTag'),
    sweepTableBody: document.getElementById('robSweepTableBody'),
    responseCanvas: document.getElementById('robResponseCanvas')
  };

  async function getAudioContext() {
    if (!state.audioCtx) {
      const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
      state.audioCtx = new AudioCtxClass();
    }
    if (state.audioCtx.state === 'suspended') {
      await state.audioCtx.resume();
    }
    return state.audioCtx;
  }

  function init() {
    registerNavigationHandlers();
    registerControlHandlers();
    checkSystemStatus();
    resetExperimentCardsToStandby();
    drawChartPlaceholder();
    loadSourceAudioPreset(state.selectedSampleId);

    window.addEventListener('resize', () => {
      if (state.sweepData) renderResponseCurveCanvas(state.sweepData);
      else drawChartPlaceholder();
    });

    // Check URL parameters for preset selection e.g. /robustness?sample=speech_deepfake_tts
    const urlParams = new URLSearchParams(window.location.search);
    const sampleParam = urlParams.get('sample');
    if (sampleParam) {
      loadSourceAudioPreset(sampleParam);
    }
  }

  function registerNavigationHandlers() {
    // Mobile Drawer
    if (el.mobileMenuBtn && el.mobileNavDrawer) {
      el.mobileMenuBtn.addEventListener('click', () => {
        el.mobileNavDrawer.classList.toggle('hidden');
        el.mobileNavBackdrop.classList.toggle('hidden');
      });
      el.closeMobileNavBtn.addEventListener('click', () => {
        el.mobileNavDrawer.classList.add('hidden');
        el.mobileNavBackdrop.classList.add('hidden');
      });
      el.mobileNavBackdrop.addEventListener('click', () => {
        el.mobileNavDrawer.classList.add('hidden');
        el.mobileNavBackdrop.classList.add('hidden');
      });
    }

    // Diagnostics Drawer
    if (el.statusPill && el.diagDrawer) {
      el.statusPill.addEventListener('click', () => {
        el.diagDrawer.classList.toggle('hidden');
        el.diagBackdrop.classList.toggle('hidden');
      });
      el.closeDiagBtn.addEventListener('click', () => {
        el.diagDrawer.classList.add('hidden');
        el.diagBackdrop.classList.add('hidden');
      });
      el.diagBackdrop.addEventListener('click', () => {
        el.diagDrawer.classList.add('hidden');
        el.diagBackdrop.classList.add('hidden');
      });
    }

    // Command palette trigger in header
    const cmdBtn = document.getElementById('headerCmdPaletteBtn');
    if (cmdBtn && window.AALPalette) {
      cmdBtn.addEventListener('click', () => window.AALPalette.open());
    }
  }

  function registerControlHandlers() {
    // Preset buttons
    if (el.presetList) {
      el.presetList.addEventListener('click', (e) => {
        const btn = e.target.closest('.preset-btn');
        if (!btn) return;
        el.presetList.querySelectorAll('.preset-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const sampleId = btn.getAttribute('data-sample');
        state.selectedSampleId = sampleId;
        state.uploadedFile = null;
        loadSourceAudioPreset(sampleId);
      });
    }

    // Dropzone
    if (el.dropZone && el.fileInput) {
      el.dropZone.addEventListener('click', () => el.fileInput.click());
      el.dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        el.dropZone.classList.add('dragover');
      });
      el.dropZone.addEventListener('dragleave', () => el.dropZone.classList.remove('dragover'));
      el.dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        el.dropZone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
          handleUploadedFile(e.dataTransfer.files[0]);
        }
      });
      el.fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
          handleUploadedFile(e.target.files[0]);
        }
      });
    }

    // Condition Radios
    el.conditionRadios.forEach((r) => {
      r.addEventListener('change', (e) => {
        state.condition = e.target.value;
        document.querySelectorAll('.channel-radio-card').forEach((card) => {
          card.classList.toggle('active', card.querySelector('input').checked);
        });
        if (state.condition.includes('noise')) {
          el.snrGroup.classList.remove('hidden');
        } else {
          el.snrGroup.classList.add('hidden');
        }
      });
    });

    // SNR Slider
    if (el.snrSlider) {
      el.snrSlider.addEventListener('input', (e) => {
        state.snrDb = parseFloat(e.target.value);
        if (el.snrDisplay) el.snrDisplay.textContent = `${state.snrDb} dB`;
      });
    }

    // Mitigation Toggle
    if (el.mitigationToggle) {
      el.mitigationToggle.addEventListener('change', (e) => {
        state.mitigationEnabled = e.target.checked;
        const details = document.getElementById('robMitigationDetails');
        if (details) details.classList.toggle('opacity-50', !state.mitigationEnabled);
      });
    }

    // Seed Input
    if (el.seedInput) {
      el.seedInput.addEventListener('change', (e) => {
        state.seed = parseInt(e.target.value, 10) || 42;
      });
    }

    // Experiment Buttons
    if (el.runSingleBtn) {
      el.runSingleBtn.addEventListener('click', runSingleExperiment);
    }
    if (el.runSweepBtn) {
      el.runSweepBtn.addEventListener('click', runMultiConditionSweep);
    }

    // Audio Playback
    if (el.playSourceBtn) el.playSourceBtn.addEventListener('click', () => playAudioBuffer(state.cleanAudioBuffer));
    if (el.stopSourceBtn) el.stopSourceBtn.addEventListener('click', stopAudioPlayback);
    if (el.playCleanBtn) el.playCleanBtn.addEventListener('click', () => {
      el.playCleanBtn.classList.add('active');
      if (el.playProcBtn) el.playProcBtn.classList.remove('active');
      playAudioBuffer(state.cleanAudioBuffer);
    });
    if (el.playProcBtn) el.playProcBtn.addEventListener('click', () => {
      el.playProcBtn.classList.add('active');
      if (el.playCleanBtn) el.playCleanBtn.classList.remove('active');
      playAudioBuffer(state.procAudioBuffer);
    });
  }

  async function checkSystemStatus() {
    try {
      const res = await fetch('/api/status');
      if (!res.ok) throw new Error('Status unavailable');
      const data = await res.json();
      if (el.statusText) el.statusText.textContent = 'ONLINE • AASIST-L';
      if (el.statusPill) el.statusPill.classList.add('online');

      // Populate diagnostics
      const diagDev = document.getElementById('diagDevice');
      const diagSha = document.getElementById('diagSha256');
      const diagFfmpeg = document.getElementById('diagFfmpeg');
      if (diagDev) diagDev.textContent = `${data.device.toUpperCase()} (AASIST-L)`;
      if (diagSha) diagSha.textContent = data.checkpoint_sha256 ? `${data.checkpoint_sha256.slice(0, 16)}...` : 'Not loaded';
      if (diagFfmpeg) diagFfmpeg.textContent = data.codecs.opus_available ? 'FFmpeg + libopus active' : 'FFmpeg not detected';
    } catch {
      if (el.statusText) el.statusText.textContent = 'STANDBY';
    }
  }

  async function loadSourceAudioPreset(sampleId) {
    stopAudioPlayback();
    resetExperimentCardsToStandby();
    if (el.activeLabel) el.activeLabel.textContent = `Loading ${sampleId}...`;

    try {
      const resp = await fetch(`/api/samples/${sampleId}/audio`);
      if (!resp.ok) throw new Error(`Could not load audio for ${sampleId}`);
      const arrayBuf = await resp.arrayBuffer();
      const ctx = await getAudioContext();
      state.cleanAudioBuffer = await ctx.decodeAudioData(arrayBuf.slice(0));

      const dur = state.cleanAudioBuffer.duration.toFixed(2);
      if (el.activeLabel) el.activeLabel.textContent = `Active: ${sampleId}.wav (16 kHz mono)`;
      if (el.audioTime) el.audioTime.textContent = `0:00 / ${dur}s`;

      drawWaveform(state.cleanAudioBuffer, el.cleanWaveCanvas, '#60A5FA');
    } catch (err) {
      console.error('Preset loading error:', err);
      if (el.activeLabel) el.activeLabel.textContent = `Error loading ${sampleId}`;
    }
  }

  async function handleUploadedFile(file) {
    stopAudioPlayback();
    resetExperimentCardsToStandby();
    state.uploadedFile = file;
    state.selectedSampleId = null;
    if (el.presetList) {
      el.presetList.querySelectorAll('.preset-btn').forEach((b) => b.classList.remove('active'));
    }
    if (el.activeLabel) el.activeLabel.textContent = `Active File: ${file.name}`;

    try {
      const arrayBuf = await file.arrayBuffer();
      const ctx = await getAudioContext();
      state.cleanAudioBuffer = await ctx.decodeAudioData(arrayBuf.slice(0));
      const dur = state.cleanAudioBuffer.duration.toFixed(2);
      if (el.audioTime) el.audioTime.textContent = `0:00 / ${dur}s`;
      drawWaveform(state.cleanAudioBuffer, el.cleanWaveCanvas, '#60A5FA');
    } catch (err) {
      console.error('File decode error:', err);
      alert(`Could not decode audio file: ${err.message}`);
    }
  }

  async function runSingleExperiment() {
    if (!state.selectedSampleId && !state.uploadedFile) {
      alert('Please select or upload an audio file first.');
      return;
    }

    setButtonLoading(el.runSingleBtn, true, 'Applying Channel Transforms...');
    if (el.expStatus) el.expStatus.textContent = 'RUNNING EXPERIMENT...';

    const formData = new FormData();
    if (state.uploadedFile) {
      formData.append('file', state.uploadedFile);
    } else {
      formData.append('sample_id', state.selectedSampleId);
    }

    const cond = state.condition === 'noise_snr' ? 'noise_snr_10db' : state.condition;
    formData.append('condition', cond);
    formData.append('snr_db', String(state.snrDb));
    formData.append('mitigation_enabled', state.mitigationEnabled ? 'true' : 'false');
    formData.append('seed', String(state.seed));

    try {
      const resp = await fetch('/api/robustness', {
        method: 'POST',
        body: formData
      });

      if (!resp.ok) {
        const errJson = await resp.json();
        throw new Error(errJson.detail || 'Experiment request failed');
      }

      const data = await resp.json();
      state.lastResult = data;
      displayExperimentResults(data);
    } catch (err) {
      console.error('Experiment error:', err);
      alert(`Experiment failed: ${err.message}`);
      if (el.expStatus) el.expStatus.textContent = 'EXPERIMENT FAILED';
    } finally {
      setButtonLoading(el.runSingleBtn, false, '▶ Run Single Experiment');
    }
  }

  async function displayExperimentResults(data) {
    if (el.expStatus) el.expStatus.textContent = 'EXPERIMENT COMPLETE';

    const comp = data.comparison;
    const cScore = comp.clean_bonafide_logit;
    const pScore = comp.degraded_bonafide_logit;
    const delta = comp.delta_logit;
    const absDelta = comp.absolute_delta_logit;
    const cPred = comp.clean_prediction;
    const pPred = comp.degraded_prediction;
    const pChanged = comp.prediction_changed;

    // Clean Baseline Card
    if (el.cleanScore) el.cleanScore.textContent = `${cScore >= 0 ? '+' : ''}${cScore.toFixed(4)}`;
    if (el.cleanPill) {
      el.cleanPill.textContent = `${cPred} (Baseline)`;
      el.cleanPill.className = `comp-pred-pill ${cPred.toLowerCase()}`;
    }
    if (el.cleanProb) el.cleanProb.textContent = `${(comp.clean_softmax_bonafide * 100).toFixed(2)}% bonafide`;

    // Processed Card
    if (el.procScore) el.procScore.textContent = `${pScore >= 0 ? '+' : ''}${pScore.toFixed(4)}`;
    if (el.procPill) {
      el.procPill.textContent = `${pPred} (${pChanged ? 'FLIPPED' : 'Retained'})`;
      el.procPill.className = `comp-pred-pill ${pPred.toLowerCase()}`;
    }
    if (el.procProb) el.procProb.textContent = `${(comp.degraded_softmax_bonafide * 100).toFixed(2)}% bonafide`;
    if (el.procTag) el.procTag.textContent = comp.mitigation_applied ? 'DEGRADED + MITIGATED' : 'DEGRADED (UNMITIGATED)';
    if (el.procCondition) el.procCondition.textContent = comp.mitigation_applied ? `${data.condition} + DSP Mitigation` : data.condition;

    // Delta Value & Badge
    if (el.deltaValue) el.deltaValue.textContent = `${delta >= 0 ? '+' : ''}${delta.toFixed(4)}`;
    if (el.boundaryBadge) {
      if (pChanged) {
        el.boundaryBadge.textContent = 'DECISION BOUNDARY BREACHED';
        el.boundaryBadge.className = 'boundary-badge breached';
      } else if (absDelta <= 1.0) {
        el.boundaryBadge.textContent = 'DECISION INVARIANT (|Δ| ≤ 1.0)';
        el.boundaryBadge.className = 'boundary-badge invariant';
      } else {
        el.boundaryBadge.textContent = 'DECISION RETAINED (SCORE SENSITIVE)';
        el.boundaryBadge.className = 'boundary-badge retained';
      }
    }

    // Explanation text
    if (el.deltaExplanation) {
      el.deltaExplanation.textContent = pChanged
        ? `Channel degradation caused a catastrophic shift of ${delta >= 0 ? '+' : ''}${delta.toFixed(4)} in the bonafide countermeasure logit, breaching the threshold θ = 0.00 and flipping the classification from ${cPred} to ${pPred}.`
        : `Channel degradation shifted the countermeasure logit by ${delta >= 0 ? '+' : ''}${delta.toFixed(4)}, but the score remained on the ${cPred} side of threshold θ = 0.00. Categorical classification remained intact.`;
    }

    // Decode processed audio base64 if present
    if (data.audio_wav_base64 && el.playProcBtn) {
      try {
        const audioBytes = hexToUint8Array(data.audio_wav_base64);
        const ctx = await getAudioContext();
        state.procAudioBuffer = await ctx.decodeAudioData(audioBytes.buffer.slice(0));
        el.playProcBtn.disabled = false;
        drawWaveform(state.procAudioBuffer, el.procWaveCanvas, '#A855F7');
      } catch (e) {
        console.warn('Could not decode processed audio payload:', e);
      }
    }
  }

  async function runMultiConditionSweep() {
    if (!state.selectedSampleId && !state.uploadedFile) {
      alert('Please select or upload an audio file first.');
      return;
    }

    setButtonLoading(el.runSweepBtn, true, 'Executing 8-Condition Sweep...');

    const formData = new FormData();
    if (state.uploadedFile) {
      formData.append('file', state.uploadedFile);
    } else {
      formData.append('sample_id', state.selectedSampleId);
    }
    formData.append('seed', String(state.seed));

    try {
      const resp = await fetch('/api/robustness/sweep', {
        method: 'POST',
        body: formData
      });

      if (!resp.ok) {
        const errJson = await resp.json();
        throw new Error(errJson.detail || 'Sweep request failed');
      }

      const data = await resp.json();
      state.sweepData = data;
      renderSweepMatrixTable(data);
      renderResponseCurveCanvas(data);
      if (el.sweepCountTag) el.sweepCountTag.textContent = `${data.trials_count} CONDITIONS EVALUATED`;
    } catch (err) {
      console.error('Sweep error:', err);
      alert(`Multi-condition sweep failed: ${err.message}`);
    } finally {
      setButtonLoading(el.runSweepBtn, false, '⚡ Multi-Condition Sweep Matrix');
    }
  }

  function renderSweepMatrixTable(data) {
    if (!el.sweepTableBody) return;
    const trials = data.trials || [];
    if (trials.length === 0) {
      el.sweepTableBody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">No sweep results available.</td></tr>`;
      return;
    }

    el.sweepTableBody.innerHTML = trials.map((t) => {
      const deltaSign = t.delta_score >= 0 ? `+${t.delta_score.toFixed(4)}` : t.delta_score.toFixed(4);
      const isBreached = t.prediction_changed;
      const badgeClass = isBreached ? 'badge-breached' : (Math.abs(t.delta_score) <= 1.0 ? 'badge-invariant' : 'badge-retained');

      return `
        <tr>
          <td>
            <div class="table-cond-name">${t.label}</div>
            <div class="table-cond-sub">${t.condition}</div>
          </td>
          <td class="font-mono text-silver">${t.clean_score.toFixed(4)}</td>
          <td class="font-mono text-electric">${t.processed_score.toFixed(4)}</td>
          <td class="font-mono ${t.delta_score < 0 ? 'text-red' : 'text-teal'}">${deltaSign}</td>
          <td><span class="status-indicator ${t.mitigation_applied ? 'active' : ''}">${t.mitigation_applied ? 'ENABLED' : 'OFF'}</span></td>
          <td><span class="comp-pred-pill-sm ${t.processed_prediction.toLowerCase()}">${t.processed_prediction}</span></td>
          <td><span class="status-pill-table ${badgeClass}">${t.badge}</span></td>
        </tr>
      `;
    }).join('');
  }

  function resetExperimentCardsToStandby() {
    if (el.expStatus) el.expStatus.textContent = 'STANDBY — READY';
    if (el.procScore) el.procScore.textContent = '—';
    if (el.procPill) {
      el.procPill.textContent = 'Awaiting Experiment';
      el.procPill.className = 'comp-pred-pill standby';
    }
    if (el.procProb) el.procProb.textContent = '—';
    if (el.procTag) el.procTag.textContent = 'DEGRADED / DEFENDED';
    if (el.procCondition) el.procCondition.textContent = 'Select condition & run';
    if (el.procWaveCanvas) {
      const ctx = el.procWaveCanvas.getContext('2d');
      ctx.clearRect(0, 0, el.procWaveCanvas.width, el.procWaveCanvas.height);
    }
    if (el.deltaValue) el.deltaValue.textContent = '—';
    if (el.boundaryBadge) {
      el.boundaryBadge.textContent = 'STANDBY';
      el.boundaryBadge.className = 'boundary-badge standby';
    }
    if (el.deltaExplanation) {
      el.deltaExplanation.textContent = 'Select a channel perturbation condition on the left and click "Run Single Experiment" or "Multi-Condition Sweep Matrix" to evaluate acoustic degradation and DSP defense.';
    }
    if (el.playProcBtn) el.playProcBtn.disabled = true;
  }

  function drawChartPlaceholder() {
    const canvas = el.responseCanvas;
    if (!canvas) return;
    const parent = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const rect = parent ? parent.getBoundingClientRect() : null;
    const paddingX = 28;
    const rawW = rect ? Math.floor(rect.width - paddingX) : ((parent && parent.clientWidth) ? parent.clientWidth - paddingX : 600);
    const w = Math.max(300, rawW);
    const h = 200;

    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = '100%';
    canvas.style.height = `${h}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    // Dark sleek background
    ctx.fillStyle = '#08080B';
    ctx.fillRect(0, 0, w, h);

    const padding = { top: 30, bottom: 42, left: 60, right: 30 };
    const chartW = w - padding.left - padding.right;
    const chartH = h - padding.top - padding.bottom;

    // Grid score marks
    const gridScores = [3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0];
    const minS = -4.0;
    const maxS = 4.0;
    const toY = (s) => padding.top + chartH * (1.0 - (s - minS) / (maxS - minS));

    gridScores.forEach((s) => {
      const y = toY(s);
      ctx.strokeStyle = s === 0.0 ? 'rgba(239, 68, 68, 0.5)' : 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = s === 0.0 ? 1.5 : 1;
      if (s === 0.0) ctx.setLineDash([4, 4]);
      else ctx.setLineDash([]);

      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(w - padding.right, y);
      ctx.stroke();

      ctx.fillStyle = s === 0.0 ? 'rgba(239, 68, 68, 0.85)' : 'rgba(156, 163, 175, 0.5)';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.fillText(s === 0.0 ? '0.00 (θ)' : s.toFixed(1), padding.left - 8, y + 3);
    });
    ctx.setLineDash([]);

    // Sample channels on X axis
    const placeholderChannels = ['RAW', 'N 0dB', 'N 10dB', 'N 20dB', 'TEL 8k', 'TEL+MIT', 'OPUS', 'OPUS+MIT'];
    const stepX = chartW / Math.max(1, placeholderChannels.length - 1);
    placeholderChannels.forEach((ch, i) => {
      const x = padding.left + i * stepX;
      ctx.fillStyle = 'rgba(156, 163, 175, 0.45)';
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(ch, x, h - 14);

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.beginPath();
      ctx.moveTo(x, h - padding.bottom);
      ctx.lineTo(x, h - padding.bottom + 5);
      ctx.stroke();
    });

    // Clean baseline indication line
    const baseY = toY(2.05);
    ctx.strokeStyle = 'rgba(96, 165, 250, 0.45)';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(padding.left, baseY);
    ctx.lineTo(w - padding.right, baseY);
    ctx.stroke();
    ctx.fillStyle = 'rgba(96, 165, 250, 0.75)';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    ctx.fillText('+2.05 (base)', padding.left - 8, baseY + 3);

    // Center callout / empty state badge
    const badgeW = Math.min(440, chartW - 20);
    const badgeH = 52;
    const bx = padding.left + (chartW - badgeW) / 2;
    const by = padding.top + (chartH - badgeH) / 2;

    ctx.fillStyle = 'rgba(17, 21, 29, 0.88)';
    ctx.strokeStyle = 'rgba(113, 153, 255, 0.28)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(bx, by, badgeW, badgeH, 6);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#60A5FA';
    ctx.font = 'bold 11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('⚡ AWAITING MULTI-CONDITION SWEEP EXECUTION', bx + badgeW / 2, by + 21);

    ctx.fillStyle = '#9CA3AF';
    ctx.font = '10.5px "Plus Jakarta Sans", sans-serif';
    ctx.fillText('Click "⚡ Multi-Condition Sweep Matrix" to evaluate & plot 8 degradation channels', bx + badgeW / 2, by + 38);
  }

  function renderResponseCurveCanvas(data) {
    const canvas = el.responseCanvas;
    if (!canvas) return;
    const parent = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const rect = parent ? parent.getBoundingClientRect() : null;
    const paddingX = 28;
    const rawW = rect ? Math.floor(rect.width - paddingX) : ((parent && parent.clientWidth) ? parent.clientWidth - paddingX : 600);
    const w = Math.max(300, rawW);
    const h = 200;

    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = '100%';
    canvas.style.height = `${h}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = '#08080B';
    ctx.fillRect(0, 0, w, h);

    const trials = (data.trials || []).filter((t) => t.status === 'SUCCESS');
    if (trials.length === 0) {
      drawChartPlaceholder();
      return;
    }

    const padding = { top: 30, bottom: 42, left: 60, right: 30 };
    const chartW = w - padding.left - padding.right;
    const chartH = h - padding.top - padding.bottom;

    // Compute dynamic min and max score
    let minScore = data.clean_baseline.score;
    let maxScore = data.clean_baseline.score;
    trials.forEach((t) => {
      if (t.processed_score < minScore) minScore = t.processed_score;
      if (t.processed_score > maxScore) maxScore = t.processed_score;
    });
    minScore = Math.min(-4.0, Math.floor(minScore) - 1);
    maxScore = Math.max(3.0, Math.ceil(maxScore) + 1);

    const scoreToY = (score) => {
      const ratio = (score - minScore) / (maxScore - minScore);
      return padding.top + chartH * (1.0 - ratio);
    };

    // Draw horizontal grid lines
    const stepVal = (maxScore - minScore) > 10 ? 2.0 : 1.0;
    for (let s = Math.ceil(minScore); s <= Math.floor(maxScore); s += stepVal) {
      const gy = scoreToY(s);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padding.left, gy);
      ctx.lineTo(w - padding.right, gy);
      ctx.stroke();

      ctx.fillStyle = 'rgba(156, 163, 175, 0.4)';
      ctx.font = '9.5px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.fillText(s.toFixed(1), padding.left - 8, gy + 3);
    }

    // Zero threshold line (theta = 0.00)
    const zeroY = scoreToY(0.0);
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.7)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padding.left, zeroY);
    ctx.lineTo(w - padding.right, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#EF4444';
    ctx.font = 'bold 9.5px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    ctx.fillText('0.00 (θ)', padding.left - 8, zeroY + 3);

    // Clean baseline horizontal reference line
    const baseY = scoreToY(data.clean_baseline.score);
    ctx.strokeStyle = 'rgba(96, 165, 250, 0.7)';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(padding.left, baseY);
    ctx.lineTo(w - padding.right, baseY);
    ctx.stroke();
    ctx.fillStyle = '#60A5FA';
    ctx.font = 'bold 9.5px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`${data.clean_baseline.score >= 0 ? '+' : ''}${data.clean_baseline.score.toFixed(2)} (base)`, padding.left - 8, baseY + 3);

    // Plot connecting line
    const stepX = chartW / Math.max(1, trials.length - 1);
    ctx.strokeStyle = '#A855F7';
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    trials.forEach((t, i) => {
      const x = padding.left + i * stepX;
      const y = scoreToY(t.processed_score);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Plot data points
    trials.forEach((t, i) => {
      const x = padding.left + i * stepX;
      const y = scoreToY(t.processed_score);

      // Outer glow halo
      ctx.fillStyle = t.mitigation_applied ? 'rgba(45, 212, 191, 0.25)' : 'rgba(168, 85, 247, 0.25)';
      ctx.beginPath();
      ctx.arc(x, y, 7, 0, Math.PI * 2);
      ctx.fill();

      // Inner dot
      ctx.fillStyle = t.mitigation_applied ? '#2DD4BF' : '#A855F7';
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();

      // Score text near point
      ctx.fillStyle = '#F3F4F6';
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      const scoreTxt = `${t.processed_score >= 0 ? '+' : ''}${t.processed_score.toFixed(1)}`;
      ctx.fillText(scoreTxt, x, y - 9);

      // Channel label on X axis below
      ctx.fillStyle = 'rgba(198, 203, 212, 0.7)';
      ctx.font = '9px "JetBrains Mono", monospace';
      const shortName = t.condition
        .replace('noise_snr_0db', 'N 0dB')
        .replace('noise_snr_10db', 'N 10dB')
        .replace('noise_snr_20db', 'N 20dB')
        .replace('telephony_8khz', 'TEL')
        .replace('opus_16kbps', 'OPUS');
      ctx.fillText(shortName + (t.mitigation_applied ? '+M' : ''), x, h - 14);
    });
  }

  function drawWaveform(buffer, canvas, color) {
    if (!canvas || !buffer) return;
    const ctx = canvas.getContext('2d');
    const w = (canvas.width = canvas.parentElement.clientWidth || 300);
    const h = (canvas.height = 60);

    ctx.clearRect(0, 0, w, h);
    const data = buffer.getChannelData(0);
    const numBars = 70;
    const step = Math.floor(data.length / numBars);
    const barWidth = Math.max(2, Math.floor(w / numBars) - 2);

    for (let i = 0; i < numBars; i++) {
      let max = 0;
      const start = i * step;
      for (let j = 0; j < step; j++) {
        const val = Math.abs(data[start + j] || 0);
        if (val > max) max = val;
      }
      const x = i * (barWidth + 2);
      const barH = Math.max(3, max * h * 0.85);
      const y = (h - barH) / 2;

      ctx.fillStyle = color || '#60A5FA';
      ctx.fillRect(x, y, barWidth, barH);
    }
  }

  async function playAudioBuffer(buffer) {
    if (!buffer) return;
    stopAudioPlayback();
    const ctx = await getAudioContext();
    const srcNode = ctx.createBufferSource();
    srcNode.buffer = buffer;
    srcNode.connect(ctx.destination);
    srcNode.onended = () => {
      state.isPlaying = false;
      if (el.playSourceBtn) el.playSourceBtn.textContent = '▶ Play Audio';
    };
    srcNode.start(0);
    state.activeAudioSourceNode = srcNode;
    state.isPlaying = true;
    if (el.playSourceBtn) el.playSourceBtn.textContent = '⏸ Pause';
  }

  function stopAudioPlayback() {
    if (state.activeAudioSourceNode) {
      try {
        state.activeAudioSourceNode.stop();
        state.activeAudioSourceNode.disconnect();
      } catch {}
      state.activeAudioSourceNode = null;
    }
    state.isPlaying = false;
    if (el.playSourceBtn) el.playSourceBtn.textContent = '▶ Play Audio';
  }

  function hexToUint8Array(hexString) {
    const bytes = new Uint8Array(hexString.length / 2);
    for (let i = 0; i < hexString.length; i += 2) {
      bytes[i / 2] = parseInt(hexString.substr(i, 2), 16);
    }
    return bytes;
  }

  function setButtonLoading(btn, isLoading, originalText) {
    if (!btn) return;
    btn.disabled = isLoading;
    btn.textContent = isLoading ? 'Processing...' : originalText;
  }

  document.addEventListener('DOMContentLoaded', init);
})();
