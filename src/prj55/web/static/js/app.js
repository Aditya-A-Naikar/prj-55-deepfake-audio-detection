/**
 * PRJ_55 Audio Forensics Platform — Application & State Controller
 * Manages model communication, audio analysis, waveform playback, and robustness testing.
 */

(function () {
  'use strict';

  // API Endpoints
  const API = {
    status: '/api/status',
    samples: '/api/samples',
    sampleAudio: (id) => `/api/samples/${id}/audio`,
    analyze: '/api/analyze',
    robustness: '/api/robustness',
    evaluation: '/api/evaluation',
    history: '/api/history',
  };

  // Application State
  const state = {
    currentFile: null,         // File object if user uploaded
    currentSampleId: null,     // string if preset selected
    currentAudioBuffer: null,  // Decoded AudioBuffer for Web Audio playback
    audioContext: null,
    audioSourceNode: null,
    playbackStartTime: 0,
    playbackOffset: 0,
    isPlaying: false,
    duration: 0,
    waveformPeaks: [],
    lastAnalysisResult: null,
    systemStatus: null,
  };

  // DOM Elements
  const el = {
    // Header & Status
    statusPill: document.getElementById('systemStatusPill'),
    statusText: document.getElementById('systemStatusText'),
    statusDot: document.querySelector('.status-dot'),

    // Hero Stats
    heroParams: document.getElementById('heroParams'),
    heroLatency: document.getElementById('heroLatency'),

    // Workspace & Upload
    dropZone: document.getElementById('dropZone'),
    fileInput: document.getElementById('audioFileInput'),
    sampleButtons: document.querySelectorAll('.preset-btn'),
    loadedFileStatus: document.getElementById('loadedFileStatus'),

    // Metadata
    metaFilename: document.getElementById('metaFilename'),
    metaSampleRate: document.getElementById('metaSampleRate'),
    metaChannels: document.getElementById('metaChannels'),
    metaDuration: document.getElementById('metaDuration'),

    // Player
    playBtn: document.getElementById('audioPlayBtn'),
    currentTime: document.getElementById('currentTime'),
    totalDuration: document.getElementById('totalDuration'),
    playerTrackName: document.getElementById('playerTrackName'),
    waveformCanvas: document.getElementById('waveformCanvas'),

    // Threshold & Run
    thresholdSlider: document.getElementById('thresholdSlider'),
    thresholdValue: document.getElementById('thresholdValue'),
    runInferenceBtn: document.getElementById('runInferenceBtn'),
    runSpinner: document.getElementById('runSpinner'),
    runBtnText: document.getElementById('runBtnText'),
    errorBanner: document.getElementById('analysisErrorBanner'),
    errorText: document.getElementById('analysisErrorText'),

    // Results
    resTimestamp: document.getElementById('resTimestamp'),
    verdictPill: document.getElementById('verdictPill'),
    verdictIcon: document.getElementById('verdictIcon'),
    verdictText: document.getElementById('verdictText'),
    verdictExplanation: document.getElementById('verdictExplanation'),
    meterNeedle: document.getElementById('meterNeedle'),
    meterThresholdMarker: document.getElementById('meterThresholdMarker'),
    meterThresholdLabel: document.getElementById('meterThresholdLabel'),
    resBonafideLogit: document.getElementById('resBonafideLogit'),
    resSpoofLogit: document.getElementById('resSpoofLogit'),
    resSoftmaxBonafide: document.getElementById('resSoftmaxBonafide'),
    resLatency: document.getElementById('resLatency'),
    resRtf: document.getElementById('resRtf'),
    exportReportBtn: document.getElementById('exportReportBtn'),
    spectrumCanvas: document.getElementById('spectrumCanvas'),

    // Robustness
    robRadios: document.querySelectorAll('input[name="robCondition"]'),
    opusRadioCard: document.getElementById('opusRadioCard'),
    opusRadio: document.getElementById('opusRadio'),
    opusRadioDesc: document.getElementById('opusRadioDesc'),
    snrSlider: document.getElementById('snrSlider'),
    snrValueDisplay: document.getElementById('snrValueDisplay'),
    mitigationToggle: document.getElementById('mitigationToggle'),
    runRobustnessBtn: document.getElementById('runRobustnessBtn'),
    robSpinner: document.getElementById('robSpinner'),
    robBtnText: document.getElementById('robBtnText'),
    robStatusBadge: document.getElementById('robStatusBadge'),
    cleanLogitDisplay: document.getElementById('cleanLogitDisplay'),
    cleanPredBadge: document.getElementById('cleanPredBadge'),
    degradedLogitDisplay: document.getElementById('degradedLogitDisplay'),
    degradedPredBadge: document.getElementById('degradedPredBadge'),
    deltaLogitDisplay: document.getElementById('deltaLogitDisplay'),
    deltaExplanation: document.getElementById('deltaExplanation'),
    degradedAudioElement: document.getElementById('degradedAudioElement'),

    // History
    historyCountText: document.getElementById('historyCountText'),
    historyTableBody: document.getElementById('historyTableBody'),
    exportHistoryBtn: document.getElementById('exportHistoryBtn'),
  };

  // -------------------------------------------------------------------------
  // INITIALIZATION
  // -------------------------------------------------------------------------

  async function init() {
    setupEventListeners();
    await checkSystemStatus();
    await refreshSessionHistory();
  }

  // -------------------------------------------------------------------------
  // SYSTEM STATUS
  // -------------------------------------------------------------------------

  async function checkSystemStatus() {
    try {
      const resp = await fetch(API.status);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      state.systemStatus = data;

      if (data.status === 'online') {
        el.statusDot.className = 'status-dot online';
        el.statusText.textContent = `AASIST-L • ${data.device.toUpperCase()} ONLINE`;
        if (data.parameter_count && el.heroParams) {
          el.heroParams.textContent = data.parameter_count.toLocaleString();
        }
      } else {
        el.statusDot.className = 'status-dot error';
        el.statusText.textContent = 'MODEL UNAVAILABLE';
      }

      // Check Opus codec availability
      if (data.codecs && !data.codecs.opus_available) {
        if (el.opusRadio) el.opusRadio.disabled = true;
        if (el.opusRadioDesc) {
          el.opusRadioDesc.textContent = 'FFmpeg/libopus not detected on host. (brew install ffmpeg)';
        }
        if (el.opusRadioCard) {
          el.opusRadioCard.style.opacity = '0.5';
          el.opusRadioCard.title = 'Requires system FFmpeg installation';
        }
      }
    } catch (err) {
      console.warn('System status fetch failed:', err);
      el.statusDot.className = 'status-dot error';
      el.statusText.textContent = 'OFFLINE / DISCONNECTED';
    }
  }

  // -------------------------------------------------------------------------
  // EVENT LISTENERS
  // -------------------------------------------------------------------------

  function setupEventListeners() {
    // Drop Zone Click & Drag
    el.dropZone.addEventListener('click', () => el.fileInput.click());
    el.dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      el.dropZone.classList.add('dragover');
    });
    el.dropZone.addEventListener('dragleave', () => {
      el.dropZone.classList.remove('dragover');
    });
    el.dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      el.dropZone.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {
        handleUserFileUpload(e.dataTransfer.files[0]);
      }
    });

    el.fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        handleUserFileUpload(e.target.files[0]);
      }
    });

    // Sample Preset Buttons
    el.sampleButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        const sampleId = btn.getAttribute('data-sample');
        handleSelectSample(sampleId, btn);
      });
    });

    // Threshold Slider
    el.thresholdSlider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value).toFixed(2);
      el.thresholdValue.textContent = val;
      el.meterThresholdLabel.textContent = val;
      updateMeterMarker(parseFloat(val));
    });

    // Run Inference Button
    el.runInferenceBtn.addEventListener('click', runAnalysis);

    // Audio Player Play/Pause
    el.playBtn.addEventListener('click', toggleAudioPlayback);

    // SNR Slider
    el.snrSlider.addEventListener('input', (e) => {
      el.snrValueDisplay.textContent = `${e.target.value} dB`;
    });

    // Run Robustness Button
    el.runRobustnessBtn.addEventListener('click', runRobustnessExperiment);

    // Export Reports
    el.exportReportBtn.addEventListener('click', exportForensicReceipt);
    el.exportHistoryBtn.addEventListener('click', exportSessionHistory);
  }

  // -------------------------------------------------------------------------
  // AUDIO FILE & SAMPLE LOADING
  // -------------------------------------------------------------------------

  async function handleUserFileUpload(file) {
    state.currentFile = file;
    state.currentSampleId = null;

    // Reset preset buttons active state
    el.sampleButtons.forEach((b) => b.classList.remove('active'));

    el.loadedFileStatus.textContent = 'FILE LOADED';
    el.metaFilename.textContent = file.name;
    el.metaFilename.title = file.name;
    el.metaChannels.textContent = 'Mono / Stereo (Downmixed)';
    el.metaSampleRate.textContent = 'Standardized to 16 kHz';
    el.playerTrackName.textContent = file.name;

    hideError();

    // Decode audio in browser for playback and waveform display
    try {
      const arrayBuffer = await file.arrayBuffer();
      await decodeAndDisplayAudio(arrayBuffer);
      enableActionButtons();
    } catch (err) {
      showError(`Could not decode audio file: ${err.message}`);
    }
  }

  async function handleSelectSample(sampleId, activeBtn) {
    state.currentFile = null;
    state.currentSampleId = sampleId;

    el.sampleButtons.forEach((b) => b.classList.remove('active'));
    activeBtn.classList.add('active');

    el.loadedFileStatus.textContent = 'SAMPLE PRESET LOADED';
    el.metaFilename.textContent = `${sampleId}.wav`;
    el.metaSampleRate.textContent = '16,000 Hz';
    el.metaChannels.textContent = '1 (Mono)';
    el.metaDuration.textContent = '4.000 s (64,600 samples)';
    el.playerTrackName.textContent = activeBtn.querySelector('.preset-name').textContent;

    hideError();

    try {
      const resp = await fetch(API.sampleAudio(sampleId));
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const arrayBuffer = await resp.arrayBuffer();
      await decodeAndDisplayAudio(arrayBuffer);
      enableActionButtons();
    } catch (err) {
      showError(`Could not load preset sample audio: ${err.message}`);
    }
  }

  async function decodeAndDisplayAudio(arrayBuffer) {
    if (!state.audioContext) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      state.audioContext = new AudioCtx({ sampleRate: 16000 });
    }

    const audioBuffer = await state.audioContext.decodeAudioData(arrayBuffer.slice(0));
    state.currentAudioBuffer = audioBuffer;
    state.duration = audioBuffer.duration;

    el.metaDuration.textContent = `${audioBuffer.duration.toFixed(3)} s (${audioBuffer.length.toLocaleString()} samples)`;
    el.totalDuration.textContent = formatTime(audioBuffer.duration);
    el.currentTime.textContent = '00:00';
    el.playBtn.disabled = false;

    // Extract peaks for waveform canvas
    const rawData = audioBuffer.getChannelData(0);
    const numBins = 160;
    const step = Math.floor(rawData.length / numBins);
    const peaks = [];
    for (let i = 0; i < numBins; i++) {
      let maxVal = 0;
      for (let j = 0; j < step; j++) {
        const val = Math.abs(rawData[i * step + j] || 0);
        if (val > maxVal) maxVal = val;
      }
      peaks.push(maxVal);
    }
    state.waveformPeaks = peaks;
    drawWaveform(peaks, 0);
  }

  function enableActionButtons() {
    el.runInferenceBtn.disabled = false;
    el.runRobustnessBtn.disabled = false;
  }

  // -------------------------------------------------------------------------
  // WAVEFORM CANVAS & PLAYBACK
  // -------------------------------------------------------------------------

  function drawWaveform(peaks, progressRatio = 0) {
    const canvas = el.waveformCanvas;
    const ctx = canvas.getContext('2d');
    const w = (canvas.width = canvas.parentElement.clientWidth || 560);
    const h = (canvas.height = canvas.parentElement.clientHeight || 90);

    ctx.clearRect(0, 0, w, h);

    if (!peaks || peaks.length === 0) return;

    const barWidth = w / peaks.length;
    const midY = h / 2;

    for (let i = 0; i < peaks.length; i++) {
      const barHeight = Math.max(2, peaks[i] * (h * 0.85));
      const x = i * barWidth;
      const isPlayed = i / peaks.length <= progressRatio;

      ctx.fillStyle = isPlayed ? '#438BFF' : 'rgba(198, 203, 212, 0.35)';
      ctx.fillRect(x, midY - barHeight / 2, barWidth - 1, barHeight);
    }

    // Playhead line
    if (progressRatio > 0 && progressRatio < 1) {
      const playheadX = progressRatio * w;
      ctx.fillStyle = '#FFFFFF';
      ctx.fillRect(playheadX - 1, 0, 2, h);
    }
  }

  function toggleAudioPlayback() {
    if (!state.currentAudioBuffer) return;

    if (state.isPlaying) {
      stopAudioPlayback();
    } else {
      startAudioPlayback();
    }
  }

  function startAudioPlayback() {
    if (!state.audioContext) return;
    if (state.audioContext.state === 'suspended') {
      state.audioContext.resume();
    }

    const source = state.audioContext.createBufferSource();
    source.buffer = state.currentAudioBuffer;
    source.connect(state.audioContext.destination);

    source.onended = () => {
      stopAudioPlayback();
      el.currentTime.textContent = '00:00';
      drawWaveform(state.waveformPeaks, 0);
    };

    source.start(0, state.playbackOffset);
    state.playbackStartTime = state.audioContext.currentTime - state.playbackOffset;
    state.audioSourceNode = source;
    state.isPlaying = true;
    el.playBtn.textContent = '⏸';

    requestAnimationFrame(updatePlaybackProgress);
  }

  function stopAudioPlayback() {
    if (state.audioSourceNode) {
      try {
        state.audioSourceNode.stop();
        state.audioSourceNode.disconnect();
      } catch (_) {}
      state.audioSourceNode = null;
    }
    state.isPlaying = false;
    state.playbackOffset = 0;
    el.playBtn.textContent = '▶';
  }

  function updatePlaybackProgress() {
    if (!state.isPlaying || !state.audioContext) return;

    const elapsed = state.audioContext.currentTime - state.playbackStartTime;
    if (elapsed < state.duration) {
      el.currentTime.textContent = formatTime(elapsed);
      const ratio = elapsed / state.duration;
      drawWaveform(state.waveformPeaks, ratio);
      requestAnimationFrame(updatePlaybackProgress);
    } else {
      stopAudioPlayback();
      drawWaveform(state.waveformPeaks, 0);
    }
  }

  function formatTime(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  }

  // -------------------------------------------------------------------------
  // RUN AASIST-L INFERENCE
  // -------------------------------------------------------------------------

  async function runAnalysis() {
    if (!state.currentFile && !state.currentSampleId) return;

    setRunLoading(true);
    hideError();

    // Trigger pulse on the 3D signal sculpture
    if (window.triggerSculpturePulse) {
      window.triggerSculpturePulse(3.2);
    }

    try {
      const formData = new FormData();
      const thresholdVal = parseFloat(el.thresholdSlider.value);
      formData.append('threshold', String(thresholdVal));

      if (state.currentFile) {
        formData.append('file', state.currentFile);
      } else if (state.currentSampleId) {
        formData.append('sample_id', state.currentSampleId);
      }

      const resp = await fetch(API.analyze, {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) {
        const errorData = await resp.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${resp.status}`);
      }

      const result = await resp.json();
      state.lastAnalysisResult = result;
      displayResults(result);
      await refreshSessionHistory();

      // Smooth scroll to results
      document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
    } catch (err) {
      showError(err.message);
    } finally {
      setRunLoading(false);
    }
  }

  function displayResults(data) {
    el.resTimestamp.textContent = data.timestamp;

    // Verdict Pill
    const isBonafide = data.prediction === 'BONAFIDE';
    el.verdictPill.className = `verdict-pill-large ${isBonafide ? 'bonafide' : 'spoof'}`;
    el.verdictIcon.textContent = isBonafide ? '✓' : '⚠';
    el.verdictText.textContent = isBonafide ? 'BONAFIDE (AUTHENTIC)' : 'SPOOF (SYNTHETIC / DEEPFAKE)';

    el.verdictExplanation.textContent = isBonafide
      ? `AASIST-L countermeasure score (${data.primary_score}) sits at or above the operating threshold (${data.scores.threshold}). Spectral-temporal graph attention exhibits natural glottal harmonic distributions.`
      : `AASIST-L countermeasure score (${data.primary_score}) sits below the operating threshold (${data.scores.threshold}). Neural attention detects phase inconsistencies, vocoder buzzing, or synthetic artifact signatures.`;

    // Scores & Latency
    el.resBonafideLogit.textContent = String(data.scores.bonafide_logit);
    el.resSpoofLogit.textContent = String(data.scores.spoof_logit);
    el.resSoftmaxBonafide.textContent = `${(data.scores.softmax_bonafide * 100).toFixed(2)}%`;
    el.resLatency.textContent = `${data.performance.latency_ms} ms`;
    el.resRtf.textContent = `${data.performance.real_time_factor}x`;

    // Meter Needle Position: Map score [-2.0, +2.0] to [0%, 100%]
    const score = data.scores.bonafide_logit;
    const clampedScore = Math.max(-2.0, Math.min(2.0, score));
    const needlePercent = ((clampedScore + 2.0) / 4.0) * 100.0;
    el.meterNeedle.style.left = `${needlePercent}%`;

    // Threshold Marker
    updateMeterMarker(data.scores.threshold);

    // Draw Frequency Spectrogram
    if (data.visual_features && data.visual_features.spectrum_magnitudes) {
      drawSpectrogram(data.visual_features.spectrum_magnitudes);
    }

    el.exportReportBtn.disabled = false;
  }

  function updateMeterMarker(threshold) {
    const clamped = Math.max(-2.0, Math.min(2.0, threshold));
    const percent = ((clamped + 2.0) / 4.0) * 100.0;
    el.meterThresholdMarker.style.left = `${percent}%`;
  }

  function drawSpectrogram(magnitudes) {
    const canvas = el.spectrumCanvas;
    const ctx = canvas.getContext('2d');
    const w = (canvas.width = canvas.parentElement.clientWidth || 560);
    const h = (canvas.height = canvas.parentElement.clientHeight || 190);

    ctx.clearRect(0, 0, w, h);

    if (!magnitudes || magnitudes.length === 0) return;

    const barWidth = w / magnitudes.length;
    const maxMag = Math.max(...magnitudes, 1e-4);

    for (let i = 0; i < magnitudes.length; i++) {
      const norm = magnitudes[i] / maxMag;
      const barH = norm * (h * 0.88);
      const x = i * barWidth;
      const y = h - barH;

      // Color gradient: aubergine -> electric blue -> pale lavender
      const grad = ctx.createLinearGradient(0, y, 0, h);
      grad.addColorStop(0, '#C9B9FF');
      grad.addColorStop(0.4, '#438BFF');
      grad.addColorStop(1, '#482746');

      ctx.fillStyle = grad;
      ctx.fillRect(x, y, barWidth - 1, barH);
    }
  }

  function setRunLoading(isLoading) {
    el.runInferenceBtn.disabled = isLoading;
    if (isLoading) {
      el.runSpinner.classList.remove('hidden');
      el.runBtnText.textContent = 'EXECUTING INFERENCE...';
    } else {
      el.runSpinner.classList.add('hidden');
      el.runBtnText.textContent = 'RUN FORENSIC ANALYSIS';
    }
  }

  // -------------------------------------------------------------------------
  // ROBUSTNESS EXPERIMENTS
  // -------------------------------------------------------------------------

  async function runRobustnessExperiment() {
    if (!state.currentFile && !state.currentSampleId) return;

    setRobLoading(true);

    try {
      // Find checked condition radio
      let selectedCondition = 'telephony_8khz';
      el.robRadios.forEach((r) => {
        if (r.checked) selectedCondition = r.value;
      });

      const snrVal = parseFloat(el.snrSlider.value);
      const isMitigated = el.mitigationToggle.checked;

      const formData = new FormData();
      formData.append('condition', selectedCondition);
      formData.append('snr_db', String(snrVal));
      formData.append('mitigation_enabled', String(isMitigated));

      if (state.currentFile) {
        formData.append('file', state.currentFile);
      } else if (state.currentSampleId) {
        formData.append('sample_id', state.currentSampleId);
      }

      const resp = await fetch(API.robustness, {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${resp.status}`);
      }

      const result = await resp.json();
      displayRobustnessResults(result);
      await refreshSessionHistory();
    } catch (err) {
      showError(`Robustness test failed: ${err.message}`);
    } finally {
      setRobLoading(false);
    }
  }

  function displayRobustnessResults(data) {
    if (data.status === 'UNAVAILABLE') {
      el.robStatusBadge.textContent = 'UNAVAILABLE';
      el.deltaExplanation.textContent = data.error;
      return;
    }

    el.robStatusBadge.textContent = 'COMPLETED';

    const clean = data.comparison.clean_bonafide_logit;
    const deg = data.comparison.degraded_bonafide_logit;
    const delta = data.comparison.delta_logit;
    const driftPct = data.comparison.score_drift_percentage;

    el.cleanLogitDisplay.textContent = String(clean);
    el.cleanPredBadge.textContent = data.comparison.clean_prediction;
    el.degradedLogitDisplay.textContent = String(deg);
    el.degradedPredBadge.textContent = data.comparison.degraded_prediction;

    el.deltaLogitDisplay.textContent = (delta >= 0 ? `+${delta}` : `${delta}`);
    el.deltaExplanation.textContent = `Score drift of ${driftPct}% observed under condition [${data.condition}]. ${
      data.parameters.mitigation_applied
        ? 'Front-end spectral mitigation was applied.'
        : 'Source was evaluated without front-end filtering.'
    }`;

    // Playback degraded audio if hex bytes returned
    if (data.audio_wav_base64) {
      const byteCharacters = hexToBytes(data.audio_wav_base64);
      const blob = new Blob([byteCharacters], { type: 'audio/wav' });
      const blobUrl = URL.createObjectURL(blob);
      el.degradedAudioElement.src = blobUrl;
    }
  }

  function hexToBytes(hexString) {
    const bytes = new Uint8Array(hexString.length / 2);
    for (let i = 0; i < bytes.length; i++) {
      bytes[i] = parseInt(hexString.substr(i * 2, 2), 16);
    }
    return bytes;
  }

  function setRobLoading(isLoading) {
    el.runRobustnessBtn.disabled = isLoading;
    if (isLoading) {
      el.robSpinner.classList.remove('hidden');
      el.robBtnText.textContent = 'PROCESSING CHANNEL PERTURBATION...';
    } else {
      el.robSpinner.classList.add('hidden');
      el.robBtnText.textContent = 'RUN PERTURBATION TEST';
    }
  }

  // -------------------------------------------------------------------------
  // SESSION HISTORY & EXPORT
  // -------------------------------------------------------------------------

  async function refreshSessionHistory() {
    try {
      const resp = await fetch(API.history);
      if (!resp.ok) return;
      const history = await resp.json();

      el.historyCountText.textContent = `${history.length} records in current session`;

      if (history.length === 0) {
        el.historyTableBody.innerHTML = `
          <tr class="empty-row">
            <td colspan="6" class="text-center text-muted">No analysis trials conducted yet in this session.</td>
          </tr>
        `;
        return;
      }

      el.historyTableBody.innerHTML = history
        .slice()
        .reverse()
        .map(
          (item) => `
          <tr>
            <td style="font-family: var(--font-mono); font-size: 11px;">${item.timestamp}</td>
            <td style="font-weight: 600;">${escapeHtml(item.filename)}</td>
            <td><span class="format-pill">${escapeHtml(item.condition)}</span></td>
            <td>
              <span class="pill-badge ${item.prediction === 'BONAFIDE' ? 'success' : 'warning'}">
                ${item.prediction}
              </span>
            </td>
            <td style="font-family: var(--font-mono);">${item.bonafide_logit}</td>
            <td style="font-family: var(--font-mono);">${item.latency_ms} ms</td>
          </tr>
        `
        )
        .join('');
    } catch (_) {}
  }

  function exportForensicReceipt() {
    if (!state.lastAnalysisResult) return;
    const blob = new Blob([JSON.stringify(state.lastAnalysisResult, null, 2)], {
      type: 'application/json',
    });
    downloadBlob(blob, `PRJ55_Forensic_Receipt_${Date.now()}.json`);
  }

  async function exportSessionHistory() {
    try {
      const resp = await fetch(API.history);
      if (!resp.ok) return;
      const history = await resp.json();
      const blob = new Blob([JSON.stringify(history, null, 2)], {
        type: 'application/json',
      });
      downloadBlob(blob, `PRJ55_Session_History_${Date.now()}.json`);
    } catch (_) {}
  }

  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function showError(msg) {
    el.errorText.textContent = msg;
    el.errorBanner.classList.remove('hidden');
  }

  function hideError() {
    el.errorBanner.classList.add('hidden');
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // Run on page load
  window.addEventListener('DOMContentLoaded', init);
})();
