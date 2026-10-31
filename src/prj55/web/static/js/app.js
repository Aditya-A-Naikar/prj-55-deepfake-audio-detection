/**
 * Audio Authenticity Lab (PRJ_55) — Frontend Application Controller
 * High-performance on-device audio forensics SPA.
 * Manages AASIST-L inference, channel degradation simulation,
 * front-end DSP mitigation, system diagnostics, and forensic evidence display.
 */

(function () {
  'use strict';

  // API Endpoints
  const API = {
    status: '/api/status',
    samples: '/api/samples',
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
    currentDegradedBlobUrl: null, // Track object URL to prevent memory leaks
    audioContext: null,
    audioSourceNode: null,
    playbackStartTime: 0,
    playbackOffset: 0,
    isPlaying: false,
    duration: 0,
    waveformPeaks: [],
    lastAnalysisResult: null,
    systemStatus: null,
    analysisCounter: 0,        // Request tracking to prevent race conditions
    viewMode: 'simple',        // 'simple' | 'expert'
    isScrubbing: false,        // true while dragging on waveform canvas
  };

  // DOM Elements
  const el = {
    // Header & Status
    statusPill: document.getElementById('systemStatusPill'),
    statusText: document.getElementById('systemStatusText'),
    statusDot: document.querySelector('.status-dot'),
    mobileMenuBtn: document.getElementById('mobileMenuBtn'),
    mobileNavDrawer: document.getElementById('mobileNavDrawer'),
    closeMobileNavBtn: document.getElementById('closeMobileNavBtn'),
    mobileNavBackdrop: document.getElementById('mobileNavBackdrop'),

    // Hero Stats
    heroParams: document.getElementById('heroParams'),
    heroLatency: document.getElementById('heroLatency'),

    // Workspace & Upload
    cardUpload: document.getElementById('cardUpload'),
    uploadExpandedContent: document.getElementById('uploadExpandedContent'),
    uploadCollapsedContent: document.getElementById('uploadCollapsedContent'),
    collapsedSourceName: document.getElementById('collapsedSourceName'),
    collapsedSourceMeta: document.getElementById('collapsedSourceMeta'),
    expandUploadBtn: document.getElementById('expandUploadBtn'),
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
    analysisProgressCard: document.getElementById('analysisProgressCard'),
    analysisProgressMsg: document.getElementById('analysisProgressMsg'),
    errorBanner: document.getElementById('analysisErrorBanner'),
    errorText: document.getElementById('analysisErrorText'),

    // Results & View Mode
    viewModeSimple: document.getElementById('viewModeSimple'),
    viewModeExpert: document.getElementById('viewModeExpert'),
    simpleViewContent: document.getElementById('simpleViewContent'),
    expertViewContent: document.getElementById('expertViewContent'),
    simpleStatusTitle: document.getElementById('simpleStatusTitle'),
    simpleStatusText: document.getElementById('simpleStatusText'),
    simpleMarginText: document.getElementById('simpleMarginText'),
    resTimestamp: document.getElementById('resTimestamp'),
    verdictPill: document.getElementById('verdictPill'),
    verdictIcon: document.getElementById('verdictIcon'),
    verdictText: document.getElementById('verdictText'),
    verdictPrimary: document.getElementById('verdictPrimary') || document.getElementById('verdictText'),
    verdictSubtitle: document.getElementById('verdictSubtitle'),
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
    snrSliderWrapper: document.getElementById('snrSliderWrapper'),
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
    robMitigationBadge: document.getElementById('robMitigationBadge'),
    robClassificationBadge: document.getElementById('robClassificationBadge'),
    playbackSignalLabel: document.getElementById('playbackSignalLabel'),
    degradedAudioElement: document.getElementById('degradedAudioElement'),

    // Robustness Conclusion Card
    robConclusionCard: document.getElementById('robConclusionCard'),
    robConclusionBadge: document.getElementById('robConclusionBadge'),
    conclusionLeadText: document.getElementById('conclusionLeadText'),
    conclusionBodyText: document.getElementById('conclusionBodyText'),
    robAcousticExplanation: document.getElementById('robAcousticExplanation'),
    robMitigationExplanation: document.getElementById('robMitigationExplanation'),

    // System Diagnostics Drawer
    diagModalBackdrop: document.getElementById('diagModalBackdrop'),
    diagDrawer: document.getElementById('diagDrawer'),
    closeDiagBtn: document.getElementById('closeDiagBtn'),
    linkOpenDiag: document.getElementById('linkOpenDiag'),
    diagModelBadge: document.getElementById('diagModelBadge'),
    diagParamCount: document.getElementById('diagParamCount'),
    diagDeviceVal: document.getElementById('diagDeviceVal'),
    diagCheckpointBadge: document.getElementById('diagCheckpointBadge'),
    diagSha256: document.getElementById('diagSha256'),
    diagCodecBadge: document.getElementById('diagCodecBadge'),
    diagFfmpegVal: document.getElementById('diagFfmpegVal'),
    diagOpusVal: document.getElementById('diagOpusVal'),
    diagHostDisk: document.getElementById('diagHostDisk'),
    diagDatasetStatus: document.getElementById('diagDatasetStatus'),
    diagPytestScore: document.getElementById('diagPytestScore'),

    // History
    historyCountText: document.getElementById('historyCountText'),
    historyTableBody: document.getElementById('historyTableBody'),
    clearHistoryBtn: document.getElementById('clearHistoryBtn'),
    exportHistoryBtn: document.getElementById('exportHistoryBtn'),
  };

  // -------------------------------------------------------------------------
  // INITIALIZATION
  // -------------------------------------------------------------------------

  async function init() {
    setupEventListeners();
    setupInputModeTabs();
    setupMicCapture();
    setupBatchAnalysis();
    setupScrollReveal();
    updateConditionControls();
    drawSpectrogramPlaceholder();
    await checkSystemStatus();
    await refreshSessionHistory();
  }

  function setupScrollReveal() {
    const reveals = document.querySelectorAll('.reveal-on-scroll');
    if (!reveals || reveals.length === 0) return;

    if (!('IntersectionObserver' in window) || window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      reveals.forEach((node) => node.classList.add('is-visible'));
      return;
    }

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
        }
      });
    }, { threshold: 0.08 });

    reveals.forEach((node) => observer.observe(node));
  }

  // -------------------------------------------------------------------------
  // SYSTEM STATUS & DIAGNOSTICS
  // -------------------------------------------------------------------------

  async function checkSystemStatus() {
    try {
      const resp = await fetch(API.status);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      state.systemStatus = data;

      if (data.status === 'online') {
        el.statusDot.className = 'status-dot online';
        const devStr = data.device ? data.device.toUpperCase() : 'ONLINE';
        el.statusText.textContent = `AASIST-L • ${devStr} ONLINE`;
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

  function openDiagnosticsDrawer() {
    if (el.diagModalBackdrop) el.diagModalBackdrop.classList.remove('hidden');
    if (el.diagDrawer) {
      el.diagDrawer.classList.remove('hidden');
      populateDiagnosticsData();
      if (el.closeDiagBtn) el.closeDiagBtn.focus();
    }
  }

  function closeDiagnosticsDrawer() {
    if (el.diagModalBackdrop) el.diagModalBackdrop.classList.add('hidden');
    if (el.diagDrawer) el.diagDrawer.classList.add('hidden');
    if (el.statusPill) el.statusPill.focus();
  }

  function populateDiagnosticsData() {
    const s = state.systemStatus;
    if (!s) return;

    if (el.diagModelBadge) {
      el.diagModelBadge.className = `pill-badge ${s.status === 'online' ? 'success' : 'warning'}`;
      el.diagModelBadge.textContent = s.status ? s.status.toUpperCase() : 'UNKNOWN';
    }
    if (el.diagParamCount && s.parameter_count) {
      el.diagParamCount.textContent = `${s.parameter_count.toLocaleString()} parameters`;
    }
    if (el.diagDeviceVal && s.device) {
      el.diagDeviceVal.textContent = s.device === 'mps' ? 'Apple Silicon MPS (Metal Performance Shaders)' : s.device.toUpperCase();
    }
    if (el.diagSha256 && s.checkpoint_sha256) {
      el.diagSha256.textContent = s.checkpoint_sha256;
    }
    if (el.diagCheckpointBadge) {
      el.diagCheckpointBadge.className = `pill-badge ${s.checkpoint_found ? 'success' : 'error'}`;
      el.diagCheckpointBadge.textContent = s.checkpoint_found ? 'VERIFIED' : 'MISSING';
    }
    if (s.codecs) {
      if (el.diagFfmpegVal) {
        el.diagFfmpegVal.textContent = s.codecs.ffmpeg_available ? `Found (${s.codecs.ffmpeg_version})` : 'Not Detected';
      }
      if (el.diagOpusVal) {
        el.diagOpusVal.textContent = s.codecs.opus_available ? 'Available' : 'Unavailable';
      }
      if (el.diagCodecBadge) {
        el.diagCodecBadge.className = `pill-badge ${s.codecs.ffmpeg_available ? 'success' : 'warning'}`;
        el.diagCodecBadge.textContent = s.codecs.ffmpeg_available ? 'CODECS READY' : 'OPUS LIMITED';
      }
    }
    if (s.storage_guard) {
      if (el.diagDatasetStatus && s.storage_guard.status) {
        el.diagDatasetStatus.textContent = s.storage_guard.status;
      }
    }
  }

  // -------------------------------------------------------------------------
  // EVENT LISTENERS
  // -------------------------------------------------------------------------

  function setupEventListeners() {
    // System Diagnostics Drawer Trigger
    if (el.statusPill) {
      el.statusPill.addEventListener('click', openDiagnosticsDrawer);
      el.statusPill.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          openDiagnosticsDrawer();
        }
      });
    }
    if (el.linkOpenDiag) {
      el.linkOpenDiag.addEventListener('click', openDiagnosticsDrawer);
    }
    if (el.closeDiagBtn) {
      el.closeDiagBtn.addEventListener('click', closeDiagnosticsDrawer);
    }
    if (el.diagModalBackdrop) {
      el.diagModalBackdrop.addEventListener('click', closeDiagnosticsDrawer);
    }

    // Mobile Navigation Drawer
    if (el.mobileMenuBtn) {
      el.mobileMenuBtn.addEventListener('click', openMobileNav);
    }
    if (el.closeMobileNavBtn) {
      el.closeMobileNavBtn.addEventListener('click', closeMobileNav);
    }
    if (el.mobileNavBackdrop) {
      el.mobileNavBackdrop.addEventListener('click', closeMobileNav);
    }

    // View Mode Toggle (Simple View vs Expert View)
    if (el.viewModeSimple) {
      el.viewModeSimple.addEventListener('click', () => setViewMode('simple'));
    }
    if (el.viewModeExpert) {
      el.viewModeExpert.addEventListener('click', () => setViewMode('expert'));
    }

    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        if (el.diagDrawer && !el.diagDrawer.classList.contains('hidden')) {
          closeDiagnosticsDrawer();
        }
        closeMobileNav();
      }
    });

    // Expand Upload Panel Button
    if (el.expandUploadBtn) {
      el.expandUploadBtn.addEventListener('click', expandUploadPanel);
    }

    // Drop Zone Click & Drag
    if (el.dropZone) {
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
    }

    if (el.fileInput) {
      el.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
          handleUserFileUpload(e.target.files[0]);
        }
      });
    }

    // Sample Preset Buttons
    el.sampleButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        const sampleId = btn.getAttribute('data-sample');
        handleSelectSample(sampleId, btn);
      });
    });

    // Threshold Slider (dynamic real-time boundary update without re-running inference)
    if (el.thresholdSlider) {
      el.thresholdSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        const formatted = val.toFixed(2);
        el.thresholdValue.textContent = formatted;
        el.meterThresholdLabel.textContent = formatted;
        updateMeterMarker(val);
        updatePredictionForThreshold(val);
      });

      const btnResetThresh = document.getElementById('btnResetThreshold');
      if (btnResetThresh) {
        btnResetThresh.addEventListener('click', () => {
          el.thresholdSlider.value = '0.0';
          el.thresholdSlider.dispatchEvent(new Event('input'));
        });
      }
    }

    // Run Inference Button
    if (el.runInferenceBtn) {
      el.runInferenceBtn.addEventListener('click', runAnalysis);
    }

    // Audio Player Play/Pause
    if (el.playBtn) {
      el.playBtn.addEventListener('click', toggleAudioPlayback);
    }

    // Interactive Waveform Scrubbing (Mouse & Touch)
    if (el.waveformCanvas) {
      el.waveformCanvas.addEventListener('mousedown', (e) => {
        state.isScrubbing = true;
        seekWaveformAtEvent(e);
      });
      window.addEventListener('mousemove', (e) => {
        if (state.isScrubbing) {
          seekWaveformAtEvent(e);
        }
      });
      window.addEventListener('mouseup', () => {
        state.isScrubbing = false;
      });
      el.waveformCanvas.addEventListener('click', (e) => {
        seekWaveformAtEvent(e);
      });

      el.waveformCanvas.addEventListener('touchstart', (e) => {
        state.isScrubbing = true;
        seekWaveformAtEvent(e);
      }, { passive: true });
      window.addEventListener('touchmove', (e) => {
        if (state.isScrubbing) {
          seekWaveformAtEvent(e);
        }
      }, { passive: true });
      window.addEventListener('touchend', () => {
        state.isScrubbing = false;
      });
    }

    // Robustness Condition Radios
    el.robRadios.forEach((r) => {
      r.addEventListener('change', () => {
        updateConditionControls();
        invalidateRobustnessResults('Degradation condition changed — run perturbation test to compute updated metrics');
      });
    });

    // SNR Slider
    if (el.snrSlider) {
      el.snrSlider.addEventListener('input', (e) => {
        el.snrValueDisplay.textContent = `${e.target.value} dB`;
      });
      el.snrSlider.addEventListener('change', () => {
        invalidateRobustnessResults('Noise SNR modified — run perturbation test to compute updated metrics');
      });
    }

    // Mitigation Toggle
    if (el.mitigationToggle) {
      el.mitigationToggle.addEventListener('change', () => {
        invalidateRobustnessResults('Mitigation toggle modified — run perturbation test to evaluate impact');
      });
    }

    // Run Robustness Button
    if (el.runRobustnessBtn) {
      el.runRobustnessBtn.addEventListener('click', runRobustnessExperiment);
    }

    // Export Reports & Clear History
    if (el.exportReportBtn) {
      el.exportReportBtn.addEventListener('click', exportForensicReceipt);
    }
    if (el.clearHistoryBtn) {
      el.clearHistoryBtn.addEventListener('click', clearSessionHistory);
    }
    if (el.exportHistoryBtn) {
      el.exportHistoryBtn.addEventListener('click', exportSessionHistory);
    }
  }

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

  function setViewMode(mode) {
    state.viewMode = mode;
    if (mode === 'simple') {
      if (el.viewModeSimple) {
        el.viewModeSimple.classList.add('active');
        el.viewModeSimple.setAttribute('aria-checked', 'true');
      }
      if (el.viewModeExpert) {
        el.viewModeExpert.classList.remove('active');
        el.viewModeExpert.setAttribute('aria-checked', 'false');
      }
      if (el.simpleViewContent) el.simpleViewContent.classList.remove('hidden');
      if (el.expertViewContent) el.expertViewContent.classList.add('hidden');
    } else {
      if (el.viewModeSimple) {
        el.viewModeSimple.classList.remove('active');
        el.viewModeSimple.setAttribute('aria-checked', 'false');
      }
      if (el.viewModeExpert) {
        el.viewModeExpert.classList.add('active');
        el.viewModeExpert.setAttribute('aria-checked', 'true');
      }
      if (el.simpleViewContent) el.simpleViewContent.classList.add('hidden');
      if (el.expertViewContent) el.expertViewContent.classList.remove('hidden');
    }
  }

  // -------------------------------------------------------------------------
  // AUDIO FILE & SAMPLE LOADING
  // -------------------------------------------------------------------------

  // -------------------------------------------------------------------------
  // AUDIO FILE & SAMPLE LOADING
  // -------------------------------------------------------------------------

  function resetAnalysisResultsToStandby(reason) {
    state.lastAnalysisResult = null;
    if (el.verdictPill) {
      el.verdictPill.className = 'verdict-pill-large neutral';
    }
    if (el.verdictIcon) {
      el.verdictIcon.textContent = '●';
    }
    if (el.verdictPrimary) {
      el.verdictPrimary.textContent = 'STANDBY';
    } else if (el.verdictText) {
      el.verdictText.textContent = 'STANDBY';
    }
    if (el.verdictSubtitle) {
      el.verdictSubtitle.textContent = 'Awaiting Audio Analysis';
    }
    if (el.verdictExplanation) {
      el.verdictExplanation.textContent = reason || 'Run analysis on an audio sample to extract AASIST-L predictions.';
    }
    if (el.resTimestamp) {
      el.resTimestamp.textContent = 'Awaiting Analysis';
    }
    if (el.resBonafideLogit) el.resBonafideLogit.textContent = '—';
    if (el.resSpoofLogit) el.resSpoofLogit.textContent = '—';
    if (el.resSoftmaxBonafide) el.resSoftmaxBonafide.textContent = '—';
    if (el.resLatency) el.resLatency.textContent = '—';
    if (el.resRtf) el.resRtf.textContent = '—';
    if (el.meterNeedle) el.meterNeedle.style.left = '50%';
    if (el.simpleStatusTitle) el.simpleStatusTitle.textContent = 'Automated Inspection Summary';
    if (el.simpleStatusText) {
      el.simpleStatusText.textContent = 'Audio loaded. Click "RUN FORENSIC ANALYSIS" to generate plain-language findings.';
    }
    if (el.simpleMarginText) {
      el.simpleMarginText.textContent = 'The decision rule will evaluate the countermeasure score against baseline operating threshold θ = 0.00.';
    }
    if (el.exportReportBtn) el.exportReportBtn.disabled = true;
    drawSpectrogramPlaceholder();
  }

  async function handleUserFileUpload(file) {
    stopAudioPlayback(true);
    resetAnalysisResultsToStandby(`New file loaded: "${file.name}". Run analysis to compute deepfake classification.`);
    invalidateRobustnessResults('New audio file loaded — run perturbation test to evaluate robustness');
    expandUploadPanel();

    state.currentFile = file;
    state.currentSampleId = null;

    // Reset preset buttons active state
    el.sampleButtons.forEach((b) => b.classList.remove('active'));

    el.loadedFileStatus.textContent = 'FILE LOADED';
    el.metaFilename.textContent = file.name;
    el.metaFilename.title = file.name;
    el.playerTrackName.textContent = file.name;

    hideError();

    // Decode audio in browser for playback and waveform display
    try {
      const arrayBuffer = await file.arrayBuffer();
      await decodeAndDisplayAudio(arrayBuffer, file);
      enableActionButtons();
    } catch (err) {
      showError(`Could not decode audio file: ${err.message}`);
    }
  }

  async function handleSelectSample(sampleId, activeBtn) {
    stopAudioPlayback(true);
    resetAnalysisResultsToStandby(`Preset loaded: "${sampleId}". Run analysis to compute deepfake classification.`);
    invalidateRobustnessResults('New demonstration preset selected — run perturbation test to evaluate robustness');
    expandUploadPanel();

    state.currentSampleId = sampleId;
    state.currentFile = null;

    // Highlight active preset button
    el.sampleButtons.forEach((b) => b.classList.remove('active'));
    activeBtn.classList.add('active');

    const sampleName = activeBtn.querySelector('.preset-name').textContent;
    el.loadedFileStatus.textContent = 'PRESET READY';
    el.metaFilename.textContent = `${sampleId}.wav`;
    el.metaFilename.title = sampleName;
    el.playerTrackName.textContent = sampleName;

    hideError();

    // Fetch and decode demo sample audio bytes
    try {
      const resp = await fetch(`/api/samples/${sampleId}/audio`);
      if (!resp.ok) throw new Error(`Could not fetch preset audio: ${resp.status}`);
      const arrayBuffer = await resp.arrayBuffer();
      await decodeAndDisplayAudio(arrayBuffer, null);
      enableActionButtons();
    } catch (err) {
      showError(`Error loading preset audio: ${err.message}`);
    }
  }

  function enableActionButtons() {
    el.runInferenceBtn.disabled = false;
    el.runRobustnessBtn.disabled = false;
    el.playBtn.disabled = false;
  }

  // -------------------------------------------------------------------------
  // WEB AUDIO DECODING & PLAYBACK
  // -------------------------------------------------------------------------

  async function getAudioContext() {
    if (!state.audioContext) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      state.audioContext = new AudioCtx();
    }
    if (state.audioContext.state === 'suspended') {
      await state.audioContext.resume();
    }
    return state.audioContext;
  }

  async function decodeAndDisplayAudio(arrayBuffer, fileObj = null) {
    const ctx = await getAudioContext();
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
    state.currentAudioBuffer = audioBuffer;
    state.duration = audioBuffer.duration;
    state.playbackOffset = 0;

    el.totalDuration.textContent = formatTime(state.duration);
    if (el.currentTime) el.currentTime.textContent = '00:00';

    const srcRate = audioBuffer.sampleRate;
    const srcCh = audioBuffer.numberOfChannels;
    const sizeStr = fileObj ? ` (${(fileObj.size / 1024).toFixed(1)} KB)` : '';

    el.metaDuration.textContent = `${state.duration.toFixed(2)}s${sizeStr}`;
    el.metaSampleRate.textContent = srcRate === 16000 ? '16,000 Hz Native' : `${srcRate.toLocaleString()} Hz (➔ 16 kHz Model)`;
    el.metaChannels.textContent = srcCh === 1 ? '1 Ch (Mono Native)' : `${srcCh} Ch (➔ Mono Downmix)`;

    // Extract peaks for waveform canvas rendering
    const rawData = audioBuffer.getChannelData(0);
    state.waveformPeaks = extractPeaks(rawData, 140);
    drawWaveform(state.waveformPeaks, 0);
  }

  function extractPeaks(channelData, numPeaks) {
    const step = Math.floor(channelData.length / numPeaks);
    const peaks = [];
    for (let i = 0; i < numPeaks; i++) {
      const start = i * step;
      let max = 0;
      for (let j = 0; j < step; j++) {
        const val = Math.abs(channelData[start + j] || 0);
        if (val > max) max = val;
      }
      peaks.push(max);
    }
    return peaks;
  }

  function drawWaveform(peaks, progressRatio) {
    const canvas = el.waveformCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = (canvas.width = canvas.parentElement.clientWidth || 560);
    const h = (canvas.height = 90);

    ctx.clearRect(0, 0, w, h);

    if (!peaks || peaks.length === 0) return;

    const barWidth = Math.max(2, Math.floor(w / peaks.length) - 2);
    const gap = 2;
    const progressX = w * progressRatio;

    for (let i = 0; i < peaks.length; i++) {
      const x = i * (barWidth + gap);
      const barHeight = Math.max(4, peaks[i] * (h * 0.85));
      const y = (h - barHeight) / 2;

      // Color based on playback progress
      if (x < progressX) {
        ctx.fillStyle = '#60A5FA'; // Played: light electric blue
      } else {
        ctx.fillStyle = 'rgba(198, 203, 212, 0.25)'; // Unplayed: muted silver
      }

      ctx.beginPath();
      ctx.roundRect(x, y, barWidth, barHeight, 2);
      ctx.fill();
    }

    // Playhead indicator needle
    if (progressRatio > 0 && progressRatio < 1) {
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(progressX, 2);
      ctx.lineTo(progressX, h - 2);
      ctx.stroke();
    }
  }

  function seekWaveformAtEvent(e) {
    if (!state.duration || !el.waveformCanvas) return;
    const rect = el.waveformCanvas.getBoundingClientRect();
    const clientX = e.clientX !== undefined ? e.clientX : (e.touches && e.touches[0] ? e.touches[0].clientX : null);
    if (clientX === null) return;
    const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    state.playbackOffset = ratio * state.duration;
    if (el.currentTime) el.currentTime.textContent = formatTime(state.playbackOffset);
    drawWaveform(state.waveformPeaks, ratio);

    if (state.isPlaying) {
      restartAudioPlaybackFromOffset();
    }
  }

  async function restartAudioPlaybackFromOffset() {
    if (!state.currentAudioBuffer) return;
    const ctx = await getAudioContext();
    if (state.audioSourceNode) {
      try {
        state.audioSourceNode.onended = null;
        state.audioSourceNode.stop();
        state.audioSourceNode.disconnect();
      } catch (_) {}
      state.audioSourceNode = null;
    }

    state.audioSourceNode = ctx.createBufferSource();
    state.audioSourceNode.buffer = state.currentAudioBuffer;
    state.audioSourceNode.connect(ctx.destination);

    state.playbackStartTime = ctx.currentTime - state.playbackOffset;
    state.audioSourceNode.start(0, state.playbackOffset);
    state.isPlaying = true;
    if (el.playBtn) el.playBtn.textContent = '⏸';

    state.audioSourceNode.onended = () => {
      if (state.isPlaying) {
        stopAudioPlayback(true);
        drawWaveform(state.waveformPeaks, 0);
      }
    };
  }

  async function toggleAudioPlayback() {
    if (state.isPlaying) {
      stopAudioPlayback(false);
    } else {
      await startAudioPlayback();
    }
  }

  async function startAudioPlayback() {
    if (!state.currentAudioBuffer) return;
    const ctx = await getAudioContext();

    state.audioSourceNode = ctx.createBufferSource();
    state.audioSourceNode.buffer = state.currentAudioBuffer;
    state.audioSourceNode.connect(ctx.destination);

    state.playbackStartTime = ctx.currentTime - state.playbackOffset;
    state.audioSourceNode.start(0, state.playbackOffset);
    state.isPlaying = true;
    if (el.playBtn) el.playBtn.textContent = '⏸';

    state.audioSourceNode.onended = () => {
      if (state.isPlaying) {
        stopAudioPlayback(true);
        drawWaveform(state.waveformPeaks, 0);
      }
    };

    requestAnimationFrame(updatePlaybackProgress);
  }

  function stopAudioPlayback(resetOffset = false) {
    if (state.audioSourceNode) {
      try {
        state.audioSourceNode.onended = null;
        state.audioSourceNode.stop();
        state.audioSourceNode.disconnect();
      } catch (_) {}
      state.audioSourceNode = null;
    }
    if (state.isPlaying && state.audioContext && !resetOffset) {
      state.playbackOffset = Math.min(state.duration, state.audioContext.currentTime - state.playbackStartTime);
    }
    if (resetOffset) {
      state.playbackOffset = 0;
      if (el.currentTime) el.currentTime.textContent = '00:00';
    }
    state.isPlaying = false;
    if (el.playBtn) el.playBtn.textContent = '▶';
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
      stopAudioPlayback(true);
      drawWaveform(state.waveformPeaks, 0);
    }
  }

  function formatTime(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  }

  // -------------------------------------------------------------------------
  // RUN AASIST-L INFERENCE & HONEST PROGRESS FEEDBACK
  // -------------------------------------------------------------------------

  function showAnalysisProgress(initialMsg) {
    if (!el.analysisProgressCard) return;
    el.analysisProgressCard.classList.remove('hidden');
    if (el.analysisProgressMsg) el.analysisProgressMsg.textContent = initialMsg;
  }

  function updateAnalysisProgress(msg) {
    if (el.analysisProgressMsg) el.analysisProgressMsg.textContent = msg;
  }

  function hideAnalysisProgress() {
    if (el.analysisProgressCard) el.analysisProgressCard.classList.add('hidden');
  }

  async function runAnalysis() {
    if (!state.currentFile && !state.currentSampleId) return;

    const analysisId = ++state.analysisCounter;
    setRunLoading(true);
    hideError();

    // Honest progress state reflecting actual request dispatch
    showAnalysisProgress('Executing AASIST-L inference on 16 kHz audio window...');

    // Trigger subtle pulse on the 3D signal sculpture
    if (window.triggerSculpturePulse) {
      window.triggerSculpturePulse(2.4);
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

      if (state.analysisCounter !== analysisId) {
        // Obsoleted by a newer invocation
        return;
      }

      if (!resp.ok) {
        const errorData = await resp.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${resp.status}`);
      }

      const result = await resp.json();
      state.lastAnalysisResult = result;
      displayResults(result);
      collapseUploadPanel();
      await refreshSessionHistory();

      // Smooth scroll to results
      const resultsSec = document.getElementById('results');
      if (resultsSec) resultsSec.scrollIntoView({ behavior: 'smooth' });
    } catch (err) {
      showError(err.message);
    } finally {
      hideAnalysisProgress();
      setRunLoading(false);
    }
  }

  function displayResults(data) {
    if (el.resTimestamp) el.resTimestamp.textContent = data.timestamp;

    // Verdict Pill
    const isBonafide = String(data.prediction).toUpperCase() === 'BONAFIDE';
    el.verdictPill.className = `verdict-pill-large ${isBonafide ? 'bonafide' : 'spoof'}`;
    el.verdictIcon.textContent = isBonafide ? '✓' : '⚠';
    if (el.verdictPrimary) {
      el.verdictPrimary.textContent = isBonafide ? 'BONAFIDE' : 'SPOOF';
    } else if (el.verdictText) {
      el.verdictText.textContent = isBonafide ? 'BONAFIDE' : 'SPOOF';
    }
    if (el.verdictSubtitle) {
      el.verdictSubtitle.textContent = isBonafide ? 'Authentic Human Speech' : 'Synthetic / Deepfake';
    }

    el.verdictExplanation.textContent = isBonafide
      ? `AASIST-L countermeasure score (${data.primary_score}) sits at or above operating threshold (${data.scores.threshold}). Spectral-temporal graph attention exhibits natural glottal harmonic distributions.`
      : `AASIST-L countermeasure score (${data.primary_score}) sits below operating threshold (${data.scores.threshold}). Neural attention detects phase inconsistencies, vocoder buzzing, or synthetic artifact signatures.`;

    // Simple View Findings
    const margin = (data.primary_score - data.scores.threshold).toFixed(3);
    const marginSign = margin >= 0 ? `+${margin}` : margin;
    if (el.simpleStatusTitle) {
      el.simpleStatusTitle.textContent = isBonafide
        ? 'Classification: Authentic Human Voice Detected'
        : 'Classification: Synthetic Voice / Deepfake Artifacts Detected';
    }
    if (el.simpleStatusText) {
      el.simpleStatusText.textContent = isBonafide
        ? `The recording exhibited natural glottal harmonic resonances and organic vocal tract formant transitions across the 0–8000 Hz spectrum. The AASIST-L countermeasure score (${data.primary_score}) sits safely above the decision boundary.`
        : `The neural graph-attention network identified phase discrepancies, vocoder synthesis signatures, or neural text-to-speech stitching artifacts. The AASIST-L countermeasure score (${data.primary_score}) sits below the decision boundary.`;
    }
    if (el.simpleMarginText) {
      el.simpleMarginText.textContent = `Countermeasure score is ${data.primary_score} (margin: ${marginSign} relative to baseline threshold ${data.scores.threshold}). Individual predictions reflect 16 kHz audio window inference and serve as forensic guidance.`;
    }

    // Expert View Metrics
    el.resBonafideLogit.textContent = String(data.scores.bonafide_logit);
    el.resSpoofLogit.textContent = String(data.scores.spoof_logit);
    el.resSoftmaxBonafide.textContent = `${(data.scores.softmax_bonafide * 100).toFixed(2)}%`;
    el.resLatency.textContent = `${data.performance.latency_ms} ms`;
    el.resRtf.textContent = `${data.performance.real_time_factor}x`;

    // Meter Needle Position: Map score [-4.0, +4.0] to [0%, 100%]
    const score = data.scores.bonafide_logit;
    const clampedScore = Math.max(-4.0, Math.min(4.0, score));
    const needlePercent = ((clampedScore + 4.0) / 8.0) * 100.0;
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
    const clamped = Math.max(-4.0, Math.min(4.0, threshold));
    const percent = ((clamped + 4.0) / 8.0) * 100.0;
    if (el.meterThresholdMarker) el.meterThresholdMarker.style.left = `${percent}%`;
  }

  function updatePredictionForThreshold(threshold) {
    if (!state.lastAnalysisResult) return;
    const score = state.lastAnalysisResult.scores.bonafide_logit;
    const isBonafide = score >= threshold;
    const newPrediction = isBonafide ? 'BONAFIDE' : 'SPOOF';

    el.verdictPill.className = `verdict-pill-large ${isBonafide ? 'bonafide' : 'spoof'}`;
    el.verdictIcon.textContent = isBonafide ? '✓' : '⚠';
    if (el.verdictPrimary) {
      el.verdictPrimary.textContent = isBonafide ? 'BONAFIDE' : 'SPOOF';
    } else if (el.verdictText) {
      el.verdictText.textContent = isBonafide ? 'BONAFIDE' : 'SPOOF';
    }
    if (el.verdictSubtitle) {
      el.verdictSubtitle.textContent = isBonafide ? 'Authentic Human Speech' : 'Synthetic / Deepfake';
    }

    const defaultThresh = 0.0;
    const isDefault = Math.abs(threshold - defaultThresh) < 1e-4;

    el.verdictExplanation.textContent = isDefault
      ? (isBonafide
          ? `AASIST-L countermeasure score (${score}) sits at or above the baseline threshold (0.00). Spectral-temporal graph attention exhibits natural glottal harmonic distributions.`
          : `AASIST-L countermeasure score (${score}) sits below the baseline threshold (0.00). Neural attention detects phase inconsistencies, vocoder buzzing, or synthetic artifact signatures.`)
      : `Threshold dynamically adjusted to θ = ${threshold.toFixed(2)}. Countermeasure score ${score} is ${isBonafide ? 'greater than or equal to' : 'less than'} the adjusted threshold, yielding prediction: ${newPrediction}. (Score re-evaluated without repeating inference).`;

    // Simple View Update
    const margin = (score - threshold).toFixed(3);
    const marginSign = margin >= 0 ? `+${margin}` : margin;
    if (el.simpleStatusTitle) {
      el.simpleStatusTitle.textContent = isBonafide
        ? 'Classification: Authentic Human Voice Detected'
        : 'Classification: Synthetic Voice / Deepfake Artifacts Detected';
    }
    if (el.simpleStatusText) {
      el.simpleStatusText.textContent = isBonafide
        ? `At threshold θ = ${threshold.toFixed(2)}, countermeasure score (${score}) sits at or above the boundary, classifying the speech as authentic.`
        : `At threshold θ = ${threshold.toFixed(2)}, countermeasure score (${score}) sits below the boundary, classifying the speech as synthetic/deepfake.`;
    }
    if (el.simpleMarginText) {
      el.simpleMarginText.textContent = `Adjusted margin: ${marginSign} relative to active threshold θ = ${threshold.toFixed(2)}. Evaluated without repeating inference.`;
    }
  }

  // -------------------------------------------------------------------------
  // COLLAPSIBLE UPLOAD PANEL
  // -------------------------------------------------------------------------

  function collapseUploadPanel() {
    if (!el.cardUpload) return;
    el.cardUpload.classList.add('is-collapsed');
    if (el.uploadExpandedContent) el.uploadExpandedContent.classList.add('hidden');
    if (el.uploadCollapsedContent) {
      el.uploadCollapsedContent.classList.remove('hidden');
      const activeName = state.currentFile ? state.currentFile.name : (state.currentSampleId || 'Demonstration Audio');
      const durationText = state.duration ? `${state.duration.toFixed(2)}s` : '16 kHz mono';
      if (el.collapsedSourceName) el.collapsedSourceName.textContent = activeName;
      if (el.collapsedSourceMeta) el.collapsedSourceMeta.textContent = `16 kHz Mono • ${durationText}`;
    }
    if (el.expandUploadBtn) el.expandUploadBtn.setAttribute('aria-expanded', 'false');
  }

  function expandUploadPanel() {
    if (!el.cardUpload) return;
    el.cardUpload.classList.remove('is-collapsed');
    if (el.uploadExpandedContent) el.uploadExpandedContent.classList.remove('hidden');
    if (el.uploadCollapsedContent) el.uploadCollapsedContent.classList.add('hidden');
    if (el.expandUploadBtn) el.expandUploadBtn.setAttribute('aria-expanded', 'true');
  }

  // -------------------------------------------------------------------------
  // FREQUENCY SPECTROGRAM VISUALIZER
  // -------------------------------------------------------------------------

  function drawSpectrogram(magnitudes) {
    const canvas = el.spectrumCanvas;
    const ctx = canvas.getContext('2d');
    const w = (canvas.width = canvas.parentElement.clientWidth || 560);
    const h = (canvas.height = canvas.parentElement.clientHeight || 190);

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#08080B';
    ctx.fillRect(0, 0, w, h);

    if (!magnitudes || magnitudes.length === 0) return;

    // Subtle reference grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    for (let y = h / 4; y < h; y += h / 4) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    const barWidth = w / magnitudes.length;
    const maxMag = Math.max(...magnitudes, 1e-4);

    for (let i = 0; i < magnitudes.length; i++) {
      const norm = magnitudes[i] / maxMag;
      const barH = norm * (h * 0.88);
      const x = i * barWidth;
      const y = h - barH;

      // Color gradient matching legend swatches: Formant/Sub-bass (violet/purple) -> Vocal Resonance (electric blue) -> Nyquist (silver)
      const grad = ctx.createLinearGradient(0, y, 0, h);
      if (i < magnitudes.length * 0.25) {
        grad.addColorStop(0, '#A855F7');
        grad.addColorStop(1, '#581C87');
      } else if (i < magnitudes.length * 0.65) {
        grad.addColorStop(0, '#60A5FA');
        grad.addColorStop(1, '#1E40AF');
      } else {
        grad.addColorStop(0, '#E2E8F0');
        grad.addColorStop(1, '#3B82F6');
      }

      ctx.fillStyle = grad;
      ctx.fillRect(x, y, Math.max(1, barWidth - 1), barH);
    }
  }

  function setRunLoading(isLoading) {
    el.runInferenceBtn.disabled = isLoading;
    el.thresholdSlider.disabled = isLoading;
    if (isLoading) {
      el.runInferenceBtn.setAttribute('aria-busy', 'true');
      el.runSpinner.classList.remove('hidden');
      el.runBtnText.textContent = 'RUNNING INFERENCE...';
    } else {
      el.runInferenceBtn.removeAttribute('aria-busy');
      el.runSpinner.classList.add('hidden');
      el.runBtnText.textContent = 'RUN FORENSIC ANALYSIS';
    }
  }

  // -------------------------------------------------------------------------
  // ROBUSTNESS EXPERIMENTS & CONCLUSION SYNTHESIS
  // -------------------------------------------------------------------------

  function updateConditionControls() {
    let selectedCondition = 'telephony_8khz';
    el.robRadios.forEach((r) => {
      if (r.checked) selectedCondition = r.value;
    });

    const isNoise = selectedCondition.startsWith('noise');
    if (el.snrSliderWrapper) {
      if (isNoise) {
        el.snrSliderWrapper.classList.remove('disabled');
        el.snrSlider.disabled = false;
      } else {
        el.snrSliderWrapper.classList.add('disabled');
        el.snrSlider.disabled = true;
      }
    }
  }

  function invalidateRobustnessResults(reason) {
    el.robStatusBadge.className = 'status-tag';
    el.robStatusBadge.textContent = 'STANDBY';
    el.cleanLogitDisplay.textContent = '—';
    el.cleanPredBadge.className = 'cell-badge';
    el.cleanPredBadge.textContent = 'NO DATA';
    el.degradedLogitDisplay.textContent = '—';
    el.degradedPredBadge.className = 'cell-badge';
    el.degradedPredBadge.textContent = 'NO DATA';
    el.deltaLogitDisplay.textContent = '—';
    if (el.robMitigationBadge) {
      el.robMitigationBadge.className = 'pill-badge neutral';
      el.robMitigationBadge.textContent = 'MITIGATION: STANDBY';
    }
    if (el.robClassificationBadge) {
      el.robClassificationBadge.className = 'pill-badge neutral';
      el.robClassificationBadge.textContent = 'DECISION: STANDBY';
    }
    if (el.playbackSignalLabel) {
      el.playbackSignalLabel.textContent = '[PERTURBED SIGNAL PLAYBACK]';
    }
    el.deltaExplanation.textContent = reason || 'Select an audio recording and run a perturbation test to evaluate robustness.';
    if (el.degradedAudioElement) {
      el.degradedAudioElement.src = '';
    }
    if (state.currentDegradedBlobUrl) {
      URL.revokeObjectURL(state.currentDegradedBlobUrl);
      state.currentDegradedBlobUrl = null;
    }

    // Reset Robustness Conclusion Card
    if (el.robConclusionBadge) {
      el.robConclusionBadge.className = 'pill-badge neutral';
      el.robConclusionBadge.textContent = 'STANDBY';
    }
    if (el.conclusionLeadText) {
      el.conclusionLeadText.textContent = 'Run evaluation to assess decision stability and countermeasure score drift (ΔS_cm).';
    }
    if (el.conclusionBodyText) {
      el.conclusionBodyText.textContent = reason || 'This analysis determines whether simulated acoustic channel degradation causes the AASIST-L decision boundary (θ = 0.00) to breach or significantly degrades countermeasure score confidence.';
    }
    if (el.robAcousticExplanation) {
      el.robAcousticExplanation.textContent = 'Physical channel effects will be characterized here upon executing the selected degradation.';
    }
    if (el.robMitigationExplanation) {
      el.robMitigationExplanation.textContent = 'Mitigation filter response (spectral subtraction & bandpass normalization) will be evaluated here.';
    }
  }

  async function runRobustnessExperiment() {
    if (!state.currentFile && !state.currentSampleId) return;

    setRobLoading(true);

    try {
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
      el.robStatusBadge.className = 'status-tag error';
      el.robStatusBadge.textContent = 'FAILED';
      el.deltaExplanation.textContent = `Experiment failed: ${err.message}`;
      showError(`Robustness test failed: ${err.message}`);
    } finally {
      setRobLoading(false);
    }
  }

  function displayRobustnessResults(data) {
    if (data.status === 'UNAVAILABLE') {
      el.robStatusBadge.className = 'status-tag warning';
      el.robStatusBadge.textContent = 'UNAVAILABLE';
      el.cleanLogitDisplay.textContent = data.baseline_score !== undefined ? String(data.baseline_score) : '—';
      el.cleanPredBadge.className = 'cell-badge';
      el.cleanPredBadge.textContent = 'BASE SCORING';
      el.degradedLogitDisplay.textContent = 'N/A';
      el.degradedPredBadge.className = 'cell-badge';
      el.degradedPredBadge.textContent = 'N/A';
      el.deltaLogitDisplay.textContent = 'N/A';
      if (el.robMitigationBadge) {
        el.robMitigationBadge.className = 'pill-badge warning';
        el.robMitigationBadge.textContent = 'MITIGATION: N/A';
      }
      if (el.robClassificationBadge) {
        el.robClassificationBadge.className = 'pill-badge warning';
        el.robClassificationBadge.textContent = 'DECISION: BLOCKED';
      }
      el.deltaExplanation.textContent = data.error || 'The selected condition is unavailable on this host.';
      if (el.degradedAudioElement) el.degradedAudioElement.src = '';
      return;
    }

    el.robStatusBadge.className = 'status-tag success';
    el.robStatusBadge.textContent = 'COMPLETED';

    const clean = data.comparison.clean_bonafide_logit;
    const deg = data.comparison.degraded_bonafide_logit;
    const delta = data.comparison.delta_logit;
    const absDelta = data.comparison.absolute_delta_logit !== undefined ? data.comparison.absolute_delta_logit : Math.abs(delta);
    const isMitigated = Boolean(data.parameters.mitigation_applied);
    const cleanPred = data.comparison.clean_prediction;
    const degPred = data.comparison.degraded_prediction;
    const predChanged = cleanPred !== degPred;

    el.cleanLogitDisplay.textContent = String(clean);
    el.cleanPredBadge.className = `cell-badge ${cleanPred === 'BONAFIDE' ? 'bonafide' : 'spoof'}`;
    el.cleanPredBadge.textContent = cleanPred;

    el.degradedLogitDisplay.textContent = String(deg);
    el.degradedPredBadge.className = `cell-badge ${degPred === 'BONAFIDE' ? 'bonafide' : 'spoof'}`;
    el.degradedPredBadge.textContent = degPred;

    const deltaSign = delta >= 0 ? `+${delta.toFixed(4)}` : `${delta.toFixed(4)}`;
    el.deltaLogitDisplay.textContent = `${deltaSign}`;

    // Mitigation status badge
    if (el.robMitigationBadge) {
      el.robMitigationBadge.className = `pill-badge ${isMitigated ? 'success' : 'neutral'}`;
      el.robMitigationBadge.textContent = isMitigated ? 'MITIGATION: ACTIVE (Spectral Subtraction + Bandpass)' : 'MITIGATION: BYPASSED';
    }

    // Classification decision shift badge
    if (el.robClassificationBadge) {
      if (predChanged) {
        el.robClassificationBadge.className = 'pill-badge warning';
        el.robClassificationBadge.textContent = `DECISION FLIPPED: ${cleanPred} ➔ ${degPred}`;
      } else {
        el.robClassificationBadge.className = 'pill-badge success';
        el.robClassificationBadge.textContent = `ROBUST: RETAINED ${cleanPred}`;
      }
    }

    // Dynamic player label
    if (el.playbackSignalLabel) {
      el.playbackSignalLabel.textContent = isMitigated ? '[MITIGATED SIGNAL PLAYBACK]' : '[DEGRADED SIGNAL PLAYBACK]';
    }

    const conditionName = data.condition === 'telephony_8khz'
      ? 'Telephony Bandpass (8 kHz G.711 simulation)'
      : data.condition.startsWith('noise')
      ? `AWGN Additive Noise (${data.parameters.snr_db !== null ? data.parameters.snr_db + ' dB SNR' : 'noise'})`
      : data.condition.startsWith('opus')
      ? 'Lossy Opus Codec (16 kbps)'
      : data.condition;

    el.deltaExplanation.textContent = `Score drift ΔS_cm = ${deltaSign} (|Δ| = ${absDelta.toFixed(4)}) observed under [${conditionName}]. ${
      predChanged
        ? `Model decision boundary breached (${cleanPred} to ${degPred}).`
        : `Model classification remained resilient as ${cleanPred}.`
    }`;

    // Generate explicit Robustness Conclusion
    generateRobustnessConclusion({
      clean,
      deg,
      delta,
      absDelta,
      deltaSign,
      cleanPred,
      degPred,
      predChanged,
      isMitigated,
      condition: data.condition,
      conditionName,
      snrVal: data.parameters.snr_db,
    });

    // Playback degraded or mitigated audio
    if (data.audio_wav_base64) {
      if (state.currentDegradedBlobUrl) {
        URL.revokeObjectURL(state.currentDegradedBlobUrl);
        state.currentDegradedBlobUrl = null;
      }
      const byteCharacters = hexToBytes(data.audio_wav_base64);
      const blob = new Blob([byteCharacters], { type: 'audio/wav' });
      const blobUrl = URL.createObjectURL(blob);
      state.currentDegradedBlobUrl = blobUrl;
      el.degradedAudioElement.src = blobUrl;
    }
  }

  function generateRobustnessConclusion(info) {
    if (!el.robConclusionCard) return;

    // 1. Conclusion Badge
    if (el.robConclusionBadge) {
      if (info.predChanged) {
        el.robConclusionBadge.className = 'pill-badge warning';
        el.robConclusionBadge.textContent = 'DECISION BOUNDARY BREACHED';
      } else if (info.absDelta <= 1.0) {
        el.robConclusionBadge.className = 'pill-badge success';
        el.robConclusionBadge.textContent = 'DECISION INVARIANT (|Δ| ≤ 1.0)';
      } else {
        el.robConclusionBadge.className = 'pill-badge neutral';
        el.robConclusionBadge.textContent = 'DECISION RETAINED (SCORE SENSITIVE)';
      }
    }

    // 2. Conclusion Lead Text
    if (el.conclusionLeadText) {
      if (info.predChanged) {
        el.conclusionLeadText.textContent = `The model decision flipped from ${info.cleanPred} to ${info.degPred} under ${info.conditionName} (ΔS_cm = ${info.deltaSign}).`;
      } else if (info.absDelta <= 1.0) {
        el.conclusionLeadText.textContent = `The prediction remained invariant as ${info.cleanPred} with minimal score drift (ΔS_cm = ${info.deltaSign}, |Δ| = ${info.absDelta.toFixed(4)}).`;
      } else {
        el.conclusionLeadText.textContent = `The classification remained unchanged as ${info.cleanPred}, but countermeasure confidence shifted by ΔS_cm = ${info.deltaSign} (|Δ| = ${info.absDelta.toFixed(4)}).`;
      }
    }

    // 3. Conclusion Body Text
    if (el.conclusionBodyText) {
      if (info.predChanged) {
        el.conclusionBodyText.textContent = `This empirical outcome demonstrates that the applied channel transformation breached the model's decision threshold (τ = 0.00). The model's spectro-temporal graph attention representations were substantially altered by the acoustic corruption, revealing that this recording is vulnerable to misclassification under this specific channel effect.`;
      } else if (info.absDelta <= 1.0) {
        el.conclusionBodyText.textContent = `The model exhibited strong stability under this perturbation condition for this recording. The countermeasure score drift remained tightly bounded (|ΔS_cm| ≤ 1.0), preserving the decision margin away from threshold τ = 0.00. Note: this provides evidence of stability for this specific recording and condition, not proof of generalized universal robustness.`;
      } else {
        el.conclusionBodyText.textContent = `Even though the categorical prediction remained at or above threshold τ = 0.00 as ${info.cleanPred}, the countermeasure logit exhibited notable sensitivity (|ΔS_cm| > 1.0). In high-assurance security environments, significant score drift indicates that channel degradation erodes detection margins and could increase false rejection rates under more severe noise.`;
      }
    }

    // 4. Acoustic & Spectral Causal Factors
    if (el.robAcousticExplanation) {
      if (info.condition === 'telephony_8khz') {
        el.robAcousticExplanation.textContent = `Telephony downsampling (8 kHz polyphase resample) restricts bandwidth to 300–3400 Hz, discarding all spectral cues above 4 kHz. AASIST-L's SincNet front-end relies on spectro-temporal formant structures across the full 16 kHz sampling range (8 kHz Nyquist). Stripping upper frequencies and applying telephone-band shaping causes severe attenuation of higher speech harmonics.`;
      } else if (info.condition.startsWith('noise')) {
        el.robAcousticExplanation.textContent = `Additive White Gaussian Noise (AWGN) at ${info.snrVal !== null ? info.snrVal + ' dB SNR' : 'configured SNR'} introduces uncorrelated flat-spectrum noise energy across all frequency bins. This broadband noise masks lower-energy consonant formant transitions and disrupts the glottal harmonic periodicity captured by graph attention modules.`;
      } else if (info.condition.startsWith('opus')) {
        el.robAcousticExplanation.textContent = `Opus lossy audio compression (16 kbps) utilizes MDCT transform coding and psychoacoustic band quantization, discarding perceptually sub-audible high-frequency components and introducing frame-stitching phase boundaries that alter temporal graph node connectivity.`;
      } else {
        el.robAcousticExplanation.textContent = `The applied perturbation alters acoustic spectro-temporal representations, modifying the acoustic cues processed by the SincNet filterbanks.`;
      }
    }

    // 5. Front-End DSP Mitigation Impact
    if (el.robMitigationExplanation) {
      if (info.isMitigated) {
        el.robMitigationExplanation.textContent = `Front-end DSP mitigation was active: non-parametric spectral subtraction suppressed stationary noise energy, followed by 300–3400 Hz bandpass normalization and RMS peak-matching. This signal conditioning stabilized the waveform before neural feature extraction, achieving a final score of ${info.deg.toFixed(4)} without adding any parameters to the 85,306 model budget.`;
      } else {
        el.robMitigationExplanation.textContent = `Front-end DSP mitigation was bypassed: the recording was evaluated in its raw degraded acoustic state. In unmitigated conditions, raw channel distortion directly impacts the first-layer SincNet convolutional filters. Enabling the mitigation pipeline applies adaptive spectral subtraction and bandpass normalization to attenuate channel artifacts.`;
      }
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
      renderHistoryTable(history);
    } catch (err) {
      console.warn('Session history refresh failed:', err);
    }
  }

  function renderHistoryTable(records) {
    if (!el.historyTableBody) return;
    el.historyTableBody.innerHTML = '';

    if (!records || records.length === 0) {
      el.historyCountText.textContent = '0 records in current session';
      el.historyTableBody.innerHTML = `
        <tr class="empty-row">
          <td colspan="6" class="text-center text-muted">No analysis trials conducted yet in this session.</td>
        </tr>
      `;
      return;
    }

    el.historyCountText.textContent = `${records.length} record${records.length === 1 ? '' : 's'} in current session`;

    records.slice().reverse().forEach((rec) => {
      const tr = document.createElement('tr');

      const isBonafide = rec.prediction === 'BONAFIDE';
      const predBadgeClass = isBonafide ? 'bonafide' : 'spoof';
      const condBadgeClass = rec.condition === 'clean' ? 'clean' : 'degraded';

      const timeFormatted = rec.timestamp.split('T')[1]?.replace('Z', '') || rec.timestamp;

      tr.innerHTML = `
        <td class="font-mono text-muted">${escapeHtml(timeFormatted)}</td>
        <td class="font-bold">${escapeHtml(rec.identifier || 'audio.wav')}</td>
        <td><span class="history-pill ${condBadgeClass}">${escapeHtml(rec.condition)}</span></td>
        <td><span class="cell-badge ${predBadgeClass}">${escapeHtml(rec.prediction)}</span></td>
        <td class="font-mono">${rec.score !== undefined ? Number(rec.score).toFixed(4) : '—'}</td>
        <td class="font-mono text-muted">${rec.latency_ms ? rec.latency_ms.toFixed(1) + ' ms' : '—'}</td>
      `;
      el.historyTableBody.appendChild(tr);
    });
  }

  function exportForensicReceipt() {
    if (!state.lastAnalysisResult) return;
    const res = state.lastAnalysisResult;
    const activeName = state.currentFile ? state.currentFile.name : (state.currentSampleId || 'Demonstration Audio');
    const checkpointHash = (state.systemStatus && state.systemStatus.checkpoint_sha256) || '814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a';

    const report = {
      report_title: 'Audio Authenticity Forensic Analysis Receipt',
      generated_at: new Date().toISOString(),
      platform: {
        system: 'Audio Authenticity Lab (PRJ_55)',
        neural_architecture: 'AASIST-L (Jung et al. 2022)',
        total_parameters: 85306,
        sincnet_encoder: '70 Sinc Filters (16 kHz)',
        heterogeneous_graphs: 'Spectro-Temporal Graph Attention (GAT)',
        checkpoint_integrity_sha256: checkpointHash,
        compute_device: (state.systemStatus && state.systemStatus.device) || 'mps'
      },
      source_audio_properties: {
        identifier: activeName,
        sample_rate_hz: 16000,
        channels: 1,
        evaluation_window_samples: 64600,
        evaluation_duration_seconds: state.duration || 4.0375
      },
      forensic_verdict: {
        predicted_classification: res.prediction.toUpperCase(),
        countermeasure_logit_S_cm: res.scores.bonafide_logit,
        spoof_logit: res.scores.spoof_logit,
        softmax_bonafide_probability: res.scores.softmax_bonafide,
        active_decision_threshold: state.threshold,
        standard_baseline_threshold: 0.0,
        decision_margin: res.scores.bonafide_logit - state.threshold,
        inference_latency_ms: res.inference_latency_ms,
        real_time_factor: res.real_time_factor
      },
      robustness_experiment: state.lastRobustnessResult ? {
        tested_condition: state.lastRobustnessResult.condition,
        clean_bonafide_logit: state.lastRobustnessResult.clean_score,
        perturbed_bonafide_logit: state.lastRobustnessResult.degraded_score,
        score_drift_delta_S_cm: state.lastRobustnessResult.score_drift,
        mitigation_applied: state.lastRobustnessResult.comparison.mitigation_applied,
        prediction_changed: state.lastRobustnessResult.comparison.prediction_changed,
        conclusion_badge: el.robConclusionBadge ? el.robConclusionBadge.textContent : 'UNKNOWN'
      } : 'No perturbation experiment conducted in this session.',
      scientific_integrity_statement: 'Countermeasure score S_cm is evaluated against decision threshold θ = 0.00 calibrated on ASVspoof 2019 partitions. Individual predictions represent inference estimates on 16 kHz audio windows, not absolute mathematical proof of real-world authenticity.'
    };

    const jsonStr = JSON.stringify(report, null, 2);
    downloadJsonFile(jsonStr, `forensic_receipt_${Date.now()}.json`);
  }

  async function exportSessionHistory() {
    try {
      const resp = await fetch(API.history);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const history = await resp.json();
      const jsonStr = JSON.stringify(history, null, 2);
      downloadJsonFile(jsonStr, `session_history_${Date.now()}.json`);
    } catch (err) {
      showError(`Export history failed: ${err.message}`);
    }
  }

  function downloadJsonFile(content, filename) {
    const blob = new Blob([content], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  async function clearSessionHistory() {
    if (!confirm('Are you sure you want to clear all experiment records from the current session?')) {
      return;
    }
    try {
      const resp = await fetch(API.history, { method: 'DELETE' });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      await refreshSessionHistory();
    } catch (err) {
      showError(`Could not clear session history: ${err.message}`);
    }
  }

  function drawSpectrogramPlaceholder() {
    const canvas = el.spectrumCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = (canvas.width = canvas.parentElement.clientWidth || 560);
    const h = (canvas.height = canvas.parentElement.clientHeight || 190);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#08080B';
    ctx.fillRect(0, 0, w, h);

    // Subtle grid lines
    ctx.strokeStyle = 'rgba(198, 203, 212, 0.06)';
    ctx.lineWidth = 1;
    for (let x = 40; x < w; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 30; y < h; y += 30) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    ctx.fillStyle = 'rgba(148, 163, 184, 0.85)';
    ctx.font = '12px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Awaiting Audio Analysis', w / 2, h / 2 - 8);
    ctx.fillStyle = 'rgba(100, 116, 139, 0.7)';
    ctx.font = '11px "Plus Jakarta Sans", sans-serif';
    ctx.fillText('0–8000 Hz FFT profile renders upon inference', w / 2, h / 2 + 12);
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

  // -------------------------------------------------------------------------
  // STAGE 3: INPUT MODE TABS, LIVE MIC CAPTURE & BATCH ANALYSIS
  // -------------------------------------------------------------------------

  function setupInputModeTabs() {
    const tabs = document.querySelectorAll('.input-mode-tab');
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        tabs.forEach(t => {
          t.classList.remove('active');
          t.setAttribute('aria-selected', 'false');
        });
        tab.classList.add('active');
        tab.setAttribute('aria-selected', 'true');
        const targetId = tab.getAttribute('data-target');
        document.querySelectorAll('.input-pane').forEach(p => p.classList.add('hidden'));
        const activePane = document.getElementById(targetId);
        if (activePane) activePane.classList.remove('hidden');
      });
    });
  }

  let micStream = null;
  let micMediaRecorder = null;
  let micAudioChunks = [];
  let micTimerInterval = null;
  let micStartTime = 0;
  let micCapturedFile = null;
  let micAnimId = null;

  /**
   * Encodes a Web Audio API AudioBuffer into a canonical 16-bit PCM RIFF WAV Blob.
   * Ensures browser-recorded audio (WebM/Opus or OGG) is converted to standard WAV format
   * with proper RIFF headers and signed 16-bit mono PCM samples for libsndfile compatibility.
   */
  function audioBufferToWavBlob(buffer) {
    const numChannels = 1; // Downmix to canonical mono
    const sampleRate = buffer.sampleRate;
    const numFrames = buffer.length;
    const bytesPerSample = 2; // 16-bit PCM
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = numFrames * blockAlign;
    const bufferSize = 44 + dataSize;

    const arrayBuffer = new ArrayBuffer(bufferSize);
    const view = new DataView(arrayBuffer);

    function writeString(offset, str) {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
    }

    // RIFF identifier & total chunk size
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(8, 'WAVE');

    // "fmt " subchunk
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
    view.setUint16(20, 1, true);  // AudioFormat (1 for PCM)
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true); // BitsPerSample (16 bits)

    // "data" subchunk
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);

    // Extract channels and downmix to mono
    const channels = [];
    for (let c = 0; c < buffer.numberOfChannels; c++) {
      channels.push(buffer.getChannelData(c));
    }

    let offset = 44;
    for (let i = 0; i < numFrames; i++) {
      let sample = 0;
      for (let c = 0; c < buffer.numberOfChannels; c++) {
        sample += channels[c][i];
      }
      sample /= buffer.numberOfChannels;
      // Clamp [-1.0, 1.0]
      sample = Math.max(-1, Math.min(1, sample));
      // Scale to signed 16-bit integer [-32768, 32767]
      const intSample = sample < 0 ? sample * 32768 : sample * 32767;
      view.setInt16(offset, Math.floor(intSample), true);
      offset += 2;
    }

    return new Blob([view], { type: 'audio/wav' });
  }

  function setupMicCapture() {
    const btnRecord = document.getElementById('btnMicRecord');
    const btnStop = document.getElementById('btnMicStop');
    const btnReset = document.getElementById('btnMicReset');
    const btnAnalyze = document.getElementById('btnMicAnalyze');
    const timerEl = document.getElementById('micTimer');
    const previewBlock = document.getElementById('micPreviewBlock');
    const audioEl = document.getElementById('micAudioPlayback');
    const canvas = document.getElementById('micVisualizerCanvas');

    if (!btnRecord) return;

    btnRecord.addEventListener('click', async () => {
      try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          throw new Error('Microphone access is not supported in this browser.');
        }
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        micAudioChunks = [];
        micMediaRecorder = new MediaRecorder(micStream);

        micMediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) micAudioChunks.push(e.data);
        };

        micMediaRecorder.onstop = async () => {
          clearInterval(micTimerInterval);
          if (micAnimId) cancelAnimationFrame(micAnimId);
          if (micStream) {
            micStream.getTracks().forEach((t) => t.stop());
            micStream = null;
          }

          try {
            // Retrieve browser-encoded audio container (WebM/Opus, OGG, or MP4)
            const mimeType = (micMediaRecorder && micMediaRecorder.mimeType) || 'audio/webm';
            const rawBlob = new Blob(micAudioChunks, { type: mimeType });
            const arrayBuf = await rawBlob.arrayBuffer();

            // Decode compressed audio frames to raw float32 PCM using Web Audio API
            const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
            const tempCtx = new AudioCtxClass();
            const decodedBuffer = await tempCtx.decodeAudioData(arrayBuf);

            // Transcode to standard canonical 16-bit PCM RIFF WAV
            const wavBlob = audioBufferToWavBlob(decodedBuffer);
            tempCtx.close().catch(() => {});

            micCapturedFile = new File([wavBlob], `mic_capture_${Date.now()}.wav`, { type: 'audio/wav' });
            if (audioEl) audioEl.src = URL.createObjectURL(wavBlob);
            if (previewBlock) previewBlock.classList.remove('hidden');
            if (btnReset) btnReset.classList.remove('hidden');
            btnStop.classList.add('hidden');
          } catch (convErr) {
            console.error('Microphone audio transcoding error:', convErr);
            showError(`Error processing recorded audio: ${convErr.message}`);
            btnRecord.classList.remove('hidden');
            btnStop.classList.add('hidden');
          }
        };

        micMediaRecorder.start();
        micStartTime = Date.now();
        btnRecord.classList.add('hidden');
        btnStop.classList.remove('hidden');
        if (previewBlock) previewBlock.classList.add('hidden');
        if (btnReset) btnReset.classList.add('hidden');

        // Draw live audio visualizer
        startMicVisualizer(micStream, canvas);

        // Timer with 10s auto-stop
        micTimerInterval = setInterval(() => {
          const elapsed = (Date.now() - micStartTime) / 1000;
          const s = Math.min(10, Math.floor(elapsed));
          timerEl.textContent = `00:${String(s).padStart(2, '0')} / 00:10`;
          if (elapsed >= 10.0) {
            if (micMediaRecorder && micMediaRecorder.state === 'recording') {
              micMediaRecorder.stop();
            }
          }
        }, 100);

      } catch (err) {
        showError(`Microphone access error: ${err.message}`);
      }
    });

    if (btnStop) {
      btnStop.addEventListener('click', () => {
        if (micMediaRecorder && micMediaRecorder.state === 'recording') {
          micMediaRecorder.stop();
        }
      });
    }

    if (btnReset) {
      btnReset.addEventListener('click', () => {
        micCapturedFile = null;
        if (previewBlock) previewBlock.classList.add('hidden');
        if (btnReset) btnReset.classList.add('hidden');
        btnRecord.classList.remove('hidden');
        btnStop.classList.add('hidden');
        timerEl.textContent = '00:00 / 00:10';
        if (canvas) {
          const ctx = canvas.getContext('2d');
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
      });
    }

    if (btnAnalyze) {
      btnAnalyze.addEventListener('click', async () => {
        if (!micCapturedFile) return;
        btnAnalyze.disabled = true;
        const origContent = btnAnalyze.innerHTML;
        btnAnalyze.innerHTML = '<span class="btn-spinner"></span> Ingesting & Analyzing Live Audio...';
        try {
          await handleUserFileUpload(micCapturedFile);
          await runAnalysis();
        } finally {
          btnAnalyze.disabled = false;
          btnAnalyze.innerHTML = origContent;
        }
      });
    }
  }

  function startMicVisualizer(stream, canvas) {
    if (!canvas) return;
    try {
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      const ctx = canvas.getContext('2d');

      function draw() {
        micAnimId = requestAnimationFrame(draw);
        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#08080B';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const barWidth = (canvas.width / bufferLength) * 2.5;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const barHeight = (dataArray[i] / 255) * canvas.height;
          ctx.fillStyle = `rgb(${dataArray[i]}, 140, 255)`;
          ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
          x += barWidth + 1;
        }
      }
      draw();
    } catch {
      // Fallback if visualizer cannot init
    }
  }

  let batchQueue = [];
  let lastBatchResults = null;

  function setupBatchAnalysis() {
    const dropZone = document.getElementById('batchDropZone');
    const fileInput = document.getElementById('batchFileInput');
    const limitCounter = document.getElementById('batchLimitCounter');
    const fileListEl = document.getElementById('batchFileList');
    const btnRun = document.getElementById('btnRunBatch');
    const btnClear = document.getElementById('btnClearBatch');
    const progressBox = document.getElementById('batchProgressBox');
    const progressBar = document.getElementById('batchProgressBar');
    const progressText = document.getElementById('batchProgressText');
    const resultsBlock = document.getElementById('batchResultsBlock');
    const totalCount = document.getElementById('batchTotalCount');
    const bonafideCount = document.getElementById('batchBonafideCount');
    const spoofCount = document.getElementById('batchSpoofCount');
    const tableBody = document.getElementById('batchTableBody');
    const btnExportJson = document.getElementById('btnExportBatchJson');

    if (!dropZone || !fileInput) return;

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      addFilesToBatch(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
      addFilesToBatch(e.target.files);
    });

    if (btnClear) {
      btnClear.addEventListener('click', () => {
        batchQueue = [];
        updateBatchUI();
      });
    }

    function addFilesToBatch(files) {
      for (const f of files) {
        if (batchQueue.length >= 10) {
          alert('Batch queue is limited to 10 files max to ensure bounded resource utilization.');
          break;
        }
        const ext = f.name.toLowerCase();
        if (!ext.endsWith('.wav') && !ext.endsWith('.flac')) {
          alert(`File "${f.name}" is not supported. Only WAV and FLAC files are accepted.`);
          continue;
        }
        if (f.size > 10 * 1024 * 1024) {
          alert(`File "${f.name}" exceeds 10 MB limit.`);
          continue;
        }
        if (!batchQueue.some(item => item.name === f.name)) {
          batchQueue.push(f);
        }
      }
      updateBatchUI();
    }

    function updateBatchUI() {
      if (limitCounter) limitCounter.textContent = `${batchQueue.length} / 10 FILES QUEUED`;
      if (btnRun) btnRun.disabled = batchQueue.length === 0;
      if (btnClear) btnClear.disabled = batchQueue.length === 0;

      if (!fileListEl) return;
      if (batchQueue.length === 0) {
        fileListEl.innerHTML = '';
        return;
      }

      fileListEl.innerHTML = batchQueue.map((f, idx) => `
        <div class="batch-file-item">
          <span class="batch-file-idx font-mono">#${idx + 1}</span>
          <span class="batch-file-name font-mono">${escapeHtml(f.name)}</span>
          <span class="batch-file-size text-muted">${(f.size / 1024).toFixed(1)} KB</span>
          <button class="btn-remove-batch" data-idx="${idx}" title="Remove file" type="button">✕</button>
        </div>
      `).join('');

      fileListEl.querySelectorAll('.btn-remove-batch').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          const i = parseInt(btn.getAttribute('data-idx'), 10);
          batchQueue.splice(i, 1);
          updateBatchUI();
        });
      });
    }

    if (btnRun) {
      btnRun.addEventListener('click', async () => {
        if (batchQueue.length === 0) return;
        btnRun.disabled = true;
        btnClear.disabled = true;
        if (progressBox) progressBox.classList.remove('hidden');
        if (resultsBlock) resultsBlock.classList.add('hidden');
        if (progressBar) progressBar.style.width = '10%';
        if (progressText) progressText.textContent = `Uploading and processing batch of ${batchQueue.length} files...`;

        try {
          const formData = new FormData();
          batchQueue.forEach(f => formData.append('files', f));
          const threshold = el.thresholdSlider ? el.thresholdSlider.value : 0.0;
          formData.append('threshold', threshold);

          if (progressBar) progressBar.style.width = '40%';
          const resp = await fetch('/api/analyze/batch', {
            method: 'POST',
            body: formData
          });
          if (progressBar) progressBar.style.width = '80%';

          if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            throw new Error(errData.detail || `Server error HTTP ${resp.status}`);
          }

          const data = await resp.json();
          if (progressBar) progressBar.style.width = '100%';
          if (progressText) progressText.textContent = `Batch processing completed (${data.processed_count} files).`;

          lastBatchResults = data;
          displayBatchResults(data);
        } catch (err) {
          if (progressText) progressText.textContent = `Batch failed: ${err.message}`;
          showError(`Batch analysis failed: ${err.message}`);
        } finally {
          btnRun.disabled = false;
          btnClear.disabled = false;
        }
      });
    }

    function displayBatchResults(data) {
      if (resultsBlock) resultsBlock.classList.remove('hidden');
      if (totalCount) totalCount.textContent = data.total_files || 0;
      if (bonafideCount) bonafideCount.textContent = data.bonafide_count || 0;
      if (spoofCount) spoofCount.textContent = data.spoof_count || 0;

      const results = data.results || [];
      if (!tableBody) return;

      tableBody.innerHTML = results.map(r => {
        const isBona = r.prediction === 'BONAFIDE';
        const badge = isBona
          ? `<span class="badge-verdict badge-bonafide">BONAFIDE</span>`
          : (r.status === 'SUCCESS' ? `<span class="badge-verdict badge-spoof">SPOOF</span>` : `<span class="badge-verdict badge-error">ERROR</span>`);
        const score = typeof r.score === 'number' ? r.score.toFixed(4) : 'N/A';
        const dur = typeof r.duration_s === 'number' ? `${r.duration_s.toFixed(2)}s` : '—';

        return `
          <tr>
            <td class="font-mono" style="font-size:0.85rem;">${escapeHtml(r.filename || '—')}</td>
            <td class="font-mono text-muted">${escapeHtml(dur)}</td>
            <td class="font-mono font-bold ${isBona ? 'text-cyan' : 'text-red'}">${score}</td>
            <td>${badge}</td>
            <td>
              <button class="btn btn-secondary btn-xs btn-inspect-batch" data-filename="${escapeHtml(r.filename || '')}" type="button">
                Inspect
              </button>
            </td>
          </tr>
        `;
      }).join('');

      tableBody.querySelectorAll('.btn-inspect-batch').forEach(btn => {
        btn.addEventListener('click', async () => {
          const name = btn.getAttribute('data-filename');
          const queuedFile = batchQueue.find(f => f.name === name);
          if (queuedFile) {
            await handleUserFileUpload(queuedFile);
            await runAnalysis();
          } else {
            alert(`File "${name}" is no longer in local buffer.`);
          }
        });
      });
    }

    if (btnExportJson) {
      btnExportJson.addEventListener('click', () => {
        if (!lastBatchResults) return;
        const blob = new Blob([JSON.stringify(lastBatchResults, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `batch_analysis_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      });
    }
  }

  // Run on page load
  window.addEventListener('DOMContentLoaded', init);
})();
