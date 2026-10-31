/**
 * Audio Authenticity Lab — Session History & Case Dossier Controller (Stage 3)
 * Manages session records, filtering, exports, and forensic case dossiers.
 */

(function () {
  'use strict';

  // State
  let rawRecords = [];
  let filteredRecords = [];
  let activeDossier = null;
  let recordToDelete = null;

  // DOM Elements
  const tableBody = document.getElementById('historyTableBody');
  const searchInput = document.getElementById('historySearchInput');
  const clearSearchBtn = document.getElementById('clearSearchBtn');
  const verdictFilter = document.getElementById('historyVerdictFilter');
  const sortSelect = document.getElementById('historySortSelect');
  const paginationInfo = document.getElementById('paginationInfo');

  const statTotal = document.getElementById('statTotalCount');
  const statBonafide = document.getElementById('statBonafideCount');
  const statSpoof = document.getElementById('statSpoofCount');
  const statAvgScore = document.getElementById('statAvgScore');

  const exportJsonBtn = document.getElementById('exportJsonBtn');
  const exportCsvBtn = document.getElementById('exportCsvBtn');
  const clearHistoryBtn = document.getElementById('clearHistoryBtn');

  // Dossier Modal
  const dossierModal = document.getElementById('dossierModal');
  const dossierBackdrop = document.getElementById('dossierModalBackdrop');
  const dossierContent = document.getElementById('dossierModalContent');
  const closeDossierBtn = document.getElementById('closeDossierBtn');
  const printDossierBtn = document.getElementById('printDossierBtn');
  const downloadDossierJsonBtn = document.getElementById('downloadDossierJsonBtn');
  const shareDossierBtn = document.getElementById('shareDossierBtn');
  const shareBox = document.getElementById('shareBox');
  const shareUrlInput = document.getElementById('shareUrlInput');
  const copyShareBtn = document.getElementById('copyShareBtn');
  const revokeShareBtn = document.getElementById('revokeShareBtn');
  const shareStatusMsg = document.getElementById('shareStatusMsg');
  let currentShareToken = null;

  // Confirm Clear Modal
  const confirmClearModal = document.getElementById('confirmClearModal');
  const confirmClearBackdrop = document.getElementById('confirmClearBackdrop');
  const cancelClearBtn = document.getElementById('cancelClearBtn');
  const closeClearModalBtn = document.getElementById('closeClearModalBtn');
  const executeClearBtn = document.getElementById('executeClearBtn');

  // Header diagnostics
  const systemStatusPill = document.getElementById('systemStatusPill');
  const systemStatusText = document.getElementById('systemStatusText');
  const diagDrawer = document.getElementById('diagDrawer');
  const diagBackdrop = document.getElementById('diagBackdrop');
  const closeDiagBtn = document.getElementById('closeDiagBtn');
  const diagDevice = document.getElementById('diagDevice');
  const diagSha256 = document.getElementById('diagSha256');
  const diagFfmpeg = document.getElementById('diagFfmpeg');

  // Mobile nav
  const mobileMenuBtn = document.getElementById('mobileMenuBtn');
  const mobileNavDrawer = document.getElementById('mobileNavDrawer');
  const mobileNavBackdrop = document.getElementById('mobileNavBackdrop');
  const closeMobileNavBtn = document.getElementById('closeMobileNavBtn');

  // -------------------------------------------------------------
  // Initialisation
  // -------------------------------------------------------------
  async function init() {
    setupEventListeners();
    await fetchSystemStatus();
    await loadHistory();
  }

  function setupEventListeners() {
    // Search & Filter
    searchInput.addEventListener('input', () => {
      clearSearchBtn.classList.toggle('hidden', !searchInput.value);
      applyFilters();
    });
    clearSearchBtn.addEventListener('click', () => {
      searchInput.value = '';
      clearSearchBtn.classList.add('hidden');
      applyFilters();
    });
    verdictFilter.addEventListener('change', applyFilters);
    sortSelect.addEventListener('change', applyFilters);

    // Export & Clear
    exportJsonBtn.addEventListener('click', exportFilteredJson);
    exportCsvBtn.addEventListener('click', exportFilteredCsv);
    clearHistoryBtn.addEventListener('click', () => showClearModal(true));

    cancelClearBtn.addEventListener('click', () => showClearModal(false));
    closeClearModalBtn.addEventListener('click', () => showClearModal(false));
    confirmClearBackdrop.addEventListener('click', () => showClearModal(false));
    executeClearBtn.addEventListener('click', executeClearHistory);

    // Dossier Modal
    closeDossierBtn.addEventListener('click', closeDossier);
    dossierBackdrop.addEventListener('click', closeDossier);
    printDossierBtn.addEventListener('click', () => window.print());
    downloadDossierJsonBtn.addEventListener('click', downloadActiveDossierJson);
    if (shareDossierBtn) shareDossierBtn.addEventListener('click', handleCreateShare);
    if (copyShareBtn) copyShareBtn.addEventListener('click', handleCopyShare);
    if (revokeShareBtn) revokeShareBtn.addEventListener('click', handleRevokeShare);

    // Diagnostics Drawer
    if (systemStatusPill) {
      systemStatusPill.addEventListener('click', () => toggleDrawer(diagDrawer, diagBackdrop, true));
    }
    if (closeDiagBtn) {
      closeDiagBtn.addEventListener('click', () => toggleDrawer(diagDrawer, diagBackdrop, false));
    }
    if (diagBackdrop) {
      diagBackdrop.addEventListener('click', () => toggleDrawer(diagDrawer, diagBackdrop, false));
    }

    // Mobile Nav
    if (mobileMenuBtn) {
      mobileMenuBtn.addEventListener('click', () => toggleDrawer(mobileNavDrawer, mobileNavBackdrop, true));
    }
    if (closeMobileNavBtn) {
      closeMobileNavBtn.addEventListener('click', () => toggleDrawer(mobileNavDrawer, mobileNavBackdrop, false));
    }
    if (mobileNavBackdrop) {
      mobileNavBackdrop.addEventListener('click', () => toggleDrawer(mobileNavDrawer, mobileNavBackdrop, false));
    }

    // Global ESC key
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        closeDossier();
        showClearModal(false);
        toggleDrawer(diagDrawer, diagBackdrop, false);
        toggleDrawer(mobileNavDrawer, mobileNavBackdrop, false);
      }
    });
  }

  function toggleDrawer(drawer, backdrop, show) {
    if (!drawer || !backdrop) return;
    drawer.classList.toggle('hidden', !show);
    backdrop.classList.toggle('hidden', !show);
  }

  // -------------------------------------------------------------
  // System Status
  // -------------------------------------------------------------
  async function fetchSystemStatus() {
    try {
      const res = await fetch('/api/status');
      if (!res.ok) throw new Error('Status failed');
      const data = await res.json();
      if (systemStatusText) systemStatusText.textContent = 'ONLINE (LOCAL)';
      if (diagDevice && data.model) diagDevice.textContent = data.model.device || 'CPU';
      if (diagSha256 && data.model) {
        diagSha256.textContent = (data.model.checkpoint_sha256 || 'N/A').substring(0, 16) + '...';
        diagSha256.title = data.model.checkpoint_sha256 || '';
      }
      if (diagFfmpeg && data.system) {
        diagFfmpeg.textContent = data.system.ffmpeg_available ? 'Available' : 'Unavailable';
      }
    } catch (err) {
      if (systemStatusText) systemStatusText.textContent = 'OFFLINE';
    }
  }

  // -------------------------------------------------------------
  // Load & Filter History
  // -------------------------------------------------------------
  async function loadHistory() {
    try {
      const res = await fetch('/api/history');
      if (!res.ok) throw new Error('Failed to load history');
      const data = await res.json();
      rawRecords = data.history || [];
      applyFilters();
    } catch (err) {
      tableBody.innerHTML = `
        <tr>
          <td colspan="8" class="text-red text-center" style="padding: 2.5rem;">
            Failed to retrieve session analysis records: ${escapeHtml(err.message)}
          </td>
        </tr>`;
    }
  }

  function applyFilters() {
    const q = searchInput.value.trim().toLowerCase();
    const verdict = verdictFilter.value;
    const sort = sortSelect.value;

    filteredRecords = rawRecords.filter(r => {
      // Verdict filter
      if (verdict !== 'ALL' && r.prediction !== verdict) return false;
      // Search filter
      if (q) {
        const idMatch = (r.record_id || '').toLowerCase().includes(q);
        const fileMatch = (r.filename || '').toLowerCase().includes(q);
        const hashMatch = (r.sha256 || '').toLowerCase().includes(q);
        if (!idMatch && !fileMatch && !hashMatch) return false;
      }
      return true;
    });

    // Sorting
    filteredRecords.sort((a, b) => {
      if (sort === 'newest') {
        return new Date(b.timestamp || 0).getTime() - new Date(a.timestamp || 0).getTime();
      } else if (sort === 'oldest') {
        return new Date(a.timestamp || 0).getTime() - new Date(b.timestamp || 0).getTime();
      } else if (sort === 'score_desc') {
        return (b.score ?? -999) - (a.score ?? -999);
      } else if (sort === 'score_asc') {
        return (a.score ?? 999) - (b.score ?? 999);
      }
      return 0;
    });

    updateSummaryStats();
    renderTable();
  }

  function updateSummaryStats() {
    const total = rawRecords.length;
    const bonafide = rawRecords.filter(r => r.prediction === 'BONAFIDE').length;
    const spoof = rawRecords.filter(r => r.prediction === 'SPOOF').length;

    let avg = '—';
    if (total > 0) {
      const sum = rawRecords.reduce((acc, r) => acc + (typeof r.score === 'number' ? r.score : 0), 0);
      avg = (sum / total).toFixed(3);
    }

    statTotal.textContent = total;
    statBonafide.textContent = bonafide;
    statSpoof.textContent = spoof;
    statAvgScore.textContent = avg;

    paginationInfo.textContent = `Showing ${filteredRecords.length} of ${rawRecords.length} session records`;
  }

  // -------------------------------------------------------------
  // Render Table
  // -------------------------------------------------------------
  function renderTable() {
    if (filteredRecords.length === 0) {
      tableBody.innerHTML = `
        <tr>
          <td colspan="8" class="text-muted text-center" style="padding: 2.5rem;">
            ${rawRecords.length === 0 
              ? 'No analysis records found in this runtime session. Perform an analysis in the Forensics Workspace to populate history.' 
              : 'No records match the current search or verdict filter.'}
          </td>
        </tr>`;
      return;
    }

    tableBody.innerHTML = filteredRecords.map(r => {
      const isBonafide = r.prediction === 'BONAFIDE';
      const verdictBadge = isBonafide
        ? `<span class="badge-verdict badge-bonafide">BONAFIDE</span>`
        : `<span class="badge-verdict badge-spoof">SPOOF</span>`;

      const scoreVal = typeof r.score === 'number' ? r.score.toFixed(4) : 'N/A';
      const durationVal = r.duration_s ? `${r.duration_s.toFixed(2)}s` : (r.duration ? `${r.duration.toFixed(2)}s` : '—');
      const srVal = r.sample_rate ? `${(r.sample_rate / 1000).toFixed(1)}k` : '16k';
      const shaShort = r.sha256 ? `${r.sha256.substring(0, 10)}…` : '—';
      const timeFormatted = formatTimestamp(r.timestamp);

      return `
        <tr data-record-id="${escapeHtml(r.record_id || '')}">
          <td class="font-mono text-cyan" style="font-size:0.82rem;">${escapeHtml(r.record_id || 'N/A')}</td>
          <td class="text-muted" style="font-size:0.82rem;">${escapeHtml(timeFormatted)}</td>
          <td class="font-mono font-medium">${escapeHtml(r.filename || 'Unknown')}</td>
          <td class="font-mono text-muted" style="font-size:0.8rem;" title="${escapeHtml(r.sha256 || '')}">${escapeHtml(shaShort)}</td>
          <td class="font-mono text-muted" style="font-size:0.82rem;">${escapeHtml(durationVal)} / ${escapeHtml(srVal)}</td>
          <td class="font-mono ${isBonafide ? 'text-cyan' : 'text-red'} font-bold">${scoreVal}</td>
          <td>${verdictBadge}</td>
          <td>
            <div class="row-actions">
              <button class="btn btn-secondary btn-xs btn-view-dossier" data-id="${escapeHtml(r.record_id || '')}" title="View 7-Stage Forensic Case Dossier">
                ⚖️ Dossier
              </button>
              <a href="/analyze?sample=${encodeURIComponent(r.filename || '')}" class="btn btn-secondary btn-xs" title="Open in Forensics Workspace">
                🔬 Analyze
              </a>
              <button class="btn btn-danger-outline btn-xs btn-delete-record" data-id="${escapeHtml(r.record_id || '')}" title="Delete record from session">
                🗑️
              </button>
            </div>
          </td>
        </tr>
      `;
    }).join('');

    // Attach event listeners to action buttons
    tableBody.querySelectorAll('.btn-view-dossier').forEach(btn => {
      btn.addEventListener('click', () => openDossier(btn.getAttribute('data-id')));
    });

    tableBody.querySelectorAll('.btn-delete-record').forEach(btn => {
      btn.addEventListener('click', () => deleteSingleRecord(btn.getAttribute('data-id')));
    });
  }

  function formatTimestamp(ts) {
    if (!ts) return '—';
    try {
      const d = new Date(ts);
      return d.toLocaleTimeString('en-US', { hour12: false }) + ' ' + d.toISOString().split('T')[0];
    } catch {
      return ts;
    }
  }

  // -------------------------------------------------------------
  // Single Record Deletion
  // -------------------------------------------------------------
  async function deleteSingleRecord(recordId) {
    if (!recordId) return;
    const confirmDelete = window.confirm(`Delete record ${recordId} from session memory?`);
    if (!confirmDelete) return;

    try {
      const res = await fetch(`/api/history/${encodeURIComponent(recordId)}`, { method: 'DELETE' });
      if (!res.ok) throw new Error('Deletion failed');
      rawRecords = rawRecords.filter(r => r.record_id !== recordId);
      applyFilters();
    } catch (err) {
      alert(`Could not delete record: ${err.message}`);
    }
  }

  // -------------------------------------------------------------
  // Clear All History
  // -------------------------------------------------------------
  function showClearModal(show) {
    confirmClearModal.classList.toggle('hidden', !show);
    confirmClearBackdrop.classList.toggle('hidden', !show);
  }

  async function executeClearHistory() {
    try {
      const res = await fetch('/api/history', { method: 'DELETE' });
      if (!res.ok) throw new Error('Clear history failed');
      showClearModal(false);
      rawRecords = [];
      filteredRecords = [];
      applyFilters();
    } catch (err) {
      alert(`Failed to clear history: ${err.message}`);
    }
  }

  // -------------------------------------------------------------
  // Case Dossier Modal
  // -------------------------------------------------------------
  async function openDossier(recordId) {
    if (!recordId) return;
    dossierModal.classList.remove('hidden');
    dossierBackdrop.classList.remove('hidden');
    dossierContent.innerHTML = `<div class="text-center text-muted" style="padding: 3rem;">Retrieving Case Dossier for ${escapeHtml(recordId)}...</div>`;

    try {
      const res = await fetch(`/api/history/${encodeURIComponent(recordId)}/dossier`);
      if (!res.ok) throw new Error('Dossier not found or session expired');
      const data = await res.json();
      activeDossier = data.dossier;
      renderDossier(activeDossier);
    } catch (err) {
      dossierContent.innerHTML = `<div class="text-red text-center" style="padding: 2.5rem;">Error loading dossier: ${escapeHtml(err.message)}</div>`;
    }
  }

  function renderDossier(d) {
    const isBonafide = d.decision_metrics?.verdict === 'BONAFIDE';
    const s_cm = typeof d.decision_metrics?.countermeasure_score_s_cm === 'number'
      ? d.decision_metrics.countermeasure_score_s_cm.toFixed(4)
      : 'N/A';
    const spoofLogit = typeof d.decision_metrics?.spoof_logit === 'number'
      ? d.decision_metrics.spoof_logit.toFixed(4)
      : 'N/A';
    const margin = typeof d.decision_metrics?.decision_margin === 'number'
      ? d.decision_metrics.decision_margin.toFixed(4)
      : 'N/A';
    const softmax = typeof d.decision_metrics?.softmax_bonafide === 'number'
      ? (d.decision_metrics.softmax_bonafide * 100).toFixed(2) + '%'
      : 'N/A';

    const timelineHtml = (d.evidence_provenance_timeline || []).map(item => `
      <div class="timeline-step">
        <div class="timeline-dot ${item.status === 'COMPLETED' ? 'dot-completed' : ''}"></div>
        <div class="timeline-content">
          <div class="timeline-header">
            <span class="timeline-stage-pill font-mono">Stage ${item.stage}</span>
            <span class="timeline-name font-medium">${escapeHtml(item.name)}</span>
            <span class="timeline-status font-mono text-cyan">[${escapeHtml(item.status)}]</span>
          </div>
          <div class="timeline-desc text-muted">${escapeHtml(item.detail)}</div>
          <div class="timeline-ts font-mono text-muted" style="font-size:0.75rem;">Timestamp: ${escapeHtml(item.timestamp)}</div>
        </div>
      </div>
    `).join('');

    const disclaimersHtml = (d.limitations_and_disclaimers || []).map(txt => `
      <li class="disclaimer-item">${escapeHtml(txt)}</li>
    `).join('');

    dossierContent.innerHTML = `
      <div class="dossier-printable-document">
        
        <!-- HEADER SUMMARY -->
        <div class="dossier-header-banner">
          <div class="dossier-id-block">
            <div class="dossier-tag">ANALYSIS AUDIT RECORD (SESSION MEMORY &amp; PERSISTENT SQLITE)</div>
            <div class="dossier-id font-mono">${escapeHtml(d.dossier_id || '')}</div>
            <div class="dossier-date text-muted">Generated: ${escapeHtml(d.generated_at_utc || '')}</div>
          </div>
          <div class="dossier-verdict-block ${isBonafide ? 'verdict-bonafide' : 'verdict-spoof'}">
            <div class="verdict-label">FINAL CLASSIFICATION</div>
            <div class="verdict-text font-bold">${isBonafide ? 'BONA FIDE SPEECH' : 'SYNTHETIC / SPOOF'}</div>
            <div class="verdict-sub font-mono">S_cm = ${s_cm} (θ = 0.00)</div>
          </div>
        </div>

        <!-- SUBJECT RECORDING & HASH METADATA -->
        <div class="dossier-section">
          <h3 class="dossier-section-title">1. Subject Recording Integrity Metadata (SHA-256 Content Digest)</h3>
          <div class="dossier-meta-grid">
            <div class="meta-item"><span class="meta-label">Original Filename:</span><span class="meta-val font-mono">${escapeHtml(d.audio_metadata?.filename || 'Unknown')}</span></div>
            <div class="meta-item"><span class="meta-label">Duration:</span><span class="meta-val font-mono">${d.audio_metadata?.duration_s?.toFixed(3) || '—'}s (${d.audio_metadata?.num_samples || '—'} samples)</span></div>
            <div class="meta-item"><span class="meta-label">Input Sample Rate:</span><span class="meta-val font-mono">${d.audio_metadata?.sample_rate_hz || 16000} Hz (Canonical Mono)</span></div>
            <div class="meta-item"><span class="meta-label">File Size:</span><span class="meta-val font-mono">${d.audio_metadata?.file_size_bytes || '—'} bytes</span></div>
            <div class="meta-item full-width"><span class="meta-label">Input Audio SHA-256 Digest:</span><span class="meta-val font-mono text-cyan" style="word-break:break-all;">${escapeHtml(d.audio_metadata?.sha256 || 'N/A')}</span></div>
          </div>
        </div>

        <!-- MODEL & CHECKPOINT PROVENANCE -->
        <div class="dossier-section">
          <h3 class="dossier-section-title">2. Countermeasure Architecture &amp; Checkpoint Provenance</h3>
          <div class="dossier-meta-grid">
            <div class="meta-item"><span class="meta-label">Model Architecture:</span><span class="meta-val">${escapeHtml(d.model_provenance?.architecture || 'AASIST-L')}</span></div>
            <div class="meta-item"><span class="meta-label">Learnable Parameters:</span><span class="meta-val font-mono">${d.model_provenance?.parameter_count?.toLocaleString() || '85,306'}</span></div>
            <div class="meta-item"><span class="meta-label">Front-end:</span><span class="meta-val">${escapeHtml(d.model_provenance?.frontend || 'SincNet Raw Waveform Bandpass')}</span></div>
            <div class="meta-item"><span class="meta-label">Decision Threshold:</span><span class="meta-val font-mono">θ = ${d.model_provenance?.canonical_threshold ?? '0.00'}</span></div>
            <div class="meta-item full-width"><span class="meta-label">Model Weights SHA-256:</span><span class="meta-val font-mono text-muted" style="word-break:break-all;">${escapeHtml(d.model_provenance?.checkpoint_sha256 || 'N/A')}</span></div>
          </div>
        </div>

        <!-- DECISION LOGIT METRICS -->
        <div class="dossier-section">
          <h3 class="dossier-section-title">3. Quantitative Model Logit Metrics</h3>
          <div class="dossier-metrics-deck">
            <div class="metric-box ${isBonafide ? 'box-bonafide' : 'box-spoof'}">
              <div class="metric-box-label">S_cm Logit (Bona Fide)</div>
              <div class="metric-box-val font-mono font-bold">${s_cm}</div>
              <div class="metric-box-sub">Logit[0, 1]</div>
            </div>
            <div class="metric-box">
              <div class="metric-box-label">Spoof Logit</div>
              <div class="metric-box-val font-mono">${spoofLogit}</div>
              <div class="metric-box-sub">Logit[0, 0]</div>
            </div>
            <div class="metric-box">
              <div class="metric-box-label">Decision Boundary Margin</div>
              <div class="metric-box-val font-mono">${margin}</div>
              <div class="metric-box-sub">S_cm − θ (0.00)</div>
            </div>
            <div class="metric-box">
              <div class="metric-box-label">Softmax Probability (Uncalibrated)</div>
              <div class="metric-box-val font-mono">${softmax}</div>
              <div class="metric-box-sub">σ(Logits)[1]</div>
            </div>
          </div>
        </div>

        <!-- 7-STAGE LOGICAL PROCESSING OVERVIEW -->
        <div class="dossier-section">
          <h3 class="dossier-section-title">4. Seven-Stage Logical Processing Overview</h3>
          <p class="text-muted" style="font-size:0.8rem; margin-top:-0.3rem; margin-bottom:0.9rem; line-height:1.4;">
            Internal execution trace documenting preprocessing, sinc filter extraction, graph attention inference, and decision boundary mapping. Corresponds to byte-level SHA-256 verification digest; does not constitute an immutable legal chain of custody.
          </p>
          <div class="evidence-timeline">
            ${timelineHtml}
          </div>
        </div>

        <!-- FORENSIC DISCLAIMERS -->
        <div class="dossier-section">
          <h3 class="dossier-section-title">5. Forensic Disclaimers &amp; Scientific Limitations</h3>
          <ul class="disclaimers-list">
            ${disclaimersHtml}
          </ul>
        </div>

      </div>
    `;
  }

  function closeDossier() {
    dossierModal.classList.add('hidden');
    dossierBackdrop.classList.add('hidden');
    if (shareBox) shareBox.classList.add('hidden');
    if (shareStatusMsg) shareStatusMsg.textContent = '';
    activeDossier = null;
    currentShareToken = null;
  }

  async function handleCreateShare() {
    if (!activeDossier || !activeDossier.dossier_id) return;
    shareStatusMsg.textContent = 'Generating link...';
    try {
      const res = await fetch(`/api/share/${encodeURIComponent(activeDossier.dossier_id)}`, { method: 'POST' });
      if (!res.ok) throw new Error('Failed to create share link');
      const data = await res.json();
      currentShareToken = data.share_token;
      shareUrlInput.value = window.location.origin + data.share_url;
      shareBox.classList.remove('hidden');
      shareStatusMsg.textContent = 'Secret bearer link active (Expires in 7 days)';
      copyShareBtn.textContent = 'Copy';
    } catch (err) {
      shareStatusMsg.textContent = `Error: ${err.message}`;
    }
  }

  async function handleCopyShare() {
    if (!shareUrlInput || !shareUrlInput.value) return;
    try {
      await navigator.clipboard.writeText(shareUrlInput.value);
      copyShareBtn.textContent = '✓ Copied';
      setTimeout(() => { if (copyShareBtn) copyShareBtn.textContent = 'Copy'; }, 2000);
    } catch {
      shareUrlInput.select();
      document.execCommand('copy');
      copyShareBtn.textContent = '✓ Copied';
    }
  }

  async function handleRevokeShare() {
    if (!currentShareToken) return;
    const confirmRevoke = window.confirm('Revoke this share link immediately? Recipients will no longer have access.');
    if (!confirmRevoke) return;
    shareStatusMsg.textContent = 'Revoking...';
    try {
      const res = await fetch(`/api/share/${encodeURIComponent(currentShareToken)}`, { method: 'DELETE' });
      if (!res.ok) throw new Error('Revocation failed');
      shareUrlInput.value = '';
      currentShareToken = null;
      shareStatusMsg.textContent = 'Link successfully revoked';
      setTimeout(() => { if (shareBox) shareBox.classList.add('hidden'); }, 1500);
    } catch (err) {
      shareStatusMsg.textContent = `Error: ${err.message}`;
    }
  }

  function downloadActiveDossierJson() {
    if (!activeDossier) return;
    const id = activeDossier.dossier_id || 'dossier';
    const jsonStr = JSON.stringify(activeDossier, null, 2);
    downloadBlob(jsonStr, `forensic_case_${id}.json`, 'application/json');
  }

  // -------------------------------------------------------------
  // Export JSON & CSV
  // -------------------------------------------------------------
  function exportFilteredJson() {
    const data = {
      export_timestamp_utc: new Date().toISOString(),
      record_count: filteredRecords.length,
      records: filteredRecords
    };
    downloadBlob(JSON.stringify(data, null, 2), `audio_authenticity_session_${Date.now()}.json`, 'application/json');
  }

  function exportFilteredCsv() {
    const headers = [
      'Record_ID',
      'Timestamp_UTC',
      'Filename',
      'Duration_s',
      'Sample_Rate_Hz',
      'SHA256_Audio_Digest',
      'S_cm_Logit',
      'Classification',
      'Spoof_Logit',
      'Softmax_Bonafide'
    ];

    const rows = filteredRecords.map(r => [
      r.record_id || '',
      r.timestamp || '',
      `"${(r.filename || '').replace(/"/g, '""')}"`,
      r.duration_s ?? (r.duration ?? ''),
      r.sample_rate || 16000,
      r.sha256 || '',
      r.score ?? '',
      r.prediction || '',
      r.spoof_logit ?? '',
      r.softmax_bonafide ?? ''
    ]);

    const csvContent = [headers.join(','), ...rows.map(row => row.join(','))].join('\n');
    downloadBlob(csvContent, `audio_authenticity_session_${Date.now()}.csv`, 'text/csv');
  }

  function downloadBlob(content, filename, type) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function escapeHtml(str) {
    if (typeof str !== 'string') return String(str);
    return str.replace(/[&<>"']/g, m => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    })[m]);
  }

  // Start on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
