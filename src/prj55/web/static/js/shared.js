/**
 * Audio Authenticity Lab — Shared Forensic Dossier Controller
 * Renders sanitized forensic analysis reports from token-based share links.
 */

(function () {
  'use strict';

  const loadingEl = document.getElementById('sharedLoading');
  const errorEl = document.getElementById('sharedError');
  const errorMsgEl = document.getElementById('sharedErrorMsg');
  const dossierViewEl = document.getElementById('sharedDossierView');
  const printReportBtn = document.getElementById('printReportBtn');

  const mobileMenuBtn = document.getElementById('mobileMenuBtn');
  const mobileNavDrawer = document.getElementById('mobileNavDrawer');
  const mobileNavBackdrop = document.getElementById('mobileNavBackdrop');
  const closeMobileNavBtn = document.getElementById('closeMobileNavBtn');

  function openMobileNav() {
    if (mobileNavDrawer) mobileNavDrawer.classList.remove('hidden');
    if (mobileNavBackdrop) mobileNavBackdrop.classList.remove('hidden');
  }

  function closeMobileNav() {
    if (mobileNavDrawer) mobileNavDrawer.classList.add('hidden');
    if (mobileNavBackdrop) mobileNavBackdrop.classList.add('hidden');
  }

  if (mobileMenuBtn) mobileMenuBtn.addEventListener('click', openMobileNav);
  if (closeMobileNavBtn) closeMobileNavBtn.addEventListener('click', closeMobileNav);
  if (mobileNavBackdrop) mobileNavBackdrop.addEventListener('click', closeMobileNav);

  if (printReportBtn) {
    printReportBtn.addEventListener('click', () => {
      document.body.classList.add('shared-view-print');
      window.print();
      document.body.classList.remove('shared-view-print');
    });
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

  async function loadSharedReport() {
    const pathParts = window.location.pathname.split('/');
    const tokenIndex = pathParts.indexOf('shared');
    const token = (tokenIndex !== -1 && pathParts[tokenIndex + 1]) ? pathParts[tokenIndex + 1] : '';

    if (!token) {
      showError('No share token was provided in the URL.');
      return;
    }

    try {
      const res = await fetch(`/api/share/${encodeURIComponent(token)}`);
      if (!res.ok) {
        if (res.status === 404) {
          showError('This shared forensic report link does not exist, has expired, or was revoked.');
        } else {
          showError(`Could not load report (HTTP ${res.status}).`);
        }
        return;
      }

      const data = await res.json();
      renderDossier(data);
    } catch (err) {
      showError(`Network connection error: ${err.message}`);
    }
  }

  function showError(msg) {
    if (loadingEl) loadingEl.classList.add('hidden');
    if (dossierViewEl) dossierViewEl.classList.add('hidden');
    if (errorMsgEl) errorMsgEl.textContent = msg;
    if (errorEl) errorEl.classList.remove('hidden');
  }

  function renderDossier(data) {
    const summary = data.forensic_summary || {};
    const meta = data.audio_metadata || {};
    const prov = data.model_provenance || {};
    const isBonafide = summary.final_prediction === 'BONAFIDE';
    const timeline = data.evidence_provenance_timeline || [];
    const sha = meta.sha256 || 'UNKNOWN';

    if (!dossierViewEl) return;

    dossierViewEl.innerHTML = `
      <!-- Header Banner -->
      <div class="shared-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1rem; margin-bottom:1rem;">
          <div>
            <span class="badge-tag" style="margin-bottom:0.5rem;"><span class="tag-icon">⚖️</span> FORENSIC CASE DOSSIER</span>
            <h1 style="font-size:1.6rem; font-weight:800; margin:0.25rem 0 0.5rem;">Authenticity Verification Report</h1>
            <div class="font-mono text-muted" style="font-size:0.85rem;">Record Reference: <strong class="text-cyan">${escapeHtml(data.record_id || '')}</strong></div>
          </div>
          <div class="text-right" style="font-size:0.8rem; color:var(--text-secondary);">
            <div>Generated: ${escapeHtml(data.shared_at || '—')}</div>
            <div>Access Count: <strong class="text-cyan">${data.access_count || 1}</strong></div>
          </div>
        </div>

        <!-- Verdict Box -->
        <div class="shared-verdict-box ${isBonafide ? 'bonafide' : 'spoof'}">
          <div>
            <div style="font-size:0.85rem; letter-spacing:0.05em; text-transform:uppercase; color:var(--text-secondary);">Operational Determination</div>
            <div class="shared-verdict-title ${isBonafide ? 'text-cyan' : 'text-red'}">
              ${isBonafide ? 'BONA FIDE (HUMAN SPEECH)' : 'SPOOF (SYNTHETIC / DEEPFAKE)'}
            </div>
            <div class="font-mono text-muted" style="font-size:0.85rem; margin-top:0.25rem;">
              ${escapeHtml(summary.verdict_rule || '')}
            </div>
          </div>
          <div class="text-right font-mono" style="font-size:0.9rem;">
            <div>Countermeasure Score (S_cm): <strong class="${isBonafide ? 'text-cyan' : 'text-red'}">${summary.primary_countermeasure_score?.toFixed(4) || '—'}</strong></div>
            <div>Decision Threshold (θ): <strong>${summary.decision_threshold?.toFixed(2) || '0.00'}</strong></div>
            <div>Decision Margin: <strong>${summary.decision_margin?.toFixed(4) || '—'}</strong></div>
          </div>
        </div>

        <!-- Metadata Grid -->
        <div class="shared-header-grid font-mono" style="font-size:0.85rem; background:rgba(0,0,0,0.2); padding:1rem; border-radius:var(--radius-md);">
          <div><span class="text-muted">Filename:</span> <strong>${escapeHtml(meta.filename || '')}</strong></div>
          <div><span class="text-muted">Duration:</span> <strong>${meta.duration_s ? meta.duration_s.toFixed(2) + 's' : '—'}</strong></div>
          <div><span class="text-muted">Sample Rate:</span> <strong>${meta.sample_rate_hz || 16000} Hz</strong></div>
          <div><span class="text-muted">Model:</span> <strong>${escapeHtml(prov.neural_architecture || 'AASIST-L')}</strong></div>
        </div>
        <div class="font-mono" style="font-size:0.8rem; margin-top:0.75rem; word-break:break-all;">
          <span class="text-muted">Audio Content Digest (SHA-256):</span> <span class="text-cyan">${escapeHtml(sha)}</span>
        </div>
      </div>

      <!-- 7-Stage Logical Processing Overview -->
      <div class="shared-card">
        <h2 style="font-size:1.15rem; font-weight:700; margin-bottom:0.35rem; letter-spacing:0.02em;">7-Stage Logical Processing Overview</h2>
        <p class="text-muted" style="font-size:0.8rem; margin-bottom:1rem; line-height:1.4;">
          Internal execution trace documenting preprocessing, sinc filter extraction, graph attention inference, and decision boundary mapping. Corresponds to byte-level SHA-256 content verification digest; does not constitute an immutable legal chain-of-custody seal.
        </p>
        <div>
          ${timeline.map(st => `
            <div class="stage-row">
              <div class="stage-num">${st.stage}</div>
              <div style="flex:1;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                  <strong style="font-size:0.95rem;">${escapeHtml(st.name)}</strong>
                  <span class="pill-badge success" style="font-size:0.72rem; padding:0.15rem 0.5rem;">${escapeHtml(st.status)}</span>
                </div>
                <div class="text-muted font-mono" style="font-size:0.85rem; margin-top:0.25rem;">${escapeHtml(st.detail)}</div>
              </div>
            </div>
          `).join('')}
        </div>
      </div>

      <!-- Governance & Privacy Notice -->
      <div class="shared-card" style="border-left: 3px solid var(--accent-cyan);">
        <h3 style="font-size:1rem; font-weight:700; margin-bottom:0.5rem;">Access Scope &amp; Research Governance (ADR-002)</h3>
        <p style="font-size:0.88rem; color:var(--text-secondary); line-height:1.6; margin-bottom:0.5rem;">
          ${escapeHtml(data.security_and_privacy_notice?.access_model || '')}
        </p>
        <p style="font-size:0.88rem; color:var(--text-secondary); line-height:1.6; margin:0;">
          ${escapeHtml(data.security_and_privacy_notice?.raw_audio_policy || '')}
        </p>
      </div>
    `;

    if (loadingEl) loadingEl.classList.add('hidden');
    if (dossierViewEl) dossierViewEl.classList.remove('hidden');
  }

  document.addEventListener('DOMContentLoaded', loadSharedReport);
})();
