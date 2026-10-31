/**
 * Audio Authenticity Lab — Global Command Palette & Productivity Suite
 * Supports Cmd+K / Ctrl+K navigation, high-contrast theme toggle, and keyboard shortcuts modal.
 */

(function () {
  'use strict';

  // Apply persisted theme on startup
  const savedTheme = localStorage.getItem('aal-theme') || 'dark';
  if (savedTheme === 'high-contrast') {
    document.documentElement.setAttribute('data-theme', 'high-contrast');
  }

  // Register Service Worker for PWA shell caching
  if ('serviceWorker' in navigator && window.location.protocol.startsWith('http')) {
    window.addEventListener('load', () => {
      navigator.serviceWorker.register('/sw.js').catch((err) => {
        console.debug('[PWA] Service worker registration skipped/failed:', err);
      });
    });
  }

  const commands = [
    {
      category: 'NAVIGATION',
      title: 'Analyze Audio Workspace',
      desc: 'Inspect signals, compute AASIST-L logits, and view evidence',
      shortcut: 'G A',
      action: () => { window.location.href = '/analyze'; }
    },
    {
      category: 'NAVIGATION',
      title: 'Robustness & Perturbation Lab',
      desc: 'Simulate telephony, additive noise, and test front-end mitigations',
      shortcut: 'G R',
      action: () => { window.location.href = '/robustness'; }
    },
    {
      category: 'NAVIGATION',
      title: 'Session Experiment History',
      desc: 'Search, filter, and inspect forensic case dossiers',
      shortcut: 'G H',
      action: () => { window.location.href = '/history'; }
    },
    {
      category: 'NAVIGATION',
      title: 'Research Methodology & Specs',
      desc: 'AASIST-L architecture, protocol baselines, and model constraints',
      shortcut: 'G M',
      action: () => { window.location.href = '/methodology'; }
    },
    {
      category: 'NAVIGATION',
      title: 'Cinematic Landing Page',
      desc: 'Return to public hero presentation and 3D harmonic sculpture',
      shortcut: 'G 0',
      action: () => { window.location.href = '/'; }
    },
    {
      category: 'PRESETS',
      title: 'Load Track TRK-01: Authentic Male Voice',
      desc: 'Natural spoken English utterance from VCTK / ASVspoof 2019',
      shortcut: '1',
      action: () => loadPresetTrack('speech_human_male')
    },
    {
      category: 'PRESETS',
      title: 'Load Track TRK-02: Authentic Female Voice',
      desc: 'Natural spoken English utterance from VCTK / ASVspoof 2019',
      shortcut: '2',
      action: () => loadPresetTrack('speech_human_female')
    },
    {
      category: 'PRESETS',
      title: 'Load Track TRK-03: Neural TTS Deepfake',
      desc: 'State-of-the-art neural speech synthesis attack (SS_6)',
      shortcut: '3',
      action: () => loadPresetTrack('speech_deepfake_tts')
    },
    {
      category: 'PRESETS',
      title: 'Load Track TRK-04: Vocoder Artifact Deepfake',
      desc: 'Deepfake voice conversion with frame stitching anomalies (SS_3)',
      shortcut: '4',
      action: () => loadPresetTrack('speech_deepfake_vocoder')
    },
    {
      category: 'PREFERENCES',
      title: 'Toggle High-Contrast Accessibility Mode',
      desc: 'Enhance visual contrast, sharp borders, and legible text markers',
      shortcut: 'T C',
      action: toggleHighContrastTheme
    },
    {
      category: 'HELP',
      title: 'View Keyboard Shortcuts',
      desc: 'Display quick navigation and playback keybindings',
      shortcut: '?',
      action: showShortcutsModal
    }
  ];

  let selectedIndex = 0;
  let filteredCommands = [...commands];

  function toggleHighContrastTheme() {
    const isHC = document.documentElement.getAttribute('data-theme') === 'high-contrast';
    if (isHC) {
      document.documentElement.removeAttribute('data-theme');
      localStorage.setItem('aal-theme', 'dark');
    } else {
      document.documentElement.setAttribute('data-theme', 'high-contrast');
      localStorage.setItem('aal-theme', 'high-contrast');
    }
    closePalette();
  }

  function loadPresetTrack(sampleId) {
    closePalette();
    if (window.location.pathname.includes('/analyze')) {
      const btn = document.querySelector(`button[data-sample="${sampleId}"]`);
      if (btn) btn.click();
    } else {
      window.location.href = `/analyze?sample=${sampleId}`;
    }
  }

  function renderPaletteDOM() {
    if (document.getElementById('cmdPaletteModal')) return;

    const modal = document.createElement('div');
    modal.id = 'cmdPaletteModal';
    modal.className = 'cmd-palette-modal hidden';
    modal.setAttribute('role', 'dialog');
    modal.setAttribute('aria-modal', 'true');
    modal.setAttribute('aria-label', 'Command Palette');

    modal.innerHTML = `
      <div class="cmd-palette-backdrop" id="cmdPaletteBackdrop"></div>
      <div class="cmd-palette-box">
        <div class="cmd-palette-header">
          <span class="cmd-search-icon">🔍</span>
          <input type="text" id="cmdPaletteInput" class="cmd-palette-input" placeholder="Type a command or search actions..." autocomplete="off" spellcheck="false">
          <kbd class="cmd-esc-badge">ESC</kbd>
        </div>
        <div class="cmd-palette-list" id="cmdPaletteList" role="listbox"></div>
        <div class="cmd-palette-footer">
          <span class="cmd-hint"><kbd>↑</kbd><kbd>↓</kbd> Navigate</span>
          <span class="cmd-hint"><kbd>↵</kbd> Select</span>
          <span class="cmd-hint"><kbd>ESC</kbd> Close</span>
          <span class="cmd-hint-brand">Audio Authenticity Lab</span>
        </div>
      </div>
    `;

    document.body.appendChild(modal);

    document.getElementById('cmdPaletteBackdrop').addEventListener('click', closePalette);
    const input = document.getElementById('cmdPaletteInput');
    input.addEventListener('input', (e) => {
      filterCommands(e.target.value);
    });

    input.addEventListener('keydown', handleKeyNavigation);
  }

  function renderShortcutsModal() {
    if (document.getElementById('shortcutsModal')) return;

    const modal = document.createElement('div');
    modal.id = 'shortcutsModal';
    modal.className = 'modal-backdrop hidden';
    modal.setAttribute('role', 'dialog');
    modal.setAttribute('aria-modal', 'true');
    modal.setAttribute('aria-label', 'Keyboard Shortcuts');

    modal.innerHTML = `
      <div class="modal-card">
        <div class="modal-header">
          <div class="modal-title">[KEYBOARD SHORTCUTS REFERENCE]</div>
          <button class="btn-close-modal" id="closeShortcutsModalBtn" aria-label="Close shortcuts modal">✕</button>
        </div>
        <div class="modal-body">
          <div class="shortcuts-grid">
            <div class="shortcut-row"><span class="shortcut-desc">Open Command Palette</span><span class="shortcut-keys"><kbd>⌘</kbd> <kbd>K</kbd> / <kbd>Ctrl</kbd> <kbd>K</kbd></span></div>
            <div class="shortcut-row"><span class="shortcut-desc">Toggle Keyboard Shortcuts</span><span class="shortcut-keys"><kbd>?</kbd></span></div>
            <div class="shortcut-row"><span class="shortcut-desc">Dismiss Active Dialog / Drawer</span><span class="shortcut-keys"><kbd>ESC</kbd></span></div>
            <div class="shortcut-row"><span class="shortcut-desc">Play / Pause Audio</span><span class="shortcut-keys"><kbd>Space</kbd></span></div>
            <div class="shortcut-row"><span class="shortcut-desc">Toggle High-Contrast Mode</span><span class="shortcut-keys"><kbd>T</kbd> <kbd>C</kbd> in Palette</span></div>
            <div class="shortcut-row"><span class="shortcut-desc">Go to Analysis Workspace</span><span class="shortcut-keys"><kbd>G</kbd> <kbd>A</kbd> in Palette</span></div>
            <div class="shortcut-row"><span class="shortcut-desc">Go to Robustness Lab</span><span class="shortcut-keys"><kbd>G</kbd> <kbd>R</kbd> in Palette</span></div>
            <div class="shortcut-row"><span class="shortcut-desc">Go to Experiment History</span><span class="shortcut-keys"><kbd>G</kbd> <kbd>H</kbd> in Palette</span></div>
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn btn-primary btn-sm" id="closeShortcutsConfirmBtn">Got It</button>
        </div>
      </div>
    `;

    document.body.appendChild(modal);

    const close = () => modal.classList.add('hidden');
    document.getElementById('closeShortcutsModalBtn').addEventListener('click', close);
    document.getElementById('closeShortcutsConfirmBtn').addEventListener('click', close);
    modal.addEventListener('click', (e) => {
      if (e.target === modal) close();
    });
  }

  function showShortcutsModal() {
    closePalette();
    renderShortcutsModal();
    const modal = document.getElementById('shortcutsModal');
    if (modal) modal.classList.remove('hidden');
  }

  function filterCommands(query) {
    const q = (query || '').toLowerCase().trim();
    if (!q) {
      filteredCommands = [...commands];
    } else {
      filteredCommands = commands.filter(
        (c) => c.title.toLowerCase().includes(q) || c.desc.toLowerCase().includes(q) || c.category.toLowerCase().includes(q)
      );
    }
    selectedIndex = 0;
    renderCommandList();
  }

  function renderCommandList() {
    const list = document.getElementById('cmdPaletteList');
    if (!list) return;

    if (filteredCommands.length === 0) {
      list.innerHTML = `<div class="cmd-no-results">No matching actions found. Try searching for "analyze", "robustness", or "theme".</div>`;
      return;
    }

    list.innerHTML = filteredCommands.map((cmd, idx) => `
      <div class="cmd-item ${idx === selectedIndex ? 'selected' : ''}" role="option" aria-selected="${idx === selectedIndex}" data-idx="${idx}">
        <div class="cmd-item-info">
          <span class="cmd-category-tag">${cmd.category}</span>
          <span class="cmd-item-title">${cmd.title}</span>
          <span class="cmd-item-desc">${cmd.desc}</span>
        </div>
        ${cmd.shortcut ? `<span class="cmd-item-shortcut">${cmd.shortcut}</span>` : ''}
      </div>
    `).join('');

    const items = list.querySelectorAll('.cmd-item');
    items.forEach((item) => {
      item.addEventListener('mouseenter', () => {
        const idx = parseInt(item.getAttribute('data-idx'), 10);
        selectedIndex = idx;
        updateSelectedHighlight();
      });
      item.addEventListener('click', () => {
        executeCommand(selectedIndex);
      });
    });
  }

  function updateSelectedHighlight() {
    const list = document.getElementById('cmdPaletteList');
    if (!list) return;
    const items = list.querySelectorAll('.cmd-item');
    items.forEach((item, idx) => {
      if (idx === selectedIndex) {
        item.classList.add('selected');
        item.setAttribute('aria-selected', 'true');
        item.scrollIntoView({ block: 'nearest' });
      } else {
        item.classList.remove('selected');
        item.setAttribute('aria-selected', 'false');
      }
    });
  }

  function handleKeyNavigation(e) {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      selectedIndex = (selectedIndex + 1) % Math.max(1, filteredCommands.length);
      updateSelectedHighlight();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      selectedIndex = (selectedIndex - 1 + filteredCommands.length) % Math.max(1, filteredCommands.length);
      updateSelectedHighlight();
    } else if (e.key === 'Enter') {
      e.preventDefault();
      executeCommand(selectedIndex);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      closePalette();
    }
  }

  function executeCommand(idx) {
    const cmd = filteredCommands[idx];
    if (cmd && typeof cmd.action === 'function') {
      cmd.action();
    }
  }

  function openPalette() {
    renderPaletteDOM();
    const modal = document.getElementById('cmdPaletteModal');
    if (!modal) return;
    modal.classList.remove('hidden');
    const input = document.getElementById('cmdPaletteInput');
    if (input) {
      input.value = '';
      input.focus();
    }
    filterCommands('');
  }

  function closePalette() {
    const modal = document.getElementById('cmdPaletteModal');
    if (modal) modal.classList.add('hidden');
  }

  // Global keydown handler
  window.addEventListener('keydown', (e) => {
    // Cmd+K or Ctrl+K opens palette
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      const modal = document.getElementById('cmdPaletteModal');
      if (modal && !modal.classList.contains('hidden')) {
        closePalette();
      } else {
        openPalette();
      }
      return;
    }

    // Escape closes active modals
    if (e.key === 'Escape') {
      closePalette();
      const shortcuts = document.getElementById('shortcutsModal');
      if (shortcuts) shortcuts.classList.add('hidden');
      return;
    }

    // Question mark triggers shortcuts when not typing in an input
    if (e.key === '?' && !['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) {
      e.preventDefault();
      showShortcutsModal();
    }
  });

  // Export palette functions to window
  window.AALPalette = {
    open: openPalette,
    close: closePalette,
    toggleTheme: toggleHighContrastTheme,
    showShortcuts: showShortcutsModal
  };

})();
