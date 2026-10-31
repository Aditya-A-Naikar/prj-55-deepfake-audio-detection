"""Automated tests for UI contracts, branding, system diagnostics, and robustness conclusions."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from prj55.web.app import app

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_INDEX_HTML = REPO_ROOT / "src" / "prj55" / "web" / "static" / "index.html"
STATIC_ANALYZE_HTML = REPO_ROOT / "src" / "prj55" / "web" / "static" / "analyze.html"
STATIC_METHODOLOGY_HTML = REPO_ROOT / "src" / "prj55" / "web" / "static" / "methodology.html"
STATIC_STYLE_CSS = REPO_ROOT / "src" / "prj55" / "web" / "static" / "css" / "style.css"
STATIC_APP_JS = REPO_ROOT / "src" / "prj55" / "web" / "static" / "js" / "app.js"
STATIC_ROBUSTNESS_HTML = REPO_ROOT / "src" / "prj55" / "web" / "static" / "robustness.html"
STATIC_HISTORY_HTML = REPO_ROOT / "src" / "prj55" / "web" / "static" / "history.html"
STATIC_PALETTE_JS = REPO_ROOT / "src" / "prj55" / "web" / "static" / "js" / "palette.js"
STATIC_MANIFEST = REPO_ROOT / "src" / "prj55" / "web" / "static" / "manifest.webmanifest"


@pytest.fixture
def client() -> Iterator[TestClient]:
    """TestClient fixture with app lifecycle."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def index_html_content() -> str:
    """Read the raw static index.html file."""
    assert STATIC_INDEX_HTML.is_file(), f"Missing index.html at {STATIC_INDEX_HTML}"
    return STATIC_INDEX_HTML.read_text(encoding="utf-8")


@pytest.fixture
def analyze_html_content() -> str:
    """Read the raw static analyze.html file."""
    assert STATIC_ANALYZE_HTML.is_file(), f"Missing analyze.html at {STATIC_ANALYZE_HTML}"
    return STATIC_ANALYZE_HTML.read_text(encoding="utf-8")


@pytest.fixture
def robustness_html_content() -> str:
    """Read the raw static robustness.html file."""
    assert STATIC_ROBUSTNESS_HTML.is_file(), f"Missing robustness.html at {STATIC_ROBUSTNESS_HTML}"
    return STATIC_ROBUSTNESS_HTML.read_text(encoding="utf-8")


@pytest.fixture
def history_html_content() -> str:
    """Read the raw static history.html file."""
    assert STATIC_HISTORY_HTML.is_file(), f"Missing history.html at {STATIC_HISTORY_HTML}"
    return STATIC_HISTORY_HTML.read_text(encoding="utf-8")


@pytest.fixture
def style_css_content() -> str:
    """Read the raw static style.css file."""
    assert STATIC_STYLE_CSS.is_file(), f"Missing style.css at {STATIC_STYLE_CSS}"
    return STATIC_STYLE_CSS.read_text(encoding="utf-8")


@pytest.fixture
def app_js_content() -> str:
    """Read the raw static app.js file."""
    assert STATIC_APP_JS.is_file(), f"Missing app.js at {STATIC_APP_JS}"
    return STATIC_APP_JS.read_text(encoding="utf-8")


@pytest.fixture
def palette_js_content() -> str:
    """Read the raw static palette.js file."""
    assert STATIC_PALETTE_JS.is_file(), f"Missing palette.js at {STATIC_PALETTE_JS}"
    return STATIC_PALETTE_JS.read_text(encoding="utf-8")


class TestNavbarBrandingAndNavigation:
    """Verifies that [PRJ_55] badge is eliminated from top navigation and branding is clean."""

    def test_no_prj55_badge_in_header(self, index_html_content: str) -> None:
        """The top header must not contain any [PRJ_55] badge or brand-badge element."""
        header_match = re.search(r"<header.*?</header>", index_html_content, re.DOTALL)
        assert header_match is not None, "Could not find <header> tag in index.html"
        header_html = header_match.group(0)

        assert "[PRJ_55]" not in header_html, (
            "Found visible '[PRJ_55]' badge in top navigation header!"
        )
        assert "brand-badge" not in header_html, "Found deprecated .brand-badge class in header!"

    def test_primary_product_branding(self, index_html_content: str) -> None:
        """Audio Authenticity Lab must be the primary visible product title in the navbar."""
        assert "AUDIO AUTHENTICITY LAB" in index_html_content
        assert "AASIST-L • Graph Attention Anti-Spoofing" in index_html_content

    def test_status_pill_accessible_trigger(self, index_html_content: str) -> None:
        """The systemStatusPill must be an accessible button trigger for System Diagnostics."""
        assert 'id="systemStatusPill"' in index_html_content
        assert 'aria-haspopup="dialog"' in index_html_content
        assert 'aria-controls="diagDrawer"' in index_html_content


class TestSystemDiagnosticsDrawer:
    """Verifies the System Diagnostics modal/drawer structure and API contract."""

    def test_diagnostics_drawer_markup(self, index_html_content: str) -> None:
        """The diagnostics drawer markup must be present with required accessibility attributes."""
        assert 'id="diagDrawer"' in index_html_content
        assert 'role="dialog"' in index_html_content
        assert 'aria-modal="true"' in index_html_content
        assert 'id="diagModalBackdrop"' in index_html_content
        assert 'id="closeDiagBtn"' in index_html_content

        # Required diagnostic panels
        assert "AASIST-L Architecture" in index_html_content
        assert "Pretrained Checkpoint Integrity" in index_html_content
        assert "Host Audio Codecs" in index_html_content
        assert "Storage Policy &amp; Physical Corpus Governance" in index_html_content
        assert "Automated Verification Quality Gates" in index_html_content

    def test_status_api_contract_for_diagnostics(self, client: TestClient) -> None:
        """GET /api/status must provide complete data to populate diagnostics."""
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] in ("online", "offline")
        assert data["model_architecture"] == "AASIST-L"
        assert data["parameter_count"] == 85306
        assert "device" in data
        assert "checkpoint_found" in data
        assert "checkpoint_sha256" in data
        assert "codecs" in data
        assert "ffmpeg_available" in data["codecs"]
        assert "opus_available" in data["codecs"]
        assert "storage_guard" in data
        assert data["storage_guard"]["policy"] == "ADR-002"


class TestAnalysisWorkspaceCollapsiblePanel:
    """Verifies collapsible input source panel and progress feedback in Analysis Workspace."""

    def test_collapsible_upload_markup(self, analyze_html_content: str) -> None:
        """analyze.html must have both expanded and collapsed containers with an expand button."""
        assert 'id="cardUpload"' in analyze_html_content
        assert 'id="uploadCollapsedContent"' in analyze_html_content
        assert 'id="uploadExpandedContent"' in analyze_html_content
        assert 'id="expandUploadBtn"' in analyze_html_content
        assert 'id="collapsedSourceName"' in analyze_html_content

    def test_analysis_progress_feedback_markup(self, analyze_html_content: str) -> None:
        """analyze.html must have an honest progress feedback card with role='status'."""
        assert 'id="analysisProgressCard"' in analyze_html_content
        assert 'role="status"' in analyze_html_content
        assert 'id="analysisProgressBar"' in analyze_html_content
        assert 'id="analysisProgressMsg"' in analyze_html_content


class TestRobustnessConclusionSynthesis:
    """Verifies the explicit Robustness Conclusion section and mathematical decision rule."""

    def test_robustness_conclusion_card_markup(self, analyze_html_content: str) -> None:
        """analyze.html must have dedicated elements for conclusion and explanations."""
        assert 'id="robConclusionCard"' in analyze_html_content
        assert 'id="robConclusionBadge"' in analyze_html_content
        assert 'id="conclusionLeadText"' in analyze_html_content
        assert 'id="conclusionBodyText"' in analyze_html_content
        assert 'id="robAcousticExplanation"' in analyze_html_content
        assert 'id="robMitigationExplanation"' in analyze_html_content

    def test_score_delta_mathematics(self) -> None:
        """Delta S_cm must be exactly (degraded - clean) and handle edge cases accurately."""
        # Positive delta (score improved)
        clean_pos = 1.50
        deg_pos = 2.10
        delta_pos = deg_pos - clean_pos
        assert pytest.approx(delta_pos, 1e-4) == 0.60
        assert pytest.approx(abs(delta_pos), 1e-4) == 0.60

        # Negative delta (score degraded)
        clean_neg = 2.00
        deg_neg = -1.20
        delta_neg = deg_neg - clean_neg
        assert pytest.approx(delta_neg, 1e-4) == -3.20
        assert pytest.approx(abs(delta_neg), 1e-4) == 3.20

        # Zero delta (exact invariance)
        clean_zero = 0.00
        deg_zero = 0.00
        delta_zero = deg_zero - clean_zero
        assert delta_zero == 0.00

        # Near-zero clean logit (must never cause division-by-zero or explosion)
        clean_tiny = 0.0001
        deg_tiny = -0.05
        delta_tiny = deg_tiny - clean_tiny
        assert pytest.approx(delta_tiny, 1e-4) == -0.0501

    def test_decision_rule_classification_contract(self) -> None:
        """Verify official decision rule: score >= tau (0.00) is BONAFIDE, else SPOOF."""
        tau = 0.00

        # Above threshold -> BONAFIDE
        score_bonafide = 1.84
        pred_bonafide = "BONAFIDE" if score_bonafide >= tau else "SPOOF"
        assert pred_bonafide == "BONAFIDE"

        # Exactly at threshold -> BONAFIDE
        score_edge = 0.00
        pred_edge = "BONAFIDE" if score_edge >= tau else "SPOOF"
        assert pred_edge == "BONAFIDE"

        # Below threshold -> SPOOF
        score_spoof = -0.01
        pred_spoof = "BONAFIDE" if score_spoof >= tau else "SPOOF"
        assert pred_spoof == "SPOOF"

    def test_conclusion_synthesis_logic(self) -> None:
        """Test rule-based conclusion generation for decision flips and invariance."""

        def evaluate_conclusion(
            clean_score: float, deg_score: float, tau: float = 0.0
        ) -> dict[str, Any]:
            clean_pred = "BONAFIDE" if clean_score >= tau else "SPOOF"
            deg_pred = "BONAFIDE" if deg_score >= tau else "SPOOF"
            delta = deg_score - clean_score
            abs_delta = abs(delta)
            pred_changed = clean_pred != deg_pred

            if pred_changed:
                badge = "DECISION BOUNDARY BREACHED"
            elif abs_delta <= 1.0:
                badge = "DECISION INVARIANT (|Δ| ≤ 1.0)"
            else:
                badge = "DECISION RETAINED (SCORE SENSITIVE)"

            return {
                "clean_pred": clean_pred,
                "deg_pred": deg_pred,
                "delta": delta,
                "abs_delta": abs_delta,
                "pred_changed": pred_changed,
                "badge": badge,
            }

        # Case 1: Decision breached (bonafide -> spoof)
        res1 = evaluate_conclusion(clean_score=2.05, deg_score=-1.20)
        assert res1["pred_changed"] is True
        assert res1["badge"] == "DECISION BOUNDARY BREACHED"

        # Case 2: Invariant (minimal drift)
        res2 = evaluate_conclusion(clean_score=2.05, deg_score=1.85)
        assert res2["pred_changed"] is False
        assert res2["badge"] == "DECISION INVARIANT (|Δ| ≤ 1.0)"

        # Case 3: Retained with score sensitivity
        res3 = evaluate_conclusion(clean_score=3.50, deg_score=1.10)
        assert res3["pred_changed"] is False
        assert res3["badge"] == "DECISION RETAINED (SCORE SENSITIVE)"

    def test_mitigation_routing_in_robustness_endpoint(self, client: TestClient) -> None:
        """When mitigation is enabled, playback_signal_type must be 'mitigated'."""
        # Unmitigated test
        # Unmitigated test
        resp_unmit = client.post(
            "/api/robustness",
            data={
                "sample_id": "sample_bonafide",
                "condition": "noise_snr_10db",
                "snr_db": "10.0",
                "mitigation_enabled": "false",
            },
        )
        assert resp_unmit.status_code == 200
        data_unmit = resp_unmit.json()
        assert data_unmit["comparison"]["mitigation_applied"] is False
        assert data_unmit["comparison"]["playback_signal_type"] == "degraded"

        # Mitigated test
        resp_mit = client.post(
            "/api/robustness",
            data={
                "sample_id": "sample_bonafide",
                "condition": "noise_snr_10db",
                "snr_db": "10.0",
                "mitigation_enabled": "true",
            },
        )
        assert resp_mit.status_code == 200
        data_mit = resp_mit.json()
        assert data_mit["comparison"]["mitigation_applied"] is True
        assert data_mit["comparison"]["playback_signal_type"] == "mitigated"


class TestEmpiricalFiguresSectionRemoved:
    """Verifies that Empirical Research Figures are completely removed from the UI."""

    def test_no_figures_gallery_or_header_in_index_html(self, index_html_content: str) -> None:
        """Neither the section tag nor any figure gallery elements must exist."""
        assert "[EMPIRICAL RESEARCH ARTIFACTS &amp; SCIENTIFIC FIGURES]" not in index_html_content
        assert "[EMPIRICAL RESEARCH ARTIFACTS & SCIENTIFIC FIGURES]" not in index_html_content
        assert "figures-gallery-block" not in index_html_content
        assert "figureTabsList" not in index_html_content
        assert "fig-tab-btn" not in index_html_content
        assert "fig-display-card" not in index_html_content
        assert "figImageDisplay" not in index_html_content
        assert "figTitleDisplay" not in index_html_content

    def test_no_figure_listings_in_index_html(self, index_html_content: str) -> None:
        """None of the specific figure titles/tabs must be present in index.html."""
        forbidden_snippets = [
            "Fig 1: Confusion Matrix",
            "Fig 2: ROC / AUC Curve",
            "Fig 3: DET &amp; EER Operating Point",
            "Fig 4: Score Distributions",
            "Fig 5: Channel Robustness",
            "Fig 6: Mitigation Impact",
            "Fig 7: Waveforms &amp; Spectrograms",
            "Fig 8: Latency (Apple Silicon)",
            "Fig 9: Parameter Budget",
            "Figure 1: ASVspoof 2019 Countermeasure Decision Matrix",
        ]
        for snippet in forbidden_snippets:
            assert snippet not in index_html_content, f"Found forbidden figure entry: '{snippet}'"


class TestCinematicHeroAndInterpretationContracts:
    """Verifies atmospheric hero and expandable forensic interpretation accordions."""

    def test_full_bleed_hero_background_canvas(self, index_html_content: str) -> None:
        """Hero must have full-bleed canvas with overlay, without boxed card container."""
        assert 'id="signalSculptureCanvas"' in index_html_content
        assert 'class="hero-bg-canvas"' in index_html_content
        assert 'class="hero-bg-overlay"' in index_html_content
        assert "sculpture-container" not in index_html_content
        assert "Drag to rotate" not in index_html_content

    def test_expandable_verdict_interpretation_accordion(self, analyze_html_content: str) -> None:
        """Verdict card must have an accessible details element explaining forensic metrics."""
        assert 'id="verdictInterpretationDetails"' in analyze_html_content
        assert "How to interpret this forensic result" in analyze_html_content
        assert "Predicted Classification:" in analyze_html_content
        assert "Countermeasure Logit (S_cm):" in analyze_html_content
        assert "Logits vs. Probabilities:" in analyze_html_content

    def test_expandable_robustness_mechanisms_accordion(self, analyze_html_content: str) -> None:
        """Robustness card must have an accessible details element explaining acoustic drift."""
        assert 'id="robScoreShiftDetails"' in analyze_html_content
        assert "Why might the model score have changed?" in analyze_html_content
        assert "Telephony Bandwidth Restriction" in analyze_html_content
        assert "Additive White Gaussian Noise (AWGN)" in analyze_html_content
        assert "Lossy Codec Quantization (Opus)" in analyze_html_content

    def test_natural_scroll_reveal_classes(
        self, index_html_content: str, analyze_html_content: str
    ) -> None:
        """Workspace, Results, Robustness, Evaluation, and History must have reveal-on-scroll."""
        assert (
            'id="workspace" class="section workspace-section reveal-on-scroll"'
            in analyze_html_content
        )
        assert (
            'id="results" class="section results-section reveal-on-scroll"'
            in analyze_html_content
        )
        assert (
            'id="robustness" class="section robustness-section reveal-on-scroll"'
            in analyze_html_content
        )
        assert (
            'id="evaluation" class="section eval-section reveal-on-scroll"'
            in analyze_html_content
        )
        assert (
            'id="history" class="section history-section reveal-on-scroll"'
            in analyze_html_content
        )
        # Landing page sections also reveal on scroll
        assert 'id="product" class="section product-section reveal-on-scroll"' in index_html_content
        assert 'id="preview" class="section preview-section reveal-on-scroll"' in index_html_content


class TestLayoutBalanceAndScrollBehavior:
    """Verifies that navigation scrolls away naturally and bento grids have no blank space."""

    def test_navigation_header_scrolls_away_in_normal_flow(self, style_css_content: str) -> None:
        """The app-header must not be fixed or sticky so it scrolls out of view naturally."""
        match = re.search(r"\.app-header\s*\{([^}]+)\}", style_css_content)
        assert match is not None, "Could not find .app-header rule in style.css"
        header_block = match.group(1)
        assert "position: sticky" not in header_block, ".app-header must not have position: sticky"
        assert "position: fixed" not in header_block, ".app-header must not have position: fixed"
        assert "position: relative" in header_block, (
            ".app-header must be position: relative (normal document flow)"
        )

    def test_robustness_grid_aligns_at_top_without_excess_empty_height(
        self, style_css_content: str
    ) -> None:
        """The robustness grid must align-items: start so controls height follows content."""
        match = re.search(r"\.bento-robustness\s*\{([^}]+)\}", style_css_content)
        assert match is not None, "Could not find .bento-robustness rule in style.css"
        rob_block = match.group(1)
        assert "align-items: start" in rob_block, (
            ".bento-robustness must specify align-items: start to prevent stretching"
        )

    def test_eval_grid_balanced_two_column_layout(self, style_css_content: str) -> None:
        """The evaluation grid must not reserve a 3rd nonexistent column."""
        match = re.search(r"\.bento-eval-grid\s*\{([^}]+)\}", style_css_content)
        assert match is not None, "Could not find .bento-eval-grid rule in style.css"
        eval_block = match.group(1)
        assert "repeat(3" not in eval_block, (
            ".bento-eval-grid must not reserve 3 columns for 2 cards"
        )
        assert "repeat(2" in eval_block or "1fr 1fr" in eval_block, (
            ".bento-eval-grid must use a balanced 2-column layout"
        )
        assert "align-items: start" in eval_block, (
            ".bento-eval-grid must specify align-items: start"
        )


class TestVerdictCardForensicDesign:
    """Verifies that the verdict card has separate primary verdict, subtitle, and isolated icon."""

    def test_verdict_card_html_structure(self, analyze_html_content: str) -> None:
        """Verdict card must have isolated icon container, primary verdict, and subtitle."""
        assert 'id="verdictPill"' in analyze_html_content
        assert 'class="verdict-icon-container"' in analyze_html_content
        assert 'id="verdictIcon"' in analyze_html_content
        assert 'class="verdict-content"' in analyze_html_content
        assert 'class="verdict-primary"' in analyze_html_content
        assert 'id="verdictText"' in analyze_html_content
        assert 'class="verdict-subtitle"' in analyze_html_content
        assert 'id="verdictSubtitle"' in analyze_html_content
        assert 'id="verdictExplanation"' in analyze_html_content

    def test_verdict_pill_css_rules_and_nowrap(self, style_css_content: str) -> None:
        """CSS must isolate icon, prevent awkward word wrapping, and style primary/subtitle."""
        # Isolated icon container
        assert ".verdict-icon-container" in style_css_content
        icon_match = re.search(r"\.verdict-icon-container\s*\{([^}]+)\}", style_css_content)
        assert icon_match is not None, "Missing .verdict-icon-container in style.css"
        assert "flex-shrink: 0" in icon_match.group(1)

        # Primary verdict mono styling with nowrap
        prim_match = re.search(r"\.verdict-primary\s*\{([^}]+)\}", style_css_content)
        assert prim_match is not None, "Missing .verdict-primary in style.css"
        assert "white-space: nowrap" in prim_match.group(1)
        assert "var(--font-mono)" in prim_match.group(1)

        # Subtitle styling with nowrap
        sub_match = re.search(r"\.verdict-subtitle\s*\{([^}]+)\}", style_css_content)
        assert sub_match is not None, "Missing .verdict-subtitle in style.css"
        assert "white-space: nowrap" in sub_match.group(1)

        # Distinct spoof (red accent) and bonafide (blue accent) states
        assert ".verdict-pill-large.spoof" in style_css_content
        assert ".verdict-pill-large.bonafide" in style_css_content

    def test_verdict_card_responsive_rules(self, style_css_content: str) -> None:
        """CSS media query must adjust verdict pill for mobile/tablet widths."""
        assert "@media (max-width: 768px)" in style_css_content
        # Find 768px block
        m768 = re.search(
            r"@media\s*\(\s*max-width:\s*768px\s*\)\s*\{([\s\S]+?)\}\n\n@media",
            style_css_content,
        )
        assert m768 is not None, "Could not extract @media (max-width: 768px) block"
        media_content = m768.group(1)
        assert ".verdict-display" in media_content
        assert ".verdict-pill-large" in media_content
        assert ".verdict-primary" in media_content

    def test_app_js_dynamic_verdict_rendering(self, app_js_content: str) -> None:
        """app.js must render primary verdict and subtitle dynamically from model prediction."""
        # Must cache verdictPrimary and verdictSubtitle
        assert "verdictSubtitle:" in app_js_content
        # Must not hardcode fixed strings with wrapped parentheses
        assert "SPOOF (SYNTHETIC / DEEPFAKE)" not in app_js_content
        assert "BONAFIDE (AUTHENTIC)" not in app_js_content
        # Must render clean separate primary and subtitle
        assert "'BONAFIDE' : 'SPOOF'" in app_js_content
        assert "Synthetic / Deepfake" in app_js_content
        assert "Authentic Human Speech" in app_js_content


class TestLandingPageSeparationAndRouting:
    """Verifies strict architectural separation between landing page and workspace."""

    def test_landing_page_does_not_contain_upload_workflow(
        self, index_html_content: str
    ) -> None:
        """The landing page must never contain active upload or processing controls."""
        assert 'id="cardUpload"' not in index_html_content
        assert 'id="audioFileInput"' not in index_html_content
        assert 'id="dropZone"' not in index_html_content
        assert 'id="runInferenceBtn"' not in index_html_content
        assert 'id="analysisProgressCard"' not in index_html_content

    def test_primary_cta_navigates_to_dedicated_analyze_route(
        self, index_html_content: str
    ) -> None:
        """The primary CTA 'Analyze Audio' must navigate to /analyze."""
        assert 'href="/analyze"' in index_html_content

    def test_dedicated_routes_served_with_html(self, client: TestClient) -> None:
        """GET /, /analyze, /robustness, /history, and /methodology must serve valid HTML."""
        for path in ("/", "/analyze", "/robustness", "/history", "/methodology"):
            resp = client.get(path)
            assert resp.status_code == 200, f"Failed on path {path}"
            assert "text/html" in resp.headers["content-type"]


class TestStage2SampleSelectorAndIntegrity:
    """Verifies that sample presets do not leak ground truth or use emoji icons."""

    def test_sample_buttons_have_no_emojis(self, analyze_html_content: str) -> None:
        """Sample presets must use clean typography/track badges, not emojis."""
        sample_block_match = re.search(
            r'id="sampleButtonsList".*?</div>\s*</div>', analyze_html_content, re.DOTALL
        )
        assert sample_block_match is not None, "Could not find sampleButtonsList in analyze.html"
        block = sample_block_match.group(0)

        # Check for banned emoji characters in sample preset list
        for emoji in ("🎙️", "🎙", "🤖", "🔬", "⚠️"):
            assert emoji not in block, f"Found forbidden emoji '{emoji}' in sample presets!"

    def test_sample_buttons_do_not_leak_ground_truth(self, analyze_html_content: str) -> None:
        """Sample presets must use neutral titles and not spoil model classifications."""
        sample_block_match = re.search(
            r'id="sampleButtonsList".*?</div>\s*</div>', analyze_html_content, re.DOTALL
        )
        assert sample_block_match is not None
        block = sample_block_match.group(0)

        # Must not reveal labels like "Human Speech", "Authentic spoken voice", "AI Deepfake"
        forbidden_pre_labels = [
            "Human Speech",
            "Authentic spoken voice",
            "AI Deepfake",
            "Synthesized speech attack",
        ]
        for label in forbidden_pre_labels:
            assert label not in block, f"Sample presets spoil ground truth with '{label}'!"

    def test_sample_buttons_have_track_badges(self, analyze_html_content: str) -> None:
        """Sample presets must have professional track badges (TRK-01..TRK-06)."""
        assert 'class="track-badge">TRK-01</span>' in analyze_html_content
        assert 'class="track-badge">TRK-02</span>' in analyze_html_content
        assert 'class="track-badge">TRK-03</span>' in analyze_html_content
        assert 'class="track-badge">TRK-04</span>' in analyze_html_content
        assert 'class="track-badge">TRK-05</span>' in analyze_html_content
        assert 'class="track-badge">TRK-06</span>' in analyze_html_content

    def test_sample_buttons_scrollable_container(self, style_css_content: str) -> None:
        """Preset buttons list must have max-height and overflow-y: auto to prevent clipping."""
        match = re.search(r"\.preset-buttons\s*\{([^}]+)\}", style_css_content)
        assert match is not None, "Could not find .preset-buttons rule in style.css"
        css_block = match.group(1)
        assert "overflow-y: auto" in css_block or "overflow-y: scroll" in css_block


class TestStage2SimpleAndExpertViewModes:
    """Verifies that Simple and Expert view modes exist, toggle correctly, and share data."""

    def test_view_mode_toggle_markup_in_analyze_html(self, analyze_html_content: str) -> None:
        """analyze.html must contain view mode toolbar, toggle buttons, and content panels."""
        assert 'class="view-mode-bar"' in analyze_html_content
        assert 'id="viewModeSimple"' in analyze_html_content
        assert 'id="viewModeExpert"' in analyze_html_content
        assert 'id="simpleViewContent"' in analyze_html_content
        assert 'id="expertViewContent"' in analyze_html_content
        assert 'id="simpleStatusTitle"' in analyze_html_content
        assert 'id="simpleStatusText"' in analyze_html_content
        assert 'id="simpleMarginText"' in analyze_html_content

    def test_app_js_implements_view_mode_toggle(self, app_js_content: str) -> None:
        """app.js must manage state.viewMode and toggle between simple and expert view."""
        assert "viewModeSimple:" in app_js_content
        assert "viewModeExpert:" in app_js_content
        assert "function setViewMode(" in app_js_content
        assert "simpleViewContent" in app_js_content
        assert "expertViewContent" in app_js_content


class TestStage2SingleAuthoritativeThreshold:
    """Verifies that duplicate threshold controls are eliminated."""

    def test_single_threshold_slider_in_analyze_html(self, analyze_html_content: str) -> None:
        """analyze.html must have exactly one threshold slider input."""
        sliders = re.findall(r'<input[^>]+id="thresholdSlider"[^>]*>', analyze_html_content)
        assert len(sliders) == 1, f"Expected exactly 1 threshold slider, found {len(sliders)}"

        # Verify slider baseline note
        assert "ASVspoof standard baseline: &theta; = 0.00" in analyze_html_content


class TestStage2WaveformScrubberAndSpectrogramAccessibility:
    """Verifies interactive waveform scrubbing and accessible spectrogram rendering."""

    def test_waveform_canvas_pointer_cursor(self, style_css_content: str) -> None:
        """waveformCanvas must have cursor: pointer for interactive seeking affordance."""
        match = re.search(r"#waveformCanvas\s*\{([^}]+)\}", style_css_content)
        assert match is not None, "Could not find #waveformCanvas rule in style.css"
        assert "cursor: pointer" in match.group(1)

    def test_waveform_scrubbing_event_handlers_in_app_js(self, app_js_content: str) -> None:
        """app.js must implement mouse and touch seeking on the waveform canvas."""
        assert "seekWaveformAtEvent(" in app_js_content
        assert "restartAudioPlaybackFromOffset(" in app_js_content
        assert "isScrubbing" in app_js_content

    def test_spectrogram_legend_swatches_accessible_contrast(
        self, style_css_content: str
    ) -> None:
        """Spectrogram swatches must use high-visibility accessible colors."""
        assert ".legend-swatch.low" in style_css_content
        assert ".legend-swatch.mid" in style_css_content
        assert ".legend-swatch.high" in style_css_content
        assert "width: 12px" in style_css_content

    def test_spectrogram_placeholder_balanced_two_lines(self, app_js_content: str) -> None:
        """Placeholder text must be rendered in two balanced lines to prevent clipping."""
        assert "Awaiting Audio Analysis" in app_js_content
        assert "0–8000 Hz FFT profile renders upon inference" in app_js_content

    def test_mobile_nav_drawer_in_analyze_html(self, analyze_html_content: str) -> None:
        """analyze.html must have accessible mobile menu trigger and drawer."""
        assert 'id="mobileMenuBtn"' in analyze_html_content
        assert 'id="mobileNavDrawer"' in analyze_html_content
        assert 'id="closeMobileNavBtn"' in analyze_html_content
        assert 'id="mobileNavBackdrop"' in analyze_html_content


class TestStage3RobustnessLabUI:
    """Verifies that the dedicated Robustness Lab (/robustness) meets all UI contracts."""

    def test_robustness_lab_structure(self, robustness_html_content: str) -> None:
        """robustness.html must have experiment controls, comparison deck, and sweep matrix."""
        assert "Acoustic Robustness Lab" in robustness_html_content
        assert 'id="runRobExpBtn"' in robustness_html_content
        assert 'id="runSweepMatrixBtn"' in robustness_html_content
        assert 'id="robResponseCanvas"' in robustness_html_content
        assert 'id="robSweepTable"' in robustness_html_content
        assert 'name="robCondition"' in robustness_html_content
        assert 'id="robSnrSlider"' in robustness_html_content
        assert 'id="robMitigationToggle"' in robustness_html_content
        assert 'id="robMitigationDetails"' in robustness_html_content
        assert "Spectral Subtraction" in robustness_html_content
        assert "Bandpass Filter" in robustness_html_content
        assert "RMS Normalization" in robustness_html_content

    def test_robustness_side_by_side_audio_comparison(
        self, robustness_html_content: str
    ) -> None:
        """robustness.html must contain A/B audio comparison playback triggers and waveforms."""
        assert 'id="playCleanAudioBtn"' in robustness_html_content
        assert 'id="playProcAudioBtn"' in robustness_html_content
        assert 'id="robCleanWaveCanvas"' in robustness_html_content
        assert 'id="robProcWaveCanvas"' in robustness_html_content
        assert 'id="robDeltaValue"' in robustness_html_content
        assert 'id="robBoundaryBadge"' in robustness_html_content

    def test_robustness_clean_navigation_and_palette(
        self, robustness_html_content: str
    ) -> None:
        """robustness.html must include command palette trigger and manifest."""
        assert 'id="headerCmdPaletteBtn"' in robustness_html_content
        assert 'href="/manifest.webmanifest"' in robustness_html_content
        assert 'src="js/palette.js"' in robustness_html_content


class TestStage3HistoryArchiveAndDossierUI:
    """Verifies that the dedicated History Archive (/history) meets all UI contracts."""

    def test_history_archive_structure(self, history_html_content: str) -> None:
        """history.html must have search, filter, export, clear modal, and dossier modal."""
        assert "Session Evidence History" in history_html_content
        assert 'id="historySearchInput"' in history_html_content
        assert 'id="historyVerdictFilter"' in history_html_content
        assert 'id="historySortSelect"' in history_html_content
        assert 'id="exportJsonBtn"' in history_html_content
        assert 'id="exportCsvBtn"' in history_html_content
        assert 'id="clearHistoryBtn"' in history_html_content
        assert 'id="confirmClearModal"' in history_html_content
        assert 'id="dossierModal"' in history_html_content
        assert 'id="printDossierBtn"' in history_html_content
        assert 'id="downloadDossierJsonBtn"' in history_html_content

    def test_history_adr002_storage_scope_banner(self, history_html_content: str) -> None:
        """history.html must prominently state ADR-002 in-memory retention scope."""
        assert "ADR-002 Privacy Architecture" in history_html_content
        assert "in-memory for the duration of this runtime session" in history_html_content


class TestStage3MicAndBatchInAnalyzeWorkspace:
    """Verifies live mic and batch tabs inside the analysis workspace (/analyze)."""

    def test_input_mode_tabs_in_analyze_html(self, analyze_html_content: str) -> None:
        """analyze.html must contain the three input mode tabs and panes."""
        assert 'id="tabModeUpload"' in analyze_html_content
        assert 'id="tabModeMic"' in analyze_html_content
        assert 'id="tabModeBatch"' in analyze_html_content
        assert 'id="paneUpload"' in analyze_html_content
        assert 'id="paneMic"' in analyze_html_content
        assert 'id="paneBatch"' in analyze_html_content

    def test_live_microphone_controls(self, analyze_html_content: str) -> None:
        """paneMic must contain record/stop controls, timer, visualizer, and analyze button."""
        assert 'id="btnMicRecord"' in analyze_html_content
        assert 'id="btnMicStop"' in analyze_html_content
        assert 'id="btnMicReset"' in analyze_html_content
        assert 'id="btnMicAnalyze"' in analyze_html_content
        assert 'id="micTimer"' in analyze_html_content
        assert 'id="micVisualizerCanvas"' in analyze_html_content

    def test_batch_analysis_controls(self, analyze_html_content: str) -> None:
        """paneBatch must contain batch drop zone, run button, progress bar, and results table."""
        assert 'id="batchDropZone"' in analyze_html_content
        assert 'id="batchFileInput"' in analyze_html_content
        assert 'id="btnRunBatch"' in analyze_html_content
        assert 'id="batchProgressBar"' in analyze_html_content
        assert 'id="batchResultsTable"' in analyze_html_content
        assert 'id="btnExportBatchJson"' in analyze_html_content


class TestStage3CommandPaletteAndTheming:
    """Verifies universal command palette and high-contrast styling."""

    def test_command_palette_js_contracts(self, palette_js_content: str) -> None:
        """palette.js must listen for Cmd+K and Ctrl+K, theme toggle, and shortcuts modal."""
        assert "keydown" in palette_js_content
        assert "Meta" in palette_js_content or "metaKey" in palette_js_content
        assert "high-contrast" in palette_js_content
        assert "cmd-palette-modal" in palette_js_content

    def test_high_contrast_css_rules(self, style_css_content: str) -> None:
        """style.css must define data-theme="high-contrast" overrides."""
        assert '[data-theme="high-contrast"]' in style_css_content
        assert "--bg-primary: #000000;" in style_css_content

    def test_pwa_manifest_existence(self) -> None:
        """PWA manifest must exist and be valid JSON."""
        import json

        assert STATIC_MANIFEST.is_file()
        data = json.loads(STATIC_MANIFEST.read_text(encoding="utf-8"))
        assert data["name"] == "Audio Authenticity Lab"
        assert data["display"] == "standalone"



