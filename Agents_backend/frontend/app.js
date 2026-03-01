/**
 * AI Content Factory — Frontend Application
 * WebSocket real-time agent visualization + 3D particle background
 */

// ============================================================================
// CONFIG
// ============================================================================
const API_BASE = window.location.origin;
const WS_BASE = `ws://${window.location.host}`;

// API key — read from the form or use a default for development
let API_KEY = '';

// ============================================================================
// STATE
// ============================================================================
let currentJobId = null;
let ws = null;
let timerInterval = null;
let startTime = null;

// ============================================================================
// DOM REFERENCES
// ============================================================================
const $ = id => document.getElementById(id);

const dom = {
    form: $('generate-form'),
    topicInput: $('topic-input'),
    topicSuggestions: $('topic-suggestions'),
    btnSuggest: $('btn-suggest'),
    toneSelect: $('tone-select'),
    audienceInput: $('audience-input'),
    includeImages: $('include-images'),
    autoApprove: $('auto-approve'),
    btnGenerate: $('btn-generate'),
    inputSection: $('input-section'),
    pipelineSection: $('pipeline-section'),
    pipelineStatusBadge: $('pipeline-status-badge'),
    timeElapsed: $('time-elapsed'),
    logFeed: $('log-feed'),
    metricsRow: $('metrics-row'),
    resultsSection: $('results-section'),
    resultsSummary: $('results-summary'),
    btnDownload: $('btn-download'),
    btnNew: $('btn-new'),
    blogPreview: $('blog-preview'),
    serverStatus: $('server-status'),
    serverStatusText: $('server-status-text'),
    // Metrics
    metricSources: $('metric-sources'),
    metricSections: $('metric-sections'),
    metricWords: $('metric-words'),
    metricFactcheck: $('metric-factcheck'),
    metricQuality: $('metric-quality'),
};

// ============================================================================
// TOPIC SUGGESTIONS
// ============================================================================
async function fetchTopicSuggestions() {
    if (!dom.topicSuggestions) return;

    // Render skeleton loaders (4 pills)
    dom.topicSuggestions.innerHTML = Array(4).fill('<div class="suggestion-skeleton"></div>').join('');

    try {
        const res = await fetch(`${API_BASE}/api/suggest`);
        if (res.ok) {
            const data = await res.json();
            const suggestions = data.suggestions || [];

            dom.topicSuggestions.innerHTML = ''; // Clear skeletons

            suggestions.forEach(topic => {
                const pill = document.createElement('div');
                pill.className = 'suggestion-pill';
                pill.textContent = topic;
                pill.onclick = () => {
                    dom.topicInput.value = topic;
                    // Optional: Add a brief pulse effect to the input
                    dom.topicInput.parentElement.classList.add('pulse');
                    setTimeout(() => dom.topicInput.parentElement.classList.remove('pulse'), 300);
                };
                dom.topicSuggestions.appendChild(pill);
            });
        }
    } catch (e) {
        console.error("Failed to fetch topic suggestions:", e);
        dom.topicSuggestions.innerHTML = ''; // Silently fail UI
    }
}

// ============================================================================
// SERVER HEALTH CHECK
// ============================================================================
async function checkServerHealth() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        if (res.ok) {
            dom.serverStatus.classList.add('connected');
            dom.serverStatusText.textContent = 'Server Online';
            return true;
        }
    } catch (e) { }
    dom.serverStatus.classList.remove('connected');
    dom.serverStatusText.textContent = 'Server Offline';
    return false;
}

// ============================================================================
// API KEY PROMPT
// ============================================================================
function getApiKey() {
    if (API_KEY) return API_KEY;
    // Try to read from localStorage
    API_KEY = localStorage.getItem('acf_api_key') || '';
    if (!API_KEY) {
        API_KEY = prompt('Enter your API key (X-API-Key):', '') || '';
        if (API_KEY) localStorage.setItem('acf_api_key', API_KEY);
    }
    return API_KEY;
}

// ============================================================================
// GENERATE BLOG
// ============================================================================
async function generateBlog(topic, settings) {
    const key = getApiKey();
    if (!key) {
        addLogEntry('pipeline', 'error', 'API key is required');
        return null;
    }

    try {
        const res = await fetch(`${API_BASE}/api/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': key,
            },
            body: JSON.stringify({
                topic,
                auto_approve: settings.autoApprove,
                include_images: settings.includeImages,
                tone: settings.tone || null,
                audience: settings.audience || null,
            }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Generation failed');
        }

        return await res.json();
    } catch (e) {
        addLogEntry('pipeline', 'error', e.message);
        return null;
    }
}

// ============================================================================
// WEBSOCKET — REAL-TIME EVENTS
// ============================================================================
function connectWebSocket(jobId) {
    if (ws) {
        ws.close();
        ws = null;
    }

    const url = `${WS_BASE}/ws/jobs/${jobId}`;
    ws = new WebSocket(url);

    ws.onopen = () => {
        document.querySelector('.live-dot').classList.add('active');
        addLogEntry('pipeline', 'started', 'Connected to real-time agent feed');
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'heartbeat') return;
            handleAgentEvent(data);
        } catch (e) {
            console.error('WS parse error:', e);
        }
    };

    ws.onclose = () => {
        document.querySelector('.live-dot').classList.remove('active');
    };

    ws.onerror = () => {
        // Will auto-fallback to polling
        startPolling(jobId);
    };
}

// ============================================================================
// POLLING FALLBACK
// ============================================================================
let pollInterval = null;

function startPolling(jobId) {
    if (pollInterval) return;
    pollInterval = setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE}/api/status/${jobId}`);
            const data = await res.json();

            if (data.status === 'completed') {
                clearInterval(pollInterval);
                pollInterval = null;
                handleJobComplete(data);
            } else if (data.status === 'failed') {
                clearInterval(pollInterval);
                pollInterval = null;
                handleJobFailed(data);
            }
        } catch (e) { }
    }, 3000);
}

// ============================================================================
// AGENT EVENT HANDLER
// ============================================================================
function handleAgentEvent(event) {
    const { agent_name, status, message, metrics } = event;

    // Update log feed
    addLogEntry(agent_name, status, message);

    // Update pipeline node
    updateNodeState(agent_name, status, message);

    // Update connector states
    updateConnectors();

    // Update metrics
    if (metrics) {
        if (metrics.sources !== undefined) updateMetric('metricSources', metrics.sources);
        if (metrics.sections !== undefined) updateMetric('metricSections', metrics.sections);
        if (metrics.words !== undefined) updateMetric('metricWords', metrics.words);
        if (metrics.score !== undefined) {
            if (agent_name === 'fact_checker') updateMetric('metricFactcheck', `${metrics.score}/10`);
            if (agent_name === 'evaluator') updateMetric('metricQuality', `${metrics.score}/10`);
        }
    }

    // Pipeline complete
    if (agent_name === 'pipeline' && status === 'completed') {
        dom.pipelineStatusBadge.textContent = 'Complete';
        dom.pipelineStatusBadge.className = 'pipeline-status completed';
        stopTimer();
        updateNodeState('done', 'completed', '');

        // Fetch final results
        setTimeout(() => fetchResults(currentJobId), 1500);
    }
}

// ============================================================================
// NODE STATE MANAGEMENT
// ============================================================================
const nodeStates = {};

function updateNodeState(agentName, status, message) {
    const node = $(`node-${agentName}`);
    if (!node) return;

    // Map status to CSS class
    let cssClass = '';
    if (status === 'started' || status === 'working') cssClass = 'active';
    else if (status === 'completed') cssClass = 'completed';
    else if (status === 'error') cssClass = 'error';

    // Remove old state
    node.classList.remove('inactive', 'active', 'completed', 'error');
    node.classList.add(cssClass);

    // Update status text
    const statusEl = node.querySelector('.node-status');
    if (statusEl) {
        statusEl.textContent = message.length > 25 ? message.substring(0, 22) + '...' : message;
    }

    // Track state
    nodeStates[agentName] = status;
}

function resetAllNodes() {
    document.querySelectorAll('.agent-node').forEach(n => {
        n.classList.remove('active', 'completed', 'error');
        n.classList.add('inactive');
        const s = n.querySelector('.node-status');
        if (s) s.textContent = '';
    });
    document.querySelectorAll('.connector').forEach(c => {
        c.classList.remove('active', 'completed');
    });
    Object.keys(nodeStates).forEach(k => delete nodeStates[k]);
}

// ============================================================================
// CONNECTOR LOGIC
// ============================================================================
const pipelineOrder = [
    ['pipeline', 'router', 'research'],
    ['orchestrator', 'writer', 'merger'],
    ['images', 'fact_checker', 'revision'],
    ['social_media', 'evaluator', 'done'],
];

function updateConnectors() {
    const rows = document.querySelectorAll('.pipeline-row');
    rows.forEach((row, rowIdx) => {
        const connectors = row.querySelectorAll('.connector');
        const agents = pipelineOrder[rowIdx] || [];

        connectors.forEach((conn, connIdx) => {
            const leftAgent = agents[connIdx];
            const rightAgent = agents[connIdx + 1];

            const leftDone = nodeStates[leftAgent] === 'completed';
            const rightActive = ['started', 'working', 'completed'].includes(nodeStates[rightAgent]);

            conn.classList.remove('active', 'completed');
            if (leftDone && rightActive) {
                conn.classList.add(nodeStates[rightAgent] === 'completed' ? 'completed' : 'active');
            } else if (leftDone) {
                conn.classList.add('active');
            }
        });
    });
}

// ============================================================================
// LOG FEED
// ============================================================================
function addLogEntry(agentName, status, message) {
    const now = new Date();
    const time = now.toLocaleTimeString('en-US', { hour12: false });

    const statusIcons = {
        started: '▶️',
        working: '⚙️',
        completed: '✅',
        error: '❌',
    };

    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-status-icon">${statusIcons[status] || '📌'}</span>
        <span class="log-agent ${agentName}">${agentName.replace('_', ' ')}</span>
        <span class="log-msg">${escapeHtml(message)}</span>
    `;

    dom.logFeed.appendChild(entry);
    dom.logFeed.scrollTop = dom.logFeed.scrollHeight;
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ============================================================================
// METRICS
// ============================================================================
function updateMetric(metricKey, value) {
    const el = dom[metricKey];
    if (!el) return;
    el.textContent = value;
    el.classList.add('animate-update');
    setTimeout(() => el.classList.remove('animate-update'), 400);
}

// ============================================================================
// TIMER
// ============================================================================
function startTimer() {
    startTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
        const secs = (elapsed % 60).toString().padStart(2, '0');
        dom.timeElapsed.textContent = `${mins}:${secs}`;
    }, 1000);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

// ============================================================================
// FETCH RESULTS
// ============================================================================
async function fetchResults(jobId) {
    try {
        const res = await fetch(`${API_BASE}/api/status/${jobId}`);
        const data = await res.json();

        if (data.status === 'completed' && data.result) {
            handleJobComplete(data);
        } else if (data.status === 'failed') {
            handleJobFailed(data);
        } else {
            // Not done yet, retry
            setTimeout(() => fetchResults(jobId), 3000);
        }
    } catch (e) {
        console.error('Fetch results error:', e);
        setTimeout(() => fetchResults(jobId), 5000);
    }
}

function handleJobComplete(data) {
    const result = data.result || {};

    // Show results section
    dom.resultsSection.style.display = 'block';
    dom.resultsSection.classList.add('celebrate');

    // Build summary stats
    // Stats UI Update
    dom.resultsSummary.innerHTML = `
        <div class="result-stat">
            <div class="result-stat-value">${result.word_count || '—'}</div>
            <div class="result-stat-label">Words</div>
        </div>
        <div class="result-stat">
            <div class="result-stat-value">${result.quality_score || '—'}/10</div>
            <div class="result-stat-label">Quality Score</div>
        </div>
        <div class="result-stat">
            <div class="result-stat-value">${result.fact_check_score || '—'}/10</div>
            <div class="result-stat-label">Fact-Check Score</div>
        </div>
        <div class="result-stat">
            <div class="result-stat-value">${(result.files?.assets || []).length}</div>
            <div class="result-stat-label">Campaign Assets</div>
        </div>
    `;

    // Download button
    dom.btnDownload.onclick = () => {
        window.open(`${API_BASE}/api/download/${data.job_id}`, '_blank');
    };

    // Render Campaign Assets
    const renderAsset = (id, content) => {
        const el = document.getElementById(id);
        if (el) {
            el.innerHTML = content ? marked.parse(content) : '<p class="suggest-empty">Content not generated.</p>';
        }
    };

    renderAsset('blog-preview', result.content);
    renderAsset('email-preview', result.email_sequence);
    renderAsset('twitter-preview', result.twitter_thread);
    renderAsset('landing-preview', result.landing_page);
    renderAsset('linkedin-preview', result.linkedin_post);
    renderAsset('facebook-preview', result.facebook_post);
    renderAsset('youtube-preview', result.youtube_script);

    // Setup Tab Switching Logic
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.campaign-content');

    tabBtns.forEach(btn => {
        btn.onclick = () => {
            // Remove active from all
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Add active to clicked
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
        };
    });

    // Scroll to results
    dom.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function handleJobFailed(data) {
    dom.pipelineStatusBadge.textContent = 'Failed';
    dom.pipelineStatusBadge.className = 'pipeline-status error';
    stopTimer();
    addLogEntry('pipeline', 'error', data.error || 'Generation failed');
}

// ============================================================================
// FORM SUBMISSION
// ============================================================================
dom.form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const topic = dom.topicInput.value.trim();
    if (!topic) return;

    const settings = {
        autoApprove: dom.autoApprove.checked,
        includeImages: dom.includeImages.checked,
        tone: dom.toneSelect.value,
        audience: dom.audienceInput.value.trim(),
    };

    // Disable form
    dom.btnGenerate.disabled = true;
    dom.btnGenerate.querySelector('.btn-text').textContent = 'Starting...';

    // Show pipeline
    dom.pipelineSection.style.display = 'block';
    dom.resultsSection.style.display = 'none';
    dom.pipelineStatusBadge.textContent = 'Running';
    dom.pipelineStatusBadge.className = 'pipeline-status running';
    dom.logFeed.innerHTML = '';

    // Reset nodes
    resetAllNodes();

    // Reset metrics
    dom.metricSources.textContent = '0';
    dom.metricSections.textContent = '0';
    dom.metricWords.textContent = '0';
    dom.metricFactcheck.textContent = '—';
    dom.metricQuality.textContent = '—';

    // Scroll to pipeline
    dom.pipelineSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Start timer
    startTimer();

    // Call API
    const result = await generateBlog(topic, settings);
    if (!result || !result.job_id) {
        dom.btnGenerate.disabled = false;
        dom.btnGenerate.querySelector('.btn-text').textContent = 'Generate Blog';
        dom.pipelineStatusBadge.textContent = 'Failed';
        dom.pipelineStatusBadge.className = 'pipeline-status error';
        stopTimer();
        return;
    }

    currentJobId = result.job_id;
    addLogEntry('pipeline', 'started', `Job created: ${result.job_id.substring(0, 8)}...`);

    // Connect WebSocket
    connectWebSocket(currentJobId);

    // Also start polling as fallback
    startPolling(currentJobId);

    // Re-enable button
    dom.btnGenerate.disabled = false;
    dom.btnGenerate.querySelector('.btn-text').textContent = 'Generate Blog';
});

// ============================================================================
// NEW BLOG BUTTON
// ============================================================================
dom.btnNew.addEventListener('click', () => {
    // Reset everything
    currentJobId = null;
    if (ws) { ws.close(); ws = null; }
    if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
    stopTimer();

    dom.pipelineSection.style.display = 'none';
    dom.resultsSection.style.display = 'none';
    dom.topicInput.value = '';
    dom.topicInput.focus();
    dom.timeElapsed.textContent = '00:00';

    window.scrollTo({ top: 0, behavior: 'smooth' });
});

// ============================================================================
// INIT
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    checkServerHealth();
    setInterval(checkServerHealth, 15000);

    // Fetch trending topic suggestions
    fetchTopicSuggestions();

    // Focus on topic input
    dom.topicInput.focus();
});

// ============================================================================
// TITLE SUGGESTIONS (Manual Button)
// ============================================================================
const btnSuggest = $('btn-suggest');
const suggestionsPanel = $('suggestions-panel');
const suggestionsLoading = $('suggestions-loading');
const suggestionsList = $('suggestions-list');
const suggestionsClose = $('suggestions-close');

const TONE_BADGE_COLORS = {
    professional: '#3b82f6',
    conversational: '#22c55e',
    technical: '#a855f7',
    educational: '#f59e0b',
    persuasive: '#ef4444',
    inspirational: '#ec4899',
};

btnSuggest.addEventListener('click', async () => {
    const topic = dom.topicInput.value.trim();
    if (!topic) {
        dom.topicInput.focus();
        dom.topicInput.placeholder = '⚠️ Enter a topic first...';
        setTimeout(() => { dom.topicInput.placeholder = 'e.g., The Future of Quantum Computing in 2026'; }, 2000);
        return;
    }

    // Show panel + loading
    suggestionsPanel.style.display = 'block';
    suggestionsLoading.style.display = 'flex';
    suggestionsList.innerHTML = '';
    btnSuggest.disabled = true;
    btnSuggest.textContent = '⏳ Generating...';

    try {
        const res = await fetch(`${API_BASE}/api/suggest`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ topic }),
        });
        const data = await res.json();
        const suggestions = data.suggestions || [];

        suggestionsLoading.style.display = 'none';

        if (suggestions.length === 0) {
            suggestionsList.innerHTML = '<p class="suggest-empty">No suggestions returned. Try a more specific topic.</p>';
            return;
        }

        suggestions.forEach(s => {
            const color = TONE_BADGE_COLORS[s.tone] || '#64748b';
            const card = document.createElement('div');
            card.className = 'suggestion-card';
            card.innerHTML = `
                <div class="suggestion-title">${escapeHtml(s.title)}</div>
                <div class="suggestion-meta">
                    <span class="suggestion-angle">${escapeHtml(s.angle)}</span>
                    <span class="suggestion-tone" style="background:${color}22;color:${color};border:1px solid ${color}44">${s.tone}</span>
                </div>
            `;
            card.addEventListener('click', () => {
                dom.topicInput.value = s.title;
                // Auto-select the tone if it matches an option
                const toneOpts = [...dom.toneSelect.options].map(o => o.value);
                if (toneOpts.includes(s.tone)) dom.toneSelect.value = s.tone;
                suggestionsPanel.style.display = 'none';
                dom.topicInput.focus();
            });
            suggestionsList.appendChild(card);
        });
    } catch (e) {
        suggestionsLoading.style.display = 'none';
        suggestionsList.innerHTML = `<p class="suggest-empty">Error: ${escapeHtml(e.message)}</p>`;
    } finally {
        btnSuggest.disabled = false;
        btnSuggest.textContent = '✨ Get Title Suggestions';
    }
});

suggestionsClose.addEventListener('click', () => {
    suggestionsPanel.style.display = 'none';
});

