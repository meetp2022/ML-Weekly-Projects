// API Configuration (Apex Domain Trigger: aichecking.me)
const API_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? ''
    : 'https://api.aichecking.me'; // Custom Domain Backend

// DOM Elements
const textInput = document.getElementById('textInput');
const charCount = document.getElementById('charCount');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const loadingOverlay = document.getElementById('loadingOverlay');
const statusBadge = document.getElementById('statusBadge');

// Result elements
const scoreValue = document.getElementById('scoreValue');
const scoreLabel = document.getElementById('scoreLabel');
const scoreExplanation = document.getElementById('scoreExplanation');
const scoreConfidence = document.getElementById('scoreConfidence');
const confidenceExplanation = document.getElementById('confidenceExplanation');
const scoreRingFill = document.getElementById('scoreRingFill');
const perplexityValue = document.getElementById('perplexityValue');
const perplexityBar = document.getElementById('perplexityBar');
const burstinessValue = document.getElementById('burstinessValue');
const burstinessBar = document.getElementById('burstinessBar');
const repetitionValue = document.getElementById('repetitionValue');
const repetitionBar = document.getElementById('repetitionBar');
const highlightedText = document.getElementById('highlightedText');

// Add SVG gradient for score ring
const svg = document.querySelector('.score-ring');
const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
gradient.setAttribute('id', 'scoreGradient');
gradient.innerHTML = `
    <stop offset="0%" stop-color="#667eea"/>
    <stop offset="100%" stop-color="#764ba2"/>
`;
defs.appendChild(gradient);
svg.appendChild(defs);

// Event Listeners
textInput.addEventListener('input', updateCharCount);
analyzeBtn.addEventListener('click', analyzeText);

// Update character count
function updateCharCount() {
    const length = textInput.value.length;
    charCount.textContent = `${length.toLocaleString()} / 10,000`;

    // Enable/disable analyze button
    analyzeBtn.disabled = length < 10;
}

// Analyze text
async function analyzeText() {
    const text = textInput.value.trim();

    if (text.length < 10) {
        alert('Please enter at least 10 characters');
        return;
    }

    // Show loading
    loadingOverlay.style.display = 'flex';
    analyzeBtn.disabled = true;
    updateStatus('Analyzing...', 'warning');

    try {
        const response = await fetch(`${API_URL}/api/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Analysis failed');
        }

        const data = await response.json();
        displayResults(data);
        updateStatus('Analysis complete', 'success');

    } catch (error) {
        console.error('Error:', error);
        alert(`Error: ${error.message}`);
        updateStatus('Error', 'danger');
    } finally {
        loadingOverlay.style.display = 'none';
        analyzeBtn.disabled = false;
    }
}

// Display results
function displayResults(data) {
    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Update score
    const score = data.score;
    scoreValue.textContent = Math.round(score);
    scoreLabel.textContent = data.label;
    scoreConfidence.textContent = `Confidence: ${data.confidence}`;

    // Add user-friendly explanations
    if (data.label === 'AI-generated') {
        scoreExplanation.textContent = 'This text appears to be written by AI. It shows patterns typical of machine-generated content.';
    } else if (data.label === 'Human-written') {
        scoreExplanation.textContent = 'This text appears to be written by a human. It shows natural writing patterns and variation.';
    } else {
        scoreExplanation.textContent = 'The results are inconclusive. The text has mixed characteristics of both AI and human writing.';
    }

    // Add confidence explanations
    if (data.confidence === 'high') {
        confidenceExplanation.textContent = 'The analysis is very confident in this classification.';
    } else if (data.confidence === 'medium') {
        confidenceExplanation.textContent = 'The analysis has moderate confidence. Some indicators are mixed.';
    } else {
        confidenceExplanation.textContent = 'The analysis has low confidence. Results should be interpreted cautiously.';
    }

    // Update score ring
    const circumference = 2 * Math.PI * 70;
    const offset = circumference - (score / 100) * circumference;
    scoreRingFill.style.strokeDashoffset = offset;

    // Update score color based on value
    const scoreGradient = document.getElementById('scoreGradient');
    if (score >= 70) {
        // High AI probability - red gradient
        scoreGradient.innerHTML = `
            <stop offset="0%" stop-color="#ef4444"/>
            <stop offset="100%" stop-color="#dc2626"/>
        `;
        scoreLabel.style.color = '#ef4444';
    } else if (score <= 30) {
        // Low AI probability - green gradient
        scoreGradient.innerHTML = `
            <stop offset="0%" stop-color="#10b981"/>
            <stop offset="100%" stop-color="#059669"/>
        `;
        scoreLabel.style.color = '#10b981';
    } else {
        // Uncertain - orange gradient
        scoreGradient.innerHTML = `
            <stop offset="0%" stop-color="#f59e0b"/>
            <stop offset="100%" stop-color="#d97706"/>
        `;
        scoreLabel.style.color = '#f59e0b';
    }

    // Update metrics
    const metrics = data.metrics;

    perplexityValue.textContent = metrics.perplexity.toFixed(2);
    perplexityBar.style.width = `${metrics.perplexity_score}%`;

    burstinessValue.textContent = metrics.burstiness.toFixed(3);
    burstinessBar.style.width = `${metrics.burstiness_score}%`;

    repetitionValue.textContent = metrics.repetition.toFixed(3);
    repetitionBar.style.width = `${metrics.repetition_score}%`;

    // Update sentence highlighting
    displayHighlightedText(data.sentence_scores);
}

// Display highlighted text
function displayHighlightedText(sentenceScores) {
    highlightedText.innerHTML = '';

    sentenceScores.forEach(item => {
        const span = document.createElement('span');
        span.className = 'sentence';
        span.textContent = item.text + ' ';

        // Color based on score
        const score = item.score;
        let backgroundColor;

        if (score >= 70) {
            // High AI probability - red
            const intensity = Math.min((score - 70) / 30, 1);
            backgroundColor = `rgba(239, 68, 68, ${0.2 + intensity * 0.5})`;
        } else if (score <= 30) {
            // Low AI probability - green
            const intensity = Math.min((30 - score) / 30, 1);
            backgroundColor = `rgba(16, 185, 129, ${0.1 + intensity * 0.3})`;
        } else {
            // Uncertain - blue
            backgroundColor = `rgba(102, 126, 234, ${0.1 + (score - 30) / 80})`;
        }

        span.style.backgroundColor = backgroundColor;
        span.title = `AI Score: ${score.toFixed(1)}`;

        highlightedText.appendChild(span);
    });
}

// Update status badge
function updateStatus(message, type) {
    const dot = statusBadge.querySelector('.status-dot');
    const text = statusBadge.querySelector('span:last-child');

    text.textContent = message;

    // Update colors
    if (type === 'success') {
        statusBadge.style.background = 'rgba(16, 185, 129, 0.1)';
        statusBadge.style.borderColor = 'rgba(16, 185, 129, 0.3)';
        statusBadge.style.color = '#10b981';
        dot.style.background = '#10b981';
    } else if (type === 'warning') {
        statusBadge.style.background = 'rgba(245, 158, 11, 0.1)';
        statusBadge.style.borderColor = 'rgba(245, 158, 11, 0.3)';
        statusBadge.style.color = '#f59e0b';
        dot.style.background = '#f59e0b';
    } else if (type === 'danger') {
        statusBadge.style.background = 'rgba(239, 68, 68, 0.1)';
        statusBadge.style.borderColor = 'rgba(239, 68, 68, 0.3)';
        statusBadge.style.color = '#ef4444';
        dot.style.background = '#ef4444';
    }
}

// Cookie Consent Logic
function initCookieBanner() {
    const cookieBanner = document.getElementById('cookieBanner');
    const acceptBtn = document.getElementById('acceptCookies');

    if (cookieBanner && acceptBtn) {
        const cookiesAccepted = localStorage.getItem('cookiesAccepted');

        if (!cookiesAccepted) {
            // Show banner after a short delay
            setTimeout(() => {
                cookieBanner.classList.add('show');
            }, 1000);
        }

        acceptBtn.addEventListener('click', () => {
            localStorage.setItem('cookiesAccepted', 'true');
            cookieBanner.classList.remove('show');
        });
    }
}

// Initialize
updateCharCount();
initCookieBanner();
