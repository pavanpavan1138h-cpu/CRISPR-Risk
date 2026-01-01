

// 1. Risk Distribution (Doughnut)
function updateRiskChart(results) {
    const ctx = document.getElementById('riskChart').getContext('2d');

    const riskCounts = { Low: 0, Medium: 0, High: 0 };
    results.forEach(r => {
        if (riskCounts[r.risk_class] !== undefined) {
            riskCounts[r.risk_class]++;
        }
    });

    if (window.riskChartInstance) window.riskChartInstance.destroy();

    window.riskChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Low', 'Medium', 'High'],
            datasets: [{
                data: [riskCounts.Low, riskCounts.Medium, riskCounts.High],
                backgroundColor: ['#34d399', '#fbbf24', '#f87171'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'right', labels: { color: '#cbd5e1', boxWidth: 10 } }
            },
            layout: { padding: 0 }
        }
    });
}

// 2. Mismatch Distribution (Bar)
function renderMismatchChart(results) {
    const ctx = document.getElementById('mismatchChart').getContext('2d');

    // Count mismatches (0, 1, 2, 3+)
    const counts = { 0: 0, 1: 0, 2: 0, 3: 0 };
    results.forEach(r => {
        let m = r.mismatches;
        if (m >= 3) counts[3]++;
        else counts[m]++;
    });

    if (window.mismatchChartInstance) window.mismatchChartInstance.destroy();

    window.mismatchChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['0 MM', '1 MM', '2 MM', '3+ MM'],
            datasets: [{
                label: 'Off-Targets',
                data: [counts[0], counts[1], counts[2], counts[3]],
                backgroundColor: '#6366f1',
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#cbd5e1' } },
                x: { grid: { display: false }, ticks: { color: '#cbd5e1' } }
            }
        }
    });
}

// 3. Biological Feature Profile (Radar)
function renderFeatureRadar(results) {
    const ctx = document.getElementById('radarChart').getContext('2d');

    // Calculate average metrics for top 5 riskiest off-targets
    const topRisky = results.slice(0, 5);
    let avgE = 0, avgC = 0, avgD = 0, avgH = 0;

    if (topRisky.length > 0) {
        topRisky.forEach(r => {
            avgE += r.features.gene_essentiality;
            avgC += r.features.chromatin_accessibility;
            avgD += r.features.tss_proximity;
            avgH += r.features.disease_involvement;
        });
        avgE /= topRisky.length;
        avgC /= topRisky.length;
        avgD /= topRisky.length;
        avgH /= topRisky.length;
    }

    if (window.radarChartInstance) window.radarChartInstance.destroy();

    window.radarChartInstance = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Essentiality', 'Chromatin', 'TSS Prox', 'Disease'],
            datasets: [{
                label: 'Avg Profile (Top 5)',
                data: [avgE, avgC, avgD, avgH],
                backgroundColor: 'rgba(236, 72, 153, 0.2)',
                borderColor: '#ec4899',
                pointBackgroundColor: '#ec4899',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                r: {
                    angleLines: { color: 'rgba(255,255,255,0.1)' },
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    pointLabels: { color: '#cbd5e1', font: { size: 11 } },
                    ticks: { display: false, backdropColor: 'transparent' }
                }
            }
        }
    });
}

// --- Navigation ---
function navigateTo(viewId) {
    // 1. Update Views
    const sections = {
        'home': 'home-section',
        'results': 'results-section',
        'about': 'about-section'
    };

    Object.keys(sections).forEach(key => {
        const el = document.getElementById(sections[key]);
        if (key === viewId) {
            el.classList.remove('hidden');
        } else {
            el.classList.add('hidden');
        }
    });

    // 2. Update Nav Links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
        const onclick = link.getAttribute('onclick');
        if (onclick && onclick.includes(`'${viewId}'`)) {
            link.classList.add('active');
        }
    });

    // 3. Reset states if returning home
    if (viewId === 'home') {
        const input = document.getElementById('grna-input');
        const btn = document.getElementById('predict-btn');
        if (input) {
            input.value = '';
            input.focus();
        }
        if (btn) btn.disabled = false;
        document.getElementById('error-msg').classList.add('hidden');
    }
}

// --- Event Listeners ---
document.addEventListener('DOMContentLoaded', () => {
    // Enter Key Support
    const input = document.getElementById('grna-input');
    if (input) {
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                predictRisk();
            }
        });
    }
});

async function predictRisk() {
    const input = document.getElementById('grna-input');
    const btn = document.getElementById('predict-btn');
    const grna = input.value.trim().toUpperCase();
    const errorDiv = document.getElementById('error-msg');
    const loadingDiv = document.getElementById('loading');

    errorDiv.classList.add('hidden');
    input.classList.remove('error-border');

    // Validation
    if (grna.length < 18 || grna.length > 25 || !/^[ATGC]+$/.test(grna)) {
        showError("Invalid sequence. Must be 18-25 nucleotides (A, T, G, C).");
        return;
    }

    // Disable UI
    btn.disabled = true;
    loadingDiv.classList.remove('hidden');

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ grna: grna })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Prediction failed');
        }

        displayResults(data.results);

        // Success: 
        // 1. Enable and show 'Results' in Nav
        const resultsLink = document.getElementById('nav-results');
        resultsLink.classList.remove('hidden');

        // 2. Navigate to Results
        loadingDiv.classList.add('hidden');
        navigateTo('results');

    } catch (err) {
        loadingDiv.classList.add('hidden');
        btn.disabled = false;
        showError(err.message);
    }
}

function showError(msg) {
    const errorDiv = document.getElementById('error-msg');
    errorDiv.textContent = msg;
    errorDiv.classList.remove('hidden');
}

function displayResults(results) {
    const tableBody = document.querySelector('#results-table tbody');
    const topScoreEl = document.getElementById('top-score');
    const topRiskEl = document.getElementById('top-risk-class');

    tableBody.innerHTML = '';

    // Top Score Logic
    if (results.length > 0) {
        const topResult = results[0]; // Already sorted by backend
        topScoreEl.textContent = topResult.brs_score.toFixed(3);

        topRiskEl.textContent = topResult.risk_class.toUpperCase();
        topRiskEl.className = 'risk-badge risk-' + topResult.risk_class.toLowerCase();
    }

    // Populate Table
    results.forEach(item => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td class="seq">${item.sequence}</td>
            <td>${item.mismatches}</td>
            <td>${item.off_target_prob.toFixed(4)}</td>
            <td><strong>${item.brs_score.toFixed(4)}</strong></td>
            <td>${item.features.gene_essentiality}</td>
            <td>${item.features.functional_region}</td>
            <td><span class="risk-badge risk-${item.risk_class.toLowerCase()}">${item.risk_class}</span></td>
        `;
        tableBody.appendChild(row);
    });

    updateRiskChart(results);
    renderMismatchChart(results);
    renderFeatureRadar(results);
}


