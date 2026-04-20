// ============================================================
// MARL Communication Demo
// Visualizes broadcast vs gated vs gated+connectivity agents
// ============================================================

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const fiedlerCanvas = document.getElementById('fiedler-chart');
const fctx = fiedlerCanvas.getContext('2d');

const W = canvas.width;
const H = canvas.height;
const N_AGENTS = 8;
const AGENT_RADIUS = 14;
const COMM_RANGE = 250;

let mode = 'broadcast';
let agents = [];
let targets = [];
let fiedlerHistory = [];
let frame = 0;

// --- Agent initialization ---

function initAgents() {
    agents = [];
    for (let i = 0; i < N_AGENTS; i++) {
        agents.push({
            x: 100 + Math.random() * (W - 200),
            y: 100 + Math.random() * (H - 200),
            vx: (Math.random() - 0.5) * 1.5,
            vy: (Math.random() - 0.5) * 1.5,
            hue: (i / N_AGENTS) * 360,
            gateOpen: true,
        });
    }
    // Target positions agents should spread to cover
    targets = [];
    for (let i = 0; i < N_AGENTS; i++) {
        targets.push({
            x: 80 + (i % 4) * ((W - 160) / 3),
            y: 80 + Math.floor(i / 4) * ((H - 160)),
        });
    }
}

function resetAgents() {
    initAgents();
    fiedlerHistory = [];
}

// --- Graph theory ---

function distance(a, b) {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

function buildAdjacency(activeEdges) {
    const adj = Array.from({ length: N_AGENTS }, () => new Float64Array(N_AGENTS));
    for (const [i, j] of activeEdges) {
        adj[i][j] = 1;
        adj[j][i] = 1;
    }
    return adj;
}

function computeLaplacian(adj) {
    const n = adj.length;
    const L = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
        let deg = 0;
        for (let j = 0; j < n; j++) {
            deg += adj[i][j];
            L[i][j] = -adj[i][j];
        }
        L[i][i] = deg;
    }
    return L;
}

// Power iteration to find smallest non-trivial eigenvalue (Fiedler value)
// Uses inverse iteration with shift to find second-smallest eigenvalue
function fiedlerValue(adj) {
    const n = adj.length;
    const L = computeLaplacian(adj);

    // Simple approach: compute all eigenvalues of symmetric matrix
    // using QR-like iteration (simplified for small n)
    // For n=8 we can afford to just use the characteristic approach

    // Actually, for a small matrix, let's use the power method on (L + shift*I)^-1
    // to find the smallest eigenvalue of L that isn't 0.

    // Even simpler: since n is small (8), use the Jacobi eigenvalue algorithm
    const eigenvalues = jacobiEigenvalues(L);
    eigenvalues.sort((a, b) => a - b);

    // Second smallest (first is ~0)
    return Math.max(0, eigenvalues[1]);
}

function jacobiEigenvalues(matrix) {
    const n = matrix.length;
    // Copy matrix
    const A = matrix.map(row => Float64Array.from(row));
    const maxIter = 100;

    for (let iter = 0; iter < maxIter; iter++) {
        // Find largest off-diagonal element
        let maxVal = 0, p = 0, q = 1;
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                if (Math.abs(A[i][j]) > maxVal) {
                    maxVal = Math.abs(A[i][j]);
                    p = i; q = j;
                }
            }
        }
        if (maxVal < 1e-10) break;

        // Compute rotation
        const theta = 0.5 * Math.atan2(2 * A[p][q], A[p][p] - A[q][q]);
        const c = Math.cos(theta), s = Math.sin(theta);

        // Apply rotation
        const newA = A.map(row => Float64Array.from(row));
        for (let i = 0; i < n; i++) {
            newA[i][p] = c * A[i][p] + s * A[i][q];
            newA[i][q] = -s * A[i][p] + c * A[i][q];
        }
        for (let j = 0; j < n; j++) {
            A[p][j] = c * newA[p][j] + s * newA[q][j];
            A[q][j] = -s * newA[p][j] + c * newA[q][j];
        }
        // Copy back symmetric parts
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                A[j][i] = A[i][j];
            }
        }
    }

    return Array.from({ length: n }, (_, i) => A[i][i]);
}

// --- Communication logic ---

function getActiveEdges() {
    const edges = [];
    const allPairs = [];

    // Build all in-range pairs
    for (let i = 0; i < N_AGENTS; i++) {
        for (let j = i + 1; j < N_AGENTS; j++) {
            if (distance(agents[i], agents[j]) <= COMM_RANGE) {
                allPairs.push([i, j]);
            }
        }
    }

    if (mode === 'broadcast') {
        // All in-range edges active
        return allPairs;
    }

    if (mode === 'gated') {
        // Each agent independently decides to transmit based on simulated "need"
        // Agents far from their targets are more likely to gate off (they're busy)
        // This is a toy heuristic — real IC3Net learns this
        for (const [i, j] of allPairs) {
            const di = distance(agents[i], targets[i % targets.length]);
            const dj = distance(agents[j], targets[j % targets.length]);
            // Agents that are close to target shut up (less need to coordinate)
            // + some randomness
            const probI = sigmoid((di - 100) / 50 + Math.sin(frame * 0.02 + i) * 0.5);
            const probJ = sigmoid((dj - 100) / 50 + Math.sin(frame * 0.02 + j) * 0.5);
            if (probI > 0.5 && probJ > 0.5) {
                edges.push([i, j]);
            }
        }
        return edges;
    }

    if (mode === 'ours') {
        // Same gating logic, but with connectivity repair
        const candidateEdges = [];
        for (const [i, j] of allPairs) {
            const di = distance(agents[i], targets[i % targets.length]);
            const dj = distance(agents[j], targets[j % targets.length]);
            const probI = sigmoid((di - 100) / 50 + Math.sin(frame * 0.02 + i) * 0.5);
            const probJ = sigmoid((dj - 100) / 50 + Math.sin(frame * 0.02 + j) * 0.5);
            if (probI > 0.5 && probJ > 0.5) {
                candidateEdges.push([i, j]);
            }
        }

        // Check connectivity — if Fiedler value is too low, add back edges
        let currentEdges = [...candidateEdges];
        let adj = buildAdjacency(currentEdges);
        let fv = fiedlerValue(adj);

        // Connectivity repair: add back dropped edges until Fiedler > threshold
        if (fv < 0.1) {
            const droppedEdges = allPairs.filter(
                e => !candidateEdges.some(c => c[0] === e[0] && c[1] === e[1])
            );
            // Sort dropped edges by how much they'd help connectivity (shortest first)
            droppedEdges.sort((a, b) =>
                distance(agents[a[0]], agents[a[1]]) - distance(agents[b[0]], agents[b[1]])
            );

            for (const edge of droppedEdges) {
                if (fv >= 0.1) break;
                currentEdges.push(edge);
                adj = buildAdjacency(currentEdges);
                fv = fiedlerValue(adj);
            }
        }

        return currentEdges;
    }

    return allPairs;
}

function sigmoid(x) {
    return 1.0 / (1.0 + Math.exp(-x));
}

// --- Agent movement ---

function updateAgents() {
    for (let i = 0; i < N_AGENTS; i++) {
        const a = agents[i];
        const t = targets[i % targets.length];

        // Move toward target with some noise
        const dx = t.x - a.x;
        const dy = t.y - a.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist > 5) {
            a.vx += (dx / dist) * 0.15 + (Math.random() - 0.5) * 0.3;
            a.vy += (dy / dist) * 0.15 + (Math.random() - 0.5) * 0.3;
        } else {
            // Wander near target
            a.vx += (Math.random() - 0.5) * 0.2;
            a.vy += (Math.random() - 0.5) * 0.2;
        }

        // Damping
        a.vx *= 0.95;
        a.vy *= 0.95;

        // Clamp speed
        const speed = Math.sqrt(a.vx * a.vx + a.vy * a.vy);
        if (speed > 2.5) {
            a.vx = (a.vx / speed) * 2.5;
            a.vy = (a.vy / speed) * 2.5;
        }

        a.x += a.vx;
        a.y += a.vy;

        // Bounce off walls
        if (a.x < AGENT_RADIUS) { a.x = AGENT_RADIUS; a.vx *= -0.5; }
        if (a.x > W - AGENT_RADIUS) { a.x = W - AGENT_RADIUS; a.vx *= -0.5; }
        if (a.y < AGENT_RADIUS) { a.y = AGENT_RADIUS; a.vy *= -0.5; }
        if (a.y > H - AGENT_RADIUS) { a.y = H - AGENT_RADIUS; a.vy *= -0.5; }
    }
}

// --- Coverage score ---

function computeCoverage() {
    let totalDist = 0;
    for (let i = 0; i < N_AGENTS; i++) {
        totalDist += distance(agents[i], targets[i % targets.length]);
    }
    // Normalize: 0 = far, 1 = perfect coverage
    const maxDist = Math.sqrt(W * W + H * H) * N_AGENTS;
    return Math.max(0, 1 - totalDist / (maxDist * 0.3));
}

// --- Rendering ---

function drawEdge(a, b, alpha, color) {
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.strokeStyle = color || `rgba(80, 200, 100, ${alpha})`;
    ctx.lineWidth = 1.5;
    ctx.stroke();
}

function drawAgent(a, index) {
    // Glow
    const gradient = ctx.createRadialGradient(a.x, a.y, 0, a.x, a.y, AGENT_RADIUS * 2);
    gradient.addColorStop(0, `hsla(${a.hue}, 70%, 60%, 0.3)`);
    gradient.addColorStop(1, `hsla(${a.hue}, 70%, 60%, 0)`);
    ctx.beginPath();
    ctx.arc(a.x, a.y, AGENT_RADIUS * 2, 0, Math.PI * 2);
    ctx.fillStyle = gradient;
    ctx.fill();

    // Body
    ctx.beginPath();
    ctx.arc(a.x, a.y, AGENT_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = `hsl(${a.hue}, 60%, 50%)`;
    ctx.fill();
    ctx.strokeStyle = `hsl(${a.hue}, 60%, 70%)`;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Label
    ctx.fillStyle = '#fff';
    ctx.font = '11px Menlo';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(index, a.x, a.y);
}

function drawTarget(t, index) {
    ctx.beginPath();
    ctx.arc(t.x, t.y, 6, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
}

function drawFiedlerChart() {
    const cw = fiedlerCanvas.width;
    const ch = fiedlerCanvas.height;
    fctx.clearRect(0, 0, cw, ch);

    if (fiedlerHistory.length < 2) return;

    const maxVal = Math.max(2, ...fiedlerHistory) * 1.1;
    const step = cw / Math.max(fiedlerHistory.length - 1, 1);

    // Threshold line
    const threshY = ch - (0.1 / maxVal) * ch;
    fctx.beginPath();
    fctx.moveTo(0, threshY);
    fctx.lineTo(cw, threshY);
    fctx.strokeStyle = 'rgba(255, 80, 80, 0.3)';
    fctx.lineWidth = 1;
    fctx.setLineDash([4, 4]);
    fctx.stroke();
    fctx.setLineDash([]);

    // Label
    fctx.fillStyle = 'rgba(255, 80, 80, 0.5)';
    fctx.font = '9px Menlo';
    fctx.fillText('disconnect', cw - 58, threshY - 4);

    // Fiedler line
    fctx.beginPath();
    for (let i = 0; i < fiedlerHistory.length; i++) {
        const x = i * step;
        const y = ch - (fiedlerHistory[i] / maxVal) * ch;
        if (i === 0) fctx.moveTo(x, y);
        else fctx.lineTo(x, y);
    }
    fctx.strokeStyle = '#4CAF50';
    fctx.lineWidth = 1.5;
    fctx.stroke();

    // Label
    fctx.fillStyle = '#888';
    fctx.font = '9px Menlo';
    fctx.fillText('Fiedler value over time', 4, 12);
}

function render() {
    ctx.clearRect(0, 0, W, H);

    // Draw targets
    for (let i = 0; i < targets.length; i++) {
        drawTarget(targets[i], i);
    }

    // Get active edges and draw them
    const activeEdges = getActiveEdges();
    const adj = buildAdjacency(activeEdges);
    const fv = fiedlerValue(adj);
    const totalPossible = N_AGENTS * (N_AGENTS - 1) / 2;
    const commRate = activeEdges.length / totalPossible;

    for (const [i, j] of activeEdges) {
        const alpha = 0.4 + 0.3 * (1 - distance(agents[i], agents[j]) / COMM_RANGE);
        const color = fv < 0.1
            ? `rgba(255, 80, 80, ${alpha})`
            : `rgba(80, 200, 100, ${alpha})`;
        drawEdge(agents[i], agents[j], alpha, color);
    }

    // Draw agents
    for (let i = 0; i < N_AGENTS; i++) {
        drawAgent(agents[i], i);
    }

    // Update stats
    const modeNames = { broadcast: 'Broadcast', gated: 'Gated', ours: 'Gated + Conn.' };
    document.getElementById('stat-mode').textContent = modeNames[mode];
    document.getElementById('stat-fiedler').textContent = fv.toFixed(3);
    document.getElementById('stat-fiedler').style.color = fv < 0.1 ? '#e55' : '#4CAF50';
    document.getElementById('stat-comm').textContent = (commRate * 100).toFixed(0) + '%';
    document.getElementById('stat-edges').textContent = activeEdges.length + ' / ' + totalPossible;
    document.getElementById('stat-coverage').textContent = computeCoverage().toFixed(2);

    // Track Fiedler history (keep last 200)
    if (frame % 3 === 0) {
        fiedlerHistory.push(fv);
        if (fiedlerHistory.length > 200) fiedlerHistory.shift();
    }
    drawFiedlerChart();
}

// --- Mode switching ---

function setMode(m) {
    mode = m;
    fiedlerHistory = [];
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('btn-' + m).classList.add('active');
}

// --- Main loop ---

function loop() {
    updateAgents();
    render();
    frame++;
    requestAnimationFrame(loop);
}

initAgents();
loop();
