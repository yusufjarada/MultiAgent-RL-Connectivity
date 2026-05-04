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
const AGENT_RADIUS = 14;

let COMM_RANGE = 250;
let N_AGENTS = 8;
let mode = 'broadcast';
let agents = [];
let targets = [];
let fiedlerHistory = [];
let frame = 0;
let hoveredAgent = -1;
let mouseX = 0, mouseY = 0;

// --- Live inference mode ---
let liveMode = false;
let liveMethod = null;
let modelData = null;   // loaded models.json
let liveAgents = [];    // agent positions for live mode
let liveLandmarks = []; // landmark positions
let liveEdges = [];     // current edges from model
let liveCommRate = 1.0;
let liveFiedler = 3.0;

// --- Mouse tracking for hover ---

canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    mouseX = e.clientX - rect.left;
    mouseY = e.clientY - rect.top;

    hoveredAgent = -1;
    for (let i = 0; i < agents.length; i++) {
        if (distance(agents[i], { x: mouseX, y: mouseY }) < AGENT_RADIUS + 5) {
            hoveredAgent = i;
            break;
        }
    }
});

canvas.addEventListener('mouseleave', () => {
    hoveredAgent = -1;
});

// --- Slider controls ---

function updateRange(val) {
    COMM_RANGE = parseInt(val);
    document.getElementById('range-value').textContent = val + ' px';
    document.getElementById('info-range').textContent = val + ' px';
    fiedlerHistory = [];
}

function updateAgentCount(val) {
    N_AGENTS = parseInt(val);
    document.getElementById('agent-value').textContent = val;
    document.getElementById('info-agents').textContent = val;
    initAgents();
    fiedlerHistory = [];
}

// --- Agent initialization ---

function initAgents() {
    agents = [];
    for (let i = 0; i < N_AGENTS; i++) {
        agents.push({
            x: 80 + Math.random() * (W - 160),
            y: 80 + Math.random() * (H - 160),
            vx: (Math.random() - 0.5) * 1.5,
            vy: (Math.random() - 0.5) * 1.5,
            hue: (i / N_AGENTS) * 360,
        });
    }
    targets = [];
    // Keep ~4 agents per row, so 8→2x4, 12→3x4, 16→4x4, 20→4x5
    const cols = Math.min(N_AGENTS, Math.ceil(N_AGENTS / Math.ceil(N_AGENTS / 4)));
    const rows = Math.ceil(N_AGENTS / cols);
    for (let i = 0; i < N_AGENTS; i++) {
        const col = i % cols;
        const row = Math.floor(i / cols);
        targets.push({
            x: 80 + col * ((W - 160) / Math.max(cols - 1, 1)),
            y: 80 + row * ((H - 160) / Math.max(rows - 1, 1)),
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

function fiedlerValue(adj) {
    const n = adj.length;
    if (n < 2) return 0;
    const L = computeLaplacian(adj);
    const eigenvalues = jacobiEigenvalues(L);
    eigenvalues.sort((a, b) => a - b);
    return Math.max(0, eigenvalues[1]);
}

function jacobiEigenvalues(matrix) {
    const n = matrix.length;
    const A = matrix.map(row => Float64Array.from(row));
    const maxIter = 100;

    for (let iter = 0; iter < maxIter; iter++) {
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

        const theta = 0.5 * Math.atan2(2 * A[p][q], A[p][p] - A[q][q]);
        const c = Math.cos(theta), s = Math.sin(theta);

        const newA = A.map(row => Float64Array.from(row));
        for (let i = 0; i < n; i++) {
            newA[i][p] = c * A[i][p] + s * A[i][q];
            newA[i][q] = -s * A[i][p] + c * A[i][q];
        }
        for (let j = 0; j < n; j++) {
            A[p][j] = c * newA[p][j] + s * newA[q][j];
            A[q][j] = -s * newA[p][j] + c * newA[q][j];
        }
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                A[j][i] = A[i][j];
            }
        }
    }

    return Array.from({ length: matrix.length }, (_, i) => A[i][i]);
}

// --- Communication logic ---

function getActiveEdges() {
    const edges = [];
    const allPairs = [];

    for (let i = 0; i < N_AGENTS; i++) {
        for (let j = i + 1; j < N_AGENTS; j++) {
            if (distance(agents[i], agents[j]) <= COMM_RANGE) {
                allPairs.push([i, j]);
            }
        }
    }

    if (mode === 'broadcast' || mode === 'tarmac_sim') {
        return allPairs;
    }

    if (mode === 'gated') {
        for (const [i, j] of allPairs) {
            const gateI = sigmoid(Math.sin(frame * 0.015 + i * 1.7) * 2.0
                                + Math.cos(frame * 0.008 + i * 3.1) * 1.5 + 0.5);
            const gateJ = sigmoid(Math.sin(frame * 0.015 + j * 1.7) * 2.0
                                + Math.cos(frame * 0.008 + j * 3.1) * 1.5 + 0.5);
            if (gateI > 0.5 || gateJ > 0.5) {
                edges.push([i, j]);
            }
        }
        return edges;
    }

    if (mode === 'ours') {
        const candidateEdges = [];
        for (const [i, j] of allPairs) {
            const gateI = sigmoid(Math.sin(frame * 0.015 + i * 1.7) * 2.0
                                + Math.cos(frame * 0.008 + i * 3.1) * 1.5 + 0.5);
            const gateJ = sigmoid(Math.sin(frame * 0.015 + j * 1.7) * 2.0
                                + Math.cos(frame * 0.008 + j * 3.1) * 1.5 + 0.5);
            if (gateI > 0.5 && gateJ > 0.5) {
                candidateEdges.push([i, j]);
            }
        }

        let currentEdges = [...candidateEdges];
        let adj = buildAdjacency(currentEdges);
        let fv = fiedlerValue(adj);

        if (fv < 0.1) {
            const droppedEdges = allPairs.filter(
                e => !candidateEdges.some(c => c[0] === e[0] && c[1] === e[1])
            );
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

        const dx = t.x - a.x;
        const dy = t.y - a.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist > 5) {
            a.vx += (dx / dist) * 0.15 + (Math.random() - 0.5) * 0.3;
            a.vy += (dy / dist) * 0.15 + (Math.random() - 0.5) * 0.3;
        } else {
            a.vx += (Math.random() - 0.5) * 0.2;
            a.vy += (Math.random() - 0.5) * 0.2;
        }

        a.vx *= 0.95;
        a.vy *= 0.95;

        const speed = Math.sqrt(a.vx * a.vx + a.vy * a.vy);
        if (speed > 2.5) {
            a.vx = (a.vx / speed) * 2.5;
            a.vy = (a.vy / speed) * 2.5;
        }

        a.x += a.vx;
        a.y += a.vy;

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
    const maxDist = Math.sqrt(W * W + H * H) * N_AGENTS;
    return Math.max(0, 1 - totalDist / (maxDist * 0.3));
}

// --- Rendering ---

function drawGrid() {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
    ctx.lineWidth = 1;

    const gridSize = 50;
    for (let x = gridSize; x < W; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, H);
        ctx.stroke();
    }
    for (let y = gridSize; y < H; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(W, y);
        ctx.stroke();
    }

    // Scale markers along bottom
    ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.font = '9px Menlo';
    ctx.textAlign = 'center';
    for (let x = 100; x < W; x += 100) {
        ctx.fillText(x + 'px', x, H - 5);
    }
}

function drawCommRange(agent) {
    ctx.beginPath();
    ctx.arc(agent.x, agent.y, COMM_RANGE, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Label
    ctx.fillStyle = 'rgba(255, 255, 255, 0.25)';
    ctx.font = '9px Menlo';
    ctx.textAlign = 'center';
    ctx.fillText('range: ' + COMM_RANGE + 'px', agent.x, agent.y - COMM_RANGE - 6);
}

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
    ctx.strokeStyle = hoveredAgent === index
        ? '#fff'
        : `hsl(${a.hue}, 60%, 70%)`;
    ctx.lineWidth = hoveredAgent === index ? 3 : 2;
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
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
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

    fctx.fillStyle = '#888';
    fctx.font = '9px Menlo';
    fctx.fillText('Fiedler value over time', 4, 12);
}

function render() {
    ctx.clearRect(0, 0, W, H);

    // Grid
    drawGrid();

    // Targets
    for (let i = 0; i < targets.length; i++) {
        drawTarget(targets[i], i);
    }

    // Hovered agent comm range
    if (hoveredAgent >= 0 && hoveredAgent < agents.length) {
        drawCommRange(agents[hoveredAgent]);
    }

    // Active edges
    const activeEdges = getActiveEdges();
    const adj = buildAdjacency(activeEdges);
    const fv = fiedlerValue(adj);
    const totalPossible = N_AGENTS * (N_AGENTS - 1) / 2;
    const commRate = totalPossible > 0 ? activeEdges.length / totalPossible : 0;

    for (const [i, j] of activeEdges) {
        const alpha = 0.4 + 0.3 * (1 - distance(agents[i], agents[j]) / COMM_RANGE);
        const color = fv < 0.1
            ? `rgba(255, 80, 80, ${alpha})`
            : `rgba(80, 200, 100, ${alpha})`;
        drawEdge(agents[i], agents[j], alpha, color);
    }

    // Agents
    for (let i = 0; i < N_AGENTS; i++) {
        drawAgent(agents[i], i);
    }

    // Stats
    const modeNames = { broadcast: 'Broadcast (CommNet)', gated: 'Gated (IC3Net)', tarmac_sim: 'TarMAC', ours: 'Gated + Conn. (Ours)' };
    document.getElementById('stat-mode').textContent = modeNames[mode];
    document.getElementById('stat-fiedler').textContent = fv.toFixed(3);
    document.getElementById('stat-fiedler').style.color = fv < 0.1 ? '#e55' : '#4CAF50';
    document.getElementById('stat-comm').textContent = (commRate * 100).toFixed(0) + '%';
    document.getElementById('stat-edges').textContent = activeEdges.length + ' / ' + totalPossible;
    document.getElementById('stat-coverage').textContent = computeCoverage().toFixed(2);

    // Fiedler history
    if (frame % 3 === 0) {
        fiedlerHistory.push(fv);
        if (fiedlerHistory.length > 200) fiedlerHistory.shift();
    }
    drawFiedlerChart();
}

// --- Mode switching ---

function clearAllButtons() {
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.live-btn').forEach(b => b.classList.remove('active'));
}

function setMode(m) {
    liveMode = false;
    mode = m;
    fiedlerHistory = [];
    clearAllButtons();
    const btn = document.getElementById('btn-' + m) || document.getElementById('btn-' + m.replace('_', '-'));
    if (btn) btn.classList.add('active');
    document.getElementById('range-slider').disabled = false;
    document.getElementById('agent-slider').disabled = false;
}

// --- Live inference mode ---

// Load model weights on startup
fetch('models.json')
    .then(r => r.json())
    .then(data => { modelData = data; console.log('Models loaded:', Object.keys(data)); })
    .catch(() => { console.log('No models.json found — live inference disabled'); });

function initLiveEnv(nAgents) {
    // Random positions for agents and landmarks in [-1, 1]
    liveAgents = [];
    liveLandmarks = [];
    for (let i = 0; i < nAgents; i++) {
        liveAgents.push({
            px: (Math.random() - 0.5) * 2,
            py: (Math.random() - 0.5) * 2,
            vx: 0, vy: 0,
        });
        liveLandmarks.push({
            px: (Math.random() - 0.5) * 1.6,
            py: (Math.random() - 0.5) * 1.6,
        });
    }
}

function buildObservation(agentIdx) {
    // Mimic MPE simple_spread observation:
    // [self_vel(2), self_pos(2), landmark_rel_pos(N*2), other_agents_rel_pos((N-1)*2)]
    const a = liveAgents[agentIdx];
    const obs = [];
    // Self velocity
    obs.push(a.vx, a.vy);
    // Self position
    obs.push(a.px, a.py);
    // Relative landmark positions
    for (const lm of liveLandmarks) {
        obs.push(lm.px - a.px, lm.py - a.py);
    }
    // Relative other agent positions
    for (let j = 0; j < liveAgents.length; j++) {
        if (j === agentIdx) continue;
        obs.push(liveAgents[j].px - a.px, liveAgents[j].py - a.py);
    }
    return obs;
}

function applyAction(agentIdx, action) {
    // Actions: 0=noop, 1=left, 2=right, 3=down, 4=up
    const force = 0.1;
    const a = liveAgents[agentIdx];
    if (action === 1) a.vx -= force;
    if (action === 2) a.vx += force;
    if (action === 3) a.vy -= force;
    if (action === 4) a.vy += force;

    // Damping
    a.vx *= 0.9;
    a.vy *= 0.9;

    // Update position
    a.px += a.vx * 0.1;
    a.py += a.vy * 0.1;

    // Clamp to bounds
    a.px = Math.max(-1.5, Math.min(1.5, a.px));
    a.py = Math.max(-1.5, Math.min(1.5, a.py));
}

function startLive(method) {
    if (!modelData || !modelData[method]) {
        console.log('No model data for ' + method);
        return;
    }

    liveMode = true;
    liveMethod = method;
    fiedlerHistory = [];

    const nAgents = modelData[method].n_agents;
    N_AGENTS = nAgents;
    initLiveEnv(nAgents);

    clearAllButtons();
    const btn = document.getElementById('btn-live-' + method);
    if (btn) btn.classList.add('active');

    document.getElementById('range-slider').disabled = true;
    document.getElementById('agent-slider').disabled = true;
    document.getElementById('agent-value').textContent = nAgents;
    document.getElementById('info-agents').textContent = nAgents;
}

function mapLiveToCanvas(px, py) {
    const cx = W / 2;
    const cy = H / 2;
    const scale = Math.min(W, H) * 0.3;
    return { x: cx + px * scale, y: cy + py * scale };
}

function updateAndRenderLive() {
    if (!modelData || !modelData[liveMethod]) return;

    const nAgents = liveAgents.length;
    const params = modelData[liveMethod].params;

    // Build observation matrix (N, obs_dim)
    const obsRows = [];
    for (let i = 0; i < nAgents; i++) obsRows.push(buildObservation(i));
    const obsDim = obsRows[0].length;
    const obsFlat = new Float32Array(nAgents * obsDim);
    for (let i = 0; i < nAgents; i++)
        for (let d = 0; d < obsDim; d++)
            obsFlat[i * obsDim + d] = obsRows[i][d];
    const obsTensor = { data: obsFlat, shape: [nAgents, obsDim] };

    // Run model (every 4 frames for performance)
    if (frame % 4 === 0) {
        const result = runModel(liveMethod, obsTensor, params, nAgents);
        if (result) {
            liveEdges = result.edges;
            liveCommRate = result.commRate;

            // Sample actions and apply
            for (let i = 0; i < nAgents; i++) {
                const action = sampleAction(result.logits, i);
                applyAction(i, action);
            }

            // Compute Fiedler
            const adj = buildAdjacency(liveEdges);
            liveFiedler = fiedlerValue(adj);
        }
    }

    // --- Render ---
    ctx.clearRect(0, 0, W, H);
    drawGrid();

    // Landmarks
    for (let i = 0; i < liveLandmarks.length; i++) {
        const p = mapLiveToCanvas(liveLandmarks[i].px, liveLandmarks[i].py);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 10, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.font = '10px Menlo';
        ctx.textAlign = 'center';
        ctx.fillText('L' + i, p.x, p.y + 20);
    }

    // Edges
    const agentCanvasPos = liveAgents.map(a => mapLiveToCanvas(a.px, a.py));
    for (const [i, j] of liveEdges) {
        const color = liveFiedler < 0.1
            ? 'rgba(255, 80, 80, 0.5)'
            : 'rgba(80, 200, 100, 0.5)';
        drawEdge(agentCanvasPos[i], agentCanvasPos[j], 0.5, color);
    }

    // Agents
    for (let i = 0; i < nAgents; i++) {
        const p = agentCanvasPos[i];
        const hue = (i / nAgents) * 360;

        const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, AGENT_RADIUS * 2);
        gradient.addColorStop(0, `hsla(${hue}, 70%, 60%, 0.3)`);
        gradient.addColorStop(1, `hsla(${hue}, 70%, 60%, 0)`);
        ctx.beginPath();
        ctx.arc(p.x, p.y, AGENT_RADIUS * 2, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();

        ctx.beginPath();
        ctx.arc(p.x, p.y, AGENT_RADIUS, 0, Math.PI * 2);
        ctx.fillStyle = `hsl(${hue}, 60%, 50%)`;
        ctx.fill();
        ctx.strokeStyle = `hsl(${hue}, 60%, 70%)`;
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = '#fff';
        ctx.font = '11px Menlo';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(i, p.x, p.y);
    }

    // Stats
    const methodNames = {
        commnet: 'CommNet (live)',
        ic3net: 'IC3Net (live)',
        tarmac: 'TarMAC (live)',
        gated_attn: 'Ours (live)',
    };
    const totalPossible = nAgents * (nAgents - 1) / 2;
    document.getElementById('stat-mode').textContent = methodNames[liveMethod] || liveMethod;
    document.getElementById('stat-fiedler').textContent = liveFiedler.toFixed(3);
    document.getElementById('stat-fiedler').style.color = liveFiedler < 0.1 ? '#e55' : '#4CAF50';
    document.getElementById('stat-comm').textContent = (liveCommRate * 100).toFixed(0) + '%';
    document.getElementById('stat-edges').textContent = liveEdges.length + ' / ' + totalPossible;
    document.getElementById('stat-coverage').textContent = '—';

    if (frame % 3 === 0) {
        fiedlerHistory.push(liveFiedler);
        if (fiedlerHistory.length > 200) fiedlerHistory.shift();
    }
    drawFiedlerChart();
}

// --- Main loop ---

function loop() {
    if (liveMode) {
        updateAndRenderLive();
    } else {
        updateAgents();
        render();
    }
    frame++;
    requestAnimationFrame(loop);
}

initAgents();
loop();
