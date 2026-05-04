// ============================================================
// Cooperative Multi-Target Pursuit Demo
// Variable agents chase variable targets
// Communication mode determines coordination quality
// ============================================================

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const fiedlerCanvas = document.getElementById('fiedler-chart');
const fctx = fiedlerCanvas.getContext('2d');

const W = canvas.width;
const H = canvas.height;
const CX = W / 2, CY = H / 2;

const AGENT_RADIUS = 14;
const TARGET_RADIUS = 11;
const SIGHT_RANGE = 230;
const AGENT_SPEED = 2.4;
const TARGET_SPEED = 1.5;
const SURROUND_RADIUS = 75;
const CAPTURE_THRESHOLD = 0.38;
const COMM_RANGE = 320;

let nAgents = 8;
let nTargets = 3;
let mode = 'broadcast';
let frame = 0;
let agents = [];
let targets = [];
let fiedlerHistory = [];
let captures = { none: 0, broadcast: 0, tarmac: 0, gated: 0, ours: 0 };
let captureFlash = 0;
let startTime = Date.now();
let totalMessages = 0;
let totalMaxMessages = 0;

// Distinct target colors (up to 5)
const T_FILL =   ['#ff4455', '#ff9922', '#bb55ff', '#22ccbb', '#ffcc00'];
const T_STROKE =  ['#ff8899', '#ffbb66', '#dd99ff', '#66eedd', '#ffdd55'];
const T_GLOW =   ['255,68,85', '255,153,34', '187,85,255', '34,204,187', '255,204,0'];

// --- Sliders ---

function updateAgentCount(val) {
    nAgents = parseInt(val);
    document.getElementById('val-agents').textContent = val;
    initSim();
}

function updateTargetCount(val) {
    nTargets = parseInt(val);
    document.getElementById('val-targets').textContent = val;
    initSim();
}

// --- Initialization ---

function initSim() {
    agents = [];
    for (let i = 0; i < nAgents; i++) {
        const row = i % Math.ceil(nAgents / 2);
        const col = Math.floor(i / Math.ceil(nAgents / 2));
        agents.push({
            x: 70 + col * 35 + (Math.random() - 0.5) * 20,
            y: CY - (nAgents * 12) + row * 45 + (Math.random() - 0.5) * 20,
            vx: 0, vy: 0,
            hue: (i / nAgents) * 300 + 180,
            assignedTarget: -1,
            assignedAngle: null,
            knownTargets: [],
        });
    }

    targets = [];
    for (let i = 0; i < nTargets; i++) {
        const angle = (i / nTargets) * Math.PI * 2 - Math.PI / 2;
        targets.push({
            x: CX + Math.cos(angle) * (W * 0.25) + (Math.random() - 0.5) * 80,
            y: CY + Math.sin(angle) * (H * 0.25) + (Math.random() - 0.5) * 80,
            vx: (Math.random() - 0.5) * 1.0,
            vy: (Math.random() - 0.5) * 1.0,
            captured: false,
        });
    }

    fiedlerHistory = [];
    captureFlash = 0;
    startTime = Date.now();
    totalMessages = 0;
    totalMaxMessages = 0;
}

function resetSim() { initSim(); }

// --- Math ---

function dist(a, b) { return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2); }

function buildAdjacency(edges) {
    const adj = Array.from({ length: nAgents }, () => new Float64Array(nAgents));
    for (const [i, j] of edges) { adj[i][j] = 1; adj[j][i] = 1; }
    return adj;
}

function computeLaplacian(adj) {
    const n = adj.length;
    const L = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
        let deg = 0;
        for (let j = 0; j < n; j++) { deg += adj[i][j]; L[i][j] = -adj[i][j]; }
        L[i][i] = deg;
    }
    return L;
}

function jacobiEigenvalues(matrix) {
    const n = matrix.length;
    const A = matrix.map(r => Float64Array.from(r));
    for (let iter = 0; iter < 120; iter++) {
        let maxVal = 0, p = 0, q = 1;
        for (let i = 0; i < n; i++)
            for (let j = i + 1; j < n; j++)
                if (Math.abs(A[i][j]) > maxVal) { maxVal = Math.abs(A[i][j]); p = i; q = j; }
        if (maxVal < 1e-10) break;
        const theta = 0.5 * Math.atan2(2 * A[p][q], A[p][p] - A[q][q]);
        const c = Math.cos(theta), s = Math.sin(theta);
        const nA = A.map(r => Float64Array.from(r));
        for (let i = 0; i < n; i++) {
            nA[i][p] = c * A[i][p] + s * A[i][q];
            nA[i][q] = -s * A[i][p] + c * A[i][q];
        }
        for (let j = 0; j < n; j++) {
            A[p][j] = c * nA[p][j] + s * nA[q][j];
            A[q][j] = -s * nA[p][j] + c * nA[q][j];
        }
        for (let i = 0; i < n; i++)
            for (let j = i + 1; j < n; j++) A[j][i] = A[i][j];
    }
    return Array.from({ length: matrix.length }, (_, i) => A[i][i]);
}

function fiedlerValue(adj) {
    if (adj.length < 2) return 0;
    const L = computeLaplacian(adj);
    const ev = jacobiEigenvalues(L);
    ev.sort((a, b) => a - b);
    return Math.max(0, ev[1]);
}

// --- Communication ---

function getCommEdges() {
    const allPairs = [];
    for (let i = 0; i < nAgents; i++)
        for (let j = i + 1; j < nAgents; j++)
            if (dist(agents[i], agents[j]) < COMM_RANGE)
                allPairs.push([i, j]);

    if (mode === 'none') return [];
    if (mode === 'broadcast' || mode === 'tarmac') return allPairs;

    const groupCX = agents.reduce((s, a) => s + a.x, 0) / nAgents;
    const groupCY = agents.reduce((s, a) => s + a.y, 0) / nAgents;

    function gateProb(i) {
        const flank = dist(agents[i], { x: groupCX, y: groupCY });
        return 1.0 / (1.0 + Math.exp((flank - 130) / 35 + Math.sin(frame * 0.012 + i * 1.7) * 1.0));
    }

    if (mode === 'gated') {
        const edges = [];
        for (const [i, j] of allPairs) {
            if (gateProb(i) > 0.35 && gateProb(j) > 0.35) edges.push([i, j]);
        }
        return edges;
    }

    if (mode === 'ours') {
        const candidateEdges = [];
        for (const [i, j] of allPairs) {
            if (gateProb(i) > 0.35 && gateProb(j) > 0.35) candidateEdges.push([i, j]);
        }

        let currentEdges = [...candidateEdges];
        let adj = buildAdjacency(currentEdges);
        let fv = fiedlerValue(adj);

        if (fv < 0.12) {
            const dropped = allPairs
                .filter(e => !candidateEdges.some(c => c[0] === e[0] && c[1] === e[1]))
                .sort((a, b) => dist(agents[a[0]], agents[a[1]]) - dist(agents[b[0]], agents[b[1]]));

            for (const edge of dropped) {
                if (fv >= 0.12) break;
                currentEdges.push(edge);
                adj = buildAdjacency(currentEdges);
                fv = fiedlerValue(adj);
            }
        }
        return currentEdges;
    }

    return allPairs;
}

// --- Agent behavior ---

function updateAgents() {
    const edges = getCommEdges();

    const neighbors = Array.from({ length: nAgents }, () => new Set());
    for (const [i, j] of edges) { neighbors[i].add(j); neighbors[j].add(i); }

    // Reset knowledge
    for (const a of agents) {
        a.knownTargets = [];
        a.assignedTarget = -1;
        a.assignedAngle = null;
    }

    // Direct sight
    const allKnown = Array.from({ length: nAgents }, () => new Set());
    for (let i = 0; i < nAgents; i++) {
        for (let t = 0; t < nTargets; t++) {
            if (!targets[t].captured && dist(agents[i], targets[t]) < SIGHT_RANGE) {
                agents[i].knownTargets.push({ idx: t, x: targets[t].x, y: targets[t].y });
                allKnown[i].add(t);
            }
        }
    }

    // Propagate through comm graph
    for (let pass = 0; pass < nAgents; pass++) {
        let changed = false;
        for (const [i, j] of edges) {
            for (const kt of agents[i].knownTargets) {
                if (!allKnown[j].has(kt.idx)) {
                    allKnown[j].add(kt.idx);
                    agents[j].knownTargets.push({ ...kt });
                    changed = true;
                }
            }
            for (const kt of agents[j].knownTargets) {
                if (!allKnown[i].has(kt.idx)) {
                    allKnown[i].add(kt.idx);
                    agents[i].knownTargets.push({ ...kt });
                    changed = true;
                }
            }
        }
        if (!changed) break;
    }

    // Connected components
    const compId = new Int32Array(nAgents).fill(-1);
    let nComps = 0;
    for (let i = 0; i < nAgents; i++) {
        if (compId[i] >= 0) continue;
        const q = [i]; compId[i] = nComps;
        while (q.length > 0) {
            const cur = q.shift();
            for (const nb of neighbors[cur]) {
                if (compId[nb] < 0) { compId[nb] = nComps; q.push(nb); }
            }
        }
        nComps++;
    }

    // Target assignment per component — FOCUS FIRE strategy
    // Agents in a connected component agree on which target to prioritize.
    // Priority: target closest to being captured (highest encirclement), then nearest.
    // Send enough agents to capture one target, then redistribute.
    for (let c = 0; c < nComps; c++) {
        const compAgents = [];
        const compTargets = new Set();
        for (let i = 0; i < nAgents; i++) {
            if (compId[i] !== c) continue;
            compAgents.push(i);
            for (const kt of agents[i].knownTargets) compTargets.add(kt.idx);
        }
        const knownList = [...compTargets].filter(t => !targets[t].captured);
        if (knownList.length === 0) continue;

        // Score each target: how close to capture? (encirclement + proximity of agents)
        const targetScores = knownList.map(tIdx => {
            const tgt = targets[tIdx];
            const enc = computeEncirclement(tIdx);
            // How many component agents are already near this target?
            const nearbyCount = compAgents.filter(i => dist(agents[i], tgt) < SURROUND_RADIUS * 2.5).length;
            // Avg distance of component agents to this target
            const avgDist = compAgents.reduce((s, i) => s + dist(agents[i], tgt), 0) / compAgents.length;
            // Score: prioritize targets that already have agents nearby and high encirclement
            return { tIdx, score: enc.score * 3 + nearbyCount * 0.5 + (1 - avgDist / 800) };
        });
        targetScores.sort((a, b) => b.score - a.score);

        // How many agents needed to capture a target? At least 3 for good encirclement.
        const agentsPerTarget = Math.max(3, Math.ceil(compAgents.length / Math.min(knownList.length, Math.floor(compAgents.length / 3))));

        // Assign: fill highest priority target first, then overflow to next
        let assigned = 0;
        let targetIdx = 0;
        const agentsByDist = [...compAgents]; // will sort per target

        for (let tRank = 0; tRank < targetScores.length && assigned < compAgents.length; tRank++) {
            const tIdx = targetScores[tRank].tIdx;
            const tgt = targets[tIdx];

            // Sort remaining unassigned agents by distance to this target
            const unassigned = agentsByDist.filter(i => agents[i].assignedTarget < 0);
            unassigned.sort((a, b) => dist(agents[a], tgt) - dist(agents[b], tgt));

            // How many to send? Fill up to agentsPerTarget, or all remaining if last target
            const remaining = compAgents.length - assigned;
            const toSend = (tRank === targetScores.length - 1) ? remaining : Math.min(agentsPerTarget, remaining);

            for (let k = 0; k < toSend && k < unassigned.length; k++) {
                agents[unassigned[k]].assignedTarget = tIdx;
                assigned++;
            }
        }

        // Assign surround angles within each target group
        for (const tIdx of knownList) {
            const group = compAgents.filter(i => agents[i].assignedTarget === tIdx);
            for (let k = 0; k < group.length; k++) {
                agents[group[k]].assignedAngle = (k / group.length) * Math.PI * 2;
            }
        }
    }

    // Movement
    for (let i = 0; i < nAgents; i++) {
        const a = agents[i];
        let goalX, goalY;

        if (a.assignedTarget >= 0 && a.assignedAngle !== null) {
            const t = targets[a.assignedTarget];
            goalX = t.x + Math.cos(a.assignedAngle) * SURROUND_RADIUS;
            goalY = t.y + Math.sin(a.assignedAngle) * SURROUND_RADIUS;
        } else if (a.knownTargets.length > 0) {
            let nearest = a.knownTargets[0];
            for (const kt of a.knownTargets) {
                if (dist(a, kt) < dist(a, nearest)) nearest = kt;
            }
            goalX = nearest.x;
            goalY = nearest.y;
        } else {
            const angle = (i / nAgents) * Math.PI * 2 + frame * 0.003;
            const r = Math.min(W, H) * 0.3;
            goalX = CX + Math.cos(angle) * r;
            goalY = CY + Math.sin(angle) * r;
        }

        const dx = goalX - a.x, dy = goalY - a.y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d > 2) { a.vx += (dx / d) * 0.35; a.vy += (dy / d) * 0.35; }

        a.vx *= 0.91; a.vy *= 0.91;
        const speed = Math.sqrt(a.vx * a.vx + a.vy * a.vy);
        if (speed > AGENT_SPEED) { a.vx = (a.vx / speed) * AGENT_SPEED; a.vy = (a.vy / speed) * AGENT_SPEED; }

        a.x += a.vx; a.y += a.vy;
        if (a.x < 18) { a.x = 18; a.vx *= -0.5; }
        if (a.x > W - 18) { a.x = W - 18; a.vx *= -0.5; }
        if (a.y < 18) { a.y = 18; a.vy *= -0.5; }
        if (a.y > H - 18) { a.y = H - 18; a.vy *= -0.5; }

        for (let j = 0; j < nAgents; j++) {
            if (j === i) continue;
            const dd = dist(a, agents[j]);
            if (dd < 26 && dd > 0) {
                a.vx += ((a.x - agents[j].x) / dd) * 0.4;
                a.vy += ((a.y - agents[j].y) / dd) * 0.4;
            }
        }
    }

    // Target movement
    for (let t = 0; t < nTargets; t++) {
        if (targets[t].captured) continue;
        const tgt = targets[t];

        let fleeX = 0, fleeY = 0;
        for (const a of agents) {
            const d = dist(a, tgt);
            if (d < 150 && d > 0) {
                const w = (150 - d) / 150;
                fleeX += ((tgt.x - a.x) / d) * w;
                fleeY += ((tgt.y - a.y) / d) * w;
            }
        }
        const fm = Math.sqrt(fleeX * fleeX + fleeY * fleeY);
        if (fm > 0) { tgt.vx += (fleeX / fm) * 0.25; tgt.vy += (fleeY / fm) * 0.25; }

        if (Math.random() < 0.04) {
            const a = Math.random() * Math.PI * 2;
            tgt.vx += Math.cos(a) * 1.3; tgt.vy += Math.sin(a) * 1.3;
        }
        tgt.vx += (Math.random() - 0.5) * 0.4;
        tgt.vy += (Math.random() - 0.5) * 0.4;

        const wm = 80;
        if (tgt.x < wm) tgt.vx += (wm - tgt.x) * 0.01;
        if (tgt.x > W - wm) tgt.vx -= (tgt.x - (W - wm)) * 0.01;
        if (tgt.y < wm) tgt.vy += (wm - tgt.y) * 0.01;
        if (tgt.y > H - wm) tgt.vy -= (tgt.y - (H - wm)) * 0.01;

        tgt.vx *= 0.94; tgt.vy *= 0.94;
        const ts = Math.sqrt(tgt.vx * tgt.vx + tgt.vy * tgt.vy);
        if (ts > TARGET_SPEED) { tgt.vx = (tgt.vx / ts) * TARGET_SPEED; tgt.vy = (tgt.vy / ts) * TARGET_SPEED; }
        tgt.x += tgt.vx; tgt.y += tgt.vy;

        if (tgt.x < 15) { tgt.x = 15; tgt.vx = Math.abs(tgt.vx); }
        if (tgt.x > W - 15) { tgt.x = W - 15; tgt.vx = -Math.abs(tgt.vx); }
        if (tgt.y < 15) { tgt.y = 15; tgt.vy = Math.abs(tgt.vy); }
        if (tgt.y > H - 15) { tgt.y = H - 15; tgt.vy = -Math.abs(tgt.vy); }
    }

    return edges;
}

// --- Capture ---

function computeEncirclement(tIdx) {
    const tgt = targets[tIdx];
    if (tgt.captured) return { score: 0 };

    const nearby = [];
    for (let i = 0; i < nAgents; i++) {
        if (dist(agents[i], tgt) < SURROUND_RADIUS * 1.8) nearby.push(agents[i]);
    }
    if (nearby.length < 2) return { score: 0 };

    const angles = nearby.map(a => Math.atan2(a.y - tgt.y, a.x - tgt.x)).sort((a, b) => a - b);
    let maxGap = 0;
    for (let i = 0; i < angles.length; i++) {
        let gap = angles[(i + 1) % angles.length] - angles[i];
        if (gap < 0) gap += Math.PI * 2;
        maxGap = Math.max(maxGap, gap);
    }

    const coverage = 1.0 - maxGap / (Math.PI * 2);
    const avgDist = nearby.reduce((s, a) => s + dist(a, tgt), 0) / nearby.length;
    const closeness = Math.max(0, 1 - avgDist / 180);
    return { score: coverage * closeness, count: nearby.length };
}

function checkCaptures() {
    let any = false;
    for (let t = 0; t < nTargets; t++) {
        if (targets[t].captured) continue;
        if (computeEncirclement(t).score > CAPTURE_THRESHOLD) {
            targets[t].captured = true;
            any = true;
        }
    }
    if (any) {
        captureFlash = 20;
        if (targets.every(t => t.captured)) {
            captures[mode]++;
            document.getElementById('score-' + mode).textContent = captures[mode];
            setTimeout(() => initSim(), 700);
        }
    }
}

// --- Drawing ---

function drawGrid() {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.025)';
    ctx.lineWidth = 1;
    for (let x = 50; x < W; x += 50) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
    for (let y = 50; y < H; y += 50) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
}

function drawTargetObj(tgt, idx) {
    const fill = T_FILL[idx % T_FILL.length];
    const stroke = T_STROKE[idx % T_STROKE.length];
    const glowRGB = T_GLOW[idx % T_GLOW.length];

    if (tgt.captured) {
        ctx.globalAlpha = 0.25;
        ctx.beginPath();
        ctx.arc(tgt.x, tgt.y, TARGET_RADIUS, 0, Math.PI * 2);
        ctx.fillStyle = '#555';
        ctx.fill();
        ctx.fillStyle = '#aaa';
        ctx.font = 'bold 12px Menlo';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('✓', tgt.x, tgt.y + 1);
        ctx.globalAlpha = 1;
        return;
    }

    // Glow
    const glow = ctx.createRadialGradient(tgt.x, tgt.y, 0, tgt.x, tgt.y, TARGET_RADIUS * 3.5);
    glow.addColorStop(0, `rgba(${glowRGB}, 0.35)`);
    glow.addColorStop(1, `rgba(${glowRGB}, 0)`);
    ctx.beginPath();
    ctx.arc(tgt.x, tgt.y, TARGET_RADIUS * 3.5, 0, Math.PI * 2);
    ctx.fillStyle = glow;
    ctx.fill();

    // Surround progress ring
    const enc = computeEncirclement(idx);
    if (enc.score > 0.05) {
        ctx.beginPath();
        ctx.arc(tgt.x, tgt.y, SURROUND_RADIUS, -Math.PI / 2, -Math.PI / 2 + enc.score * Math.PI * 2);
        ctx.strokeStyle = `rgba(100, 255, 130, ${0.2 + enc.score * 0.6})`;
        ctx.lineWidth = 3;
        ctx.stroke();
    }

    // Body — diamond shape for targets (distinct from circular agents)
    ctx.save();
    ctx.translate(tgt.x, tgt.y);
    ctx.rotate(Math.PI / 4);
    ctx.beginPath();
    const s = TARGET_RADIUS * 0.85;
    ctx.rect(-s, -s, s * 2, s * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();

    // Label
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 9px Menlo';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('T' + idx, tgt.x, tgt.y);
}

function drawAgentObj(a, index) {
    // Outer glow
    const glow = ctx.createRadialGradient(a.x, a.y, 0, a.x, a.y, AGENT_RADIUS * 2.2);
    glow.addColorStop(0, `hsla(${a.hue}, 60%, 55%, 0.25)`);
    glow.addColorStop(1, `hsla(${a.hue}, 60%, 55%, 0)`);
    ctx.beginPath();
    ctx.arc(a.x, a.y, AGENT_RADIUS * 2.2, 0, Math.PI * 2);
    ctx.fillStyle = glow;
    ctx.fill();

    // Assignment ring (thick, colored to match target)
    if (a.assignedTarget >= 0) {
        const ringColor = T_FILL[a.assignedTarget % T_FILL.length];
        ctx.beginPath();
        ctx.arc(a.x, a.y, AGENT_RADIUS + 3, 0, Math.PI * 2);
        ctx.strokeStyle = ringColor;
        ctx.lineWidth = 3;
        ctx.stroke();
    }

    // Body
    ctx.beginPath();
    ctx.arc(a.x, a.y, AGENT_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = `hsl(${a.hue}, 50%, 45%)`;
    ctx.fill();
    ctx.strokeStyle = `hsl(${a.hue}, 50%, 65%)`;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Label
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 10px Menlo';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(index, a.x, a.y);

    // "?" if no knowledge
    if (a.knownTargets.length === 0) {
        ctx.fillStyle = '#ffcc33';
        ctx.font = 'bold 14px Menlo';
        ctx.fillText('?', a.x, a.y - 22);
    }
}

function drawEdge(a, b, color) {
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2;
    ctx.stroke();
}

function drawFiedlerChart() {
    const cw = fiedlerCanvas.width, ch = fiedlerCanvas.height;
    fctx.clearRect(0, 0, cw, ch);
    if (fiedlerHistory.length < 2) return;

    const maxVal = Math.max(2, ...fiedlerHistory) * 1.1;
    const step = cw / Math.max(fiedlerHistory.length - 1, 1);

    const threshY = ch - (0.12 / maxVal) * ch;
    fctx.beginPath(); fctx.moveTo(0, threshY); fctx.lineTo(cw, threshY);
    fctx.strokeStyle = 'rgba(255, 80, 80, 0.3)'; fctx.lineWidth = 1;
    fctx.setLineDash([4, 4]); fctx.stroke(); fctx.setLineDash([]);
    fctx.fillStyle = 'rgba(255, 80, 80, 0.5)'; fctx.font = '9px Menlo';
    fctx.fillText('disconnect', cw - 58, threshY - 4);

    fctx.beginPath();
    for (let i = 0; i < fiedlerHistory.length; i++) {
        const x = i * step, y = ch - (fiedlerHistory[i] / maxVal) * ch;
        if (i === 0) fctx.moveTo(x, y); else fctx.lineTo(x, y);
    }
    fctx.strokeStyle = '#4CAF50'; fctx.lineWidth = 1.5; fctx.stroke();
    fctx.fillStyle = '#666'; fctx.font = '9px Menlo'; fctx.fillText('Fiedler value', 4, 12);
}

// --- Main render ---

function render() {
    ctx.clearRect(0, 0, W, H);

    if (captureFlash > 0) {
        ctx.fillStyle = `rgba(80, 255, 80, ${captureFlash / 40})`;
        ctx.fillRect(0, 0, W, H);
        captureFlash--;
    }

    drawGrid();

    const edges = getCommEdges();
    const adj = buildAdjacency(edges);
    const fv = fiedlerValue(adj);
    const totalPossible = nAgents * (nAgents - 1) / 2;

    // Track bandwidth
    totalMessages += edges.length;
    totalMaxMessages += totalPossible;

    // Edges
    for (const [i, j] of edges) {
        const color = fv < 0.12 ? 'rgba(255, 80, 80, 0.25)' : 'rgba(80, 200, 100, 0.25)';
        drawEdge(agents[i], agents[j], color);
    }

    // Targets (draw first so agents appear on top)
    for (let t = 0; t < nTargets; t++) drawTargetObj(targets[t], t);

    // Agents
    for (let i = 0; i < nAgents; i++) drawAgentObj(agents[i], i);

    // Stats
    const modeNames = { none: 'No Comm', broadcast: 'CommNet', tarmac: 'TarMAC', gated: 'IC3Net', ours: 'Gated + Conn.' };
    const capturedCount = targets.filter(t => t.captured).length;

    document.getElementById('stat-mode').textContent = modeNames[mode];
    document.getElementById('stat-fiedler').textContent = mode === 'none' ? 'N/A' : fv.toFixed(3);
    document.getElementById('stat-fiedler').style.color = (mode === 'none' || fv < 0.12) ? '#e55' : '#4CAF50';
    document.getElementById('stat-edges').textContent = edges.length + ' / ' + totalPossible;
    document.getElementById('stat-comm').textContent = totalPossible > 0 ? ((edges.length / totalPossible) * 100).toFixed(0) + '%' : '0%';
    document.getElementById('stat-encircle').textContent = capturedCount + ' / ' + nTargets;
    document.getElementById('stat-encircle').style.color = capturedCount === nTargets ? '#4CAF50' : '#FF9800';
    document.getElementById('stat-captures').textContent = captures[mode];

    // Bandwidth display
    const savings = totalMaxMessages > 0 ? (1 - totalMessages / totalMaxMessages) * 100 : 0;
    const usage = totalMaxMessages > 0 ? (totalMessages / totalMaxMessages) * 100 : 0;
    document.getElementById('bw-total').textContent = totalMessages.toLocaleString();
    document.getElementById('bw-max').textContent = totalMaxMessages.toLocaleString();
    document.getElementById('bw-savings').textContent = savings.toFixed(0) + '%';
    document.getElementById('bw-savings').style.color = savings > 30 ? '#4CAF50' : savings > 10 ? '#FF9800' : '#e55';
    const barFill = document.getElementById('bw-bar-fill');
    barFill.style.width = usage.toFixed(1) + '%';
    barFill.style.background = savings > 30 ? '#4CAF50' : savings > 10 ? '#FF9800' : '#2196F3';

    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    document.getElementById('stat-time').textContent = elapsed + 's';

    if (frame % 3 === 0) {
        fiedlerHistory.push(mode === 'none' ? 0 : fv);
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
    initSim();
}

// --- Loop ---

function loop() {
    updateAgents();
    if (frame % 20 === 0) checkCaptures();
    render();
    frame++;
    requestAnimationFrame(loop);
}

initSim();
loop();
