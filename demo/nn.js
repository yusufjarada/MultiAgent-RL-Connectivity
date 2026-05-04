// ============================================================
// Minimal neural network inference in JavaScript
// Runs trained PyTorch models exported as JSON weight matrices
// ============================================================

// --- Tensor operations (flat arrays with shape tracking) ---

function zeros(shape) {
    const size = shape.reduce((a, b) => a * b, 1);
    return { data: new Float32Array(size), shape: [...shape] };
}

function fromArray(arr) {
    // Convert nested JS array to flat tensor
    if (typeof arr[0] === 'number') {
        return { data: new Float32Array(arr), shape: [arr.length] };
    }
    if (typeof arr[0][0] === 'number') {
        const rows = arr.length, cols = arr[0].length;
        const data = new Float32Array(rows * cols);
        for (let i = 0; i < rows; i++)
            for (let j = 0; j < cols; j++)
                data[i * cols + j] = arr[i][j];
        return { data, shape: [rows, cols] };
    }
    throw new Error('Only 1D and 2D arrays supported');
}

function linear(input, weight, bias) {
    // input: (N, in_dim), weight: (out_dim, in_dim), bias: (out_dim,)
    // output: (N, out_dim)
    const N = input.shape[0];
    const inDim = weight.shape[1];
    const outDim = weight.shape[0];
    const out = zeros([N, outDim]);

    for (let n = 0; n < N; n++) {
        for (let o = 0; o < outDim; o++) {
            let sum = bias ? bias.data[o] : 0;
            for (let i = 0; i < inDim; i++) {
                sum += input.data[n * inDim + i] * weight.data[o * inDim + i];
            }
            out.data[n * outDim + o] = sum;
        }
    }
    return out;
}

function relu(t) {
    const out = { data: new Float32Array(t.data.length), shape: [...t.shape] };
    for (let i = 0; i < t.data.length; i++) {
        out.data[i] = Math.max(0, t.data[i]);
    }
    return out;
}

function sigmoid(t) {
    const out = { data: new Float32Array(t.data.length), shape: [...t.shape] };
    for (let i = 0; i < t.data.length; i++) {
        out.data[i] = 1.0 / (1.0 + Math.exp(-t.data[i]));
    }
    return out;
}

function softmax1D(arr, offset, length) {
    let max = -Infinity;
    for (let i = 0; i < length; i++) max = Math.max(max, arr[offset + i]);
    let sum = 0;
    const out = new Float32Array(length);
    for (let i = 0; i < length; i++) {
        out[i] = Math.exp(arr[offset + i] - max);
        sum += out[i];
    }
    for (let i = 0; i < length; i++) out[i] /= sum;
    return out;
}

function getRow(t, row) {
    // Get a single row from a 2D tensor as a 1D tensor
    const cols = t.shape[1];
    const data = new Float32Array(cols);
    for (let j = 0; j < cols; j++) data[j] = t.data[row * cols + j];
    return { data, shape: [cols] };
}

function setRow(t, row, values) {
    const cols = t.shape[1];
    for (let j = 0; j < cols; j++) t.data[row * cols + j] = values.data[j];
}

// --- Model runners ---
// Each takes (obs, params) where obs is (N, obs_dim) and returns { logits, edges, comm_rate }

function runCommNet(obs, params, nAgents) {
    const p = params;
    // Encoder: linear + relu
    let h = relu(linear(obs, fromArray(p['encoder.0.weight']), fromArray(p['encoder.0.bias'])));
    // Message
    const messages = linear(h, fromArray(p['msg_fn.weight']), fromArray(p['msg_fn.bias']));
    // Mean pool (exclude self)
    const msgDim = messages.shape[1];
    const hidDim = h.shape[1];
    const pooled = zeros([nAgents, msgDim]);
    for (let i = 0; i < nAgents; i++) {
        for (let d = 0; d < msgDim; d++) {
            let sum = 0;
            for (let j = 0; j < nAgents; j++) {
                if (j !== i) sum += messages.data[j * msgDim + d];
            }
            pooled.data[i * msgDim + d] = sum / (nAgents - 1);
        }
    }
    // Integrate: concat h and pooled, then linear + relu
    const concat = zeros([nAgents, hidDim + msgDim]);
    for (let i = 0; i < nAgents; i++) {
        for (let d = 0; d < hidDim; d++) concat.data[i * (hidDim + msgDim) + d] = h.data[i * hidDim + d];
        for (let d = 0; d < msgDim; d++) concat.data[i * (hidDim + msgDim) + hidDim + d] = pooled.data[i * msgDim + d];
    }
    h = relu(linear(concat, fromArray(p['integrate.0.weight']), fromArray(p['integrate.0.bias'])));
    // Action head
    const logits = linear(h, fromArray(p['action_head.weight']), fromArray(p['action_head.bias']));

    // All edges active (broadcast)
    const edges = [];
    for (let i = 0; i < nAgents; i++)
        for (let j = i + 1; j < nAgents; j++)
            edges.push([i, j]);

    return { logits, edges, commRate: 1.0 };
}

function runIC3Net(obs, params, nAgents) {
    const p = params;
    let h = relu(linear(obs, fromArray(p['encoder.0.weight']), fromArray(p['encoder.0.bias'])));
    // Gate
    const gateLogits = linear(h, fromArray(p['gate_fn.0.weight']), fromArray(p['gate_fn.0.bias']));
    const gateProbs = sigmoid(gateLogits);
    const gates = [];
    for (let i = 0; i < nAgents; i++) gates.push(gateProbs.data[i] > 0.5 ? 1 : 0);

    // Message
    const messages = linear(h, fromArray(p['msg_fn.weight']), fromArray(p['msg_fn.bias']));
    const msgDim = messages.shape[1];
    // Gate messages
    for (let i = 0; i < nAgents; i++) {
        if (!gates[i]) {
            for (let d = 0; d < msgDim; d++) messages.data[i * msgDim + d] = 0;
        }
    }
    // Mean pool active senders
    const hidDim = h.shape[1];
    const pooled = zeros([nAgents, msgDim]);
    for (let i = 0; i < nAgents; i++) {
        let count = 0;
        for (let j = 0; j < nAgents; j++) {
            if (j !== i && gates[j]) {
                count++;
                for (let d = 0; d < msgDim; d++) pooled.data[i * msgDim + d] += messages.data[j * msgDim + d];
            }
        }
        if (count > 0) {
            for (let d = 0; d < msgDim; d++) pooled.data[i * msgDim + d] /= count;
        }
    }
    // Integrate
    const concat = zeros([nAgents, hidDim + msgDim]);
    for (let i = 0; i < nAgents; i++) {
        for (let d = 0; d < hidDim; d++) concat.data[i * (hidDim + msgDim) + d] = h.data[i * hidDim + d];
        for (let d = 0; d < msgDim; d++) concat.data[i * (hidDim + msgDim) + hidDim + d] = pooled.data[i * msgDim + d];
    }
    h = relu(linear(concat, fromArray(p['integrate.0.weight']), fromArray(p['integrate.0.bias'])));
    const logits = linear(h, fromArray(p['action_head.weight']), fromArray(p['action_head.bias']));

    const edges = [];
    for (let i = 0; i < nAgents; i++)
        for (let j = i + 1; j < nAgents; j++)
            if (gates[i] || gates[j]) edges.push([i, j]);

    const total = nAgents * (nAgents - 1) / 2;
    return { logits, edges, commRate: edges.length / total, gates };
}

function runTarMAC(obs, params, nAgents) {
    const p = params;
    const nHeads = 4;
    let h = relu(linear(obs, fromArray(p['encoder.0.weight']), fromArray(p['encoder.0.bias'])));
    const hidDim = h.shape[1];

    // Q, K, V
    const Q = linear(h, fromArray(p['msg_query.weight']), fromArray(p['msg_query.bias']));
    const K = linear(h, fromArray(p['msg_key.weight']), fromArray(p['msg_key.bias']));
    const V = linear(h, fromArray(p['msg_value.weight']), fromArray(p['msg_value.bias']));
    const msgDim = Q.shape[1];
    const headDim = msgDim / nHeads;
    const scale = Math.sqrt(headDim);

    // Simple single-head attention for efficiency (multi-head adds complexity but same idea)
    // Compute attention: (N, N)
    const attended = zeros([nAgents, msgDim]);
    for (let i = 0; i < nAgents; i++) {
        // Compute scores
        const scores = new Float32Array(nAgents);
        for (let j = 0; j < nAgents; j++) {
            if (j === i) { scores[j] = -1e9; continue; }
            let dot = 0;
            for (let d = 0; d < msgDim; d++) dot += Q.data[i * msgDim + d] * K.data[j * msgDim + d];
            scores[j] = dot / scale;
        }
        // Softmax
        const weights = softmax1D(scores, 0, nAgents);
        // Weighted sum of V
        for (let d = 0; d < msgDim; d++) {
            let sum = 0;
            for (let j = 0; j < nAgents; j++) sum += weights[j] * V.data[j * msgDim + d];
            attended.data[i * msgDim + d] = sum;
        }
    }

    // Integrate
    const concat = zeros([nAgents, hidDim + msgDim]);
    for (let i = 0; i < nAgents; i++) {
        for (let d = 0; d < hidDim; d++) concat.data[i * (hidDim + msgDim) + d] = h.data[i * hidDim + d];
        for (let d = 0; d < msgDim; d++) concat.data[i * (hidDim + msgDim) + hidDim + d] = attended.data[i * msgDim + d];
    }
    h = relu(linear(concat, fromArray(p['integrate.0.weight']), fromArray(p['integrate.0.bias'])));
    const logits = linear(h, fromArray(p['action_head.weight']), fromArray(p['action_head.bias']));

    const edges = [];
    for (let i = 0; i < nAgents; i++)
        for (let j = i + 1; j < nAgents; j++)
            edges.push([i, j]);

    return { logits, edges, commRate: 1.0 };
}

function runGatedAttn(obs, params, nAgents) {
    const p = params;
    let h = relu(linear(obs, fromArray(p['encoder.0.weight']), fromArray(p['encoder.0.bias'])));
    const hidDim = h.shape[1];

    // Pairwise gates
    const gateW1 = fromArray(p['gate_fn.0.weight']);
    const gateB1 = fromArray(p['gate_fn.0.bias']);
    const gateW2 = fromArray(p['gate_fn.2.weight']);
    const gateB2 = fromArray(p['gate_fn.2.bias']);

    const gateMatrix = zeros([nAgents, nAgents]); // gate probs
    const gateHard = zeros([nAgents, nAgents]);

    for (let i = 0; i < nAgents; i++) {
        for (let j = 0; j < nAgents; j++) {
            if (i === j) continue;
            // Concat h[i] and h[j]
            const pair = zeros([1, hidDim * 2]);
            for (let d = 0; d < hidDim; d++) {
                pair.data[d] = h.data[i * hidDim + d];
                pair.data[hidDim + d] = h.data[j * hidDim + d];
            }
            let g = relu(linear(pair, gateW1, gateB1));
            g = linear(g, gateW2, gateB2);
            const prob = 1.0 / (1.0 + Math.exp(-g.data[0]));
            gateMatrix.data[i * nAgents + j] = prob;
            gateHard.data[i * nAgents + j] = prob > 0.5 ? 1 : 0;
        }
    }

    // Q, K, V
    const Q = linear(h, fromArray(p['msg_query.weight']), fromArray(p['msg_query.bias']));
    const K = linear(h, fromArray(p['msg_key.weight']), fromArray(p['msg_key.bias']));
    const V = linear(h, fromArray(p['msg_value.weight']), fromArray(p['msg_value.bias']));
    const msgDim = Q.shape[1];
    const scale = Math.sqrt(msgDim);

    // Gated attention
    const attended = zeros([nAgents, msgDim]);
    for (let i = 0; i < nAgents; i++) {
        const scores = new Float32Array(nAgents);
        let anyOpen = false;
        for (let j = 0; j < nAgents; j++) {
            if (j === i || gateHard.data[i * nAgents + j] === 0) {
                scores[j] = -1e9;
                continue;
            }
            anyOpen = true;
            let dot = 0;
            for (let d = 0; d < msgDim; d++) dot += Q.data[i * msgDim + d] * K.data[j * msgDim + d];
            scores[j] = dot / scale;
        }
        if (!anyOpen) continue; // no messages received
        const weights = softmax1D(scores, 0, nAgents);
        for (let d = 0; d < msgDim; d++) {
            let sum = 0;
            for (let j = 0; j < nAgents; j++) sum += weights[j] * V.data[j * msgDim + d];
            attended.data[i * msgDim + d] = sum;
        }
    }

    // Integrate
    const concat = zeros([nAgents, hidDim + msgDim]);
    for (let i = 0; i < nAgents; i++) {
        for (let d = 0; d < hidDim; d++) concat.data[i * (hidDim + msgDim) + d] = h.data[i * hidDim + d];
        for (let d = 0; d < msgDim; d++) concat.data[i * (hidDim + msgDim) + hidDim + d] = attended.data[i * msgDim + d];
    }
    h = relu(linear(concat, fromArray(p['integrate.0.weight']), fromArray(p['integrate.0.bias'])));
    const logits = linear(h, fromArray(p['action_head.weight']), fromArray(p['action_head.bias']));

    // Build edges from gates
    const edges = [];
    for (let i = 0; i < nAgents; i++)
        for (let j = i + 1; j < nAgents; j++)
            if (gateHard.data[i * nAgents + j] > 0 || gateHard.data[j * nAgents + i] > 0)
                edges.push([i, j]);

    const total = nAgents * (nAgents - 1) / 2;
    return { logits, edges, commRate: edges.length / total };
}

// --- Dispatch ---

function runModel(method, obs, params, nAgents) {
    if (method === 'commnet') return runCommNet(obs, params, nAgents);
    if (method === 'ic3net') return runIC3Net(obs, params, nAgents);
    if (method === 'tarmac') return runTarMAC(obs, params, nAgents);
    if (method === 'gated_attn') return runGatedAttn(obs, params, nAgents);
    return null;
}

// --- Sample from logits ---

function sampleAction(logits, agentIdx) {
    const actDim = logits.shape[1];
    const probs = softmax1D(logits.data, agentIdx * actDim, actDim);
    const r = Math.random();
    let cumsum = 0;
    for (let a = 0; a < actDim; a++) {
        cumsum += probs[a];
        if (r < cumsum) return a;
    }
    return actDim - 1;
}
