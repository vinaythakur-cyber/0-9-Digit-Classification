/* Canvas drawing + calls to the prediction API.
 *
 * The canvas is deliberately white-ink-on-black, which is MNIST's own
 * convention. The server can auto-detect and flip polarity for uploads, but
 * for the drawing path it is better to simply produce the right thing than to
 * rely on a heuristic.
 */

const CANVAS_SIZE = 280;
// MNIST strokes are roughly 2-3 px inside a 28x28 frame. This canvas is 10x
// that, so ~22 px reproduces the same relative stroke weight the model saw in
// training. Too thin and the digit nearly vanishes when downsampled to 28x28.
const STROKE_WIDTH = 22;

const DIGIT_LABELS = ["0","1","2","3","4","5","6","7","8","9","unclassifiable"];

const pad = document.getElementById("pad");
const ctx = pad.getContext("2d", { willReadFrequently: true });

function resetCanvas() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = STROKE_WIDTH;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
}
resetCanvas();

/* ---------------- drawing ---------------- */
let drawing = false;
let last = null;

function posOf(event) {
  const rect = pad.getBoundingClientRect();
  // The canvas is CSS-scalable, so map client coords through the actual
  // displayed size rather than assuming 1:1 with the backing store.
  return {
    x: (event.clientX - rect.left) * (CANVAS_SIZE / rect.width),
    y: (event.clientY - rect.top) * (CANVAS_SIZE / rect.height),
  };
}

pad.addEventListener("pointerdown", (e) => {
  drawing = true;
  pad.setPointerCapture(e.pointerId);
  last = posOf(e);
  // A dot, so a single tap still leaves ink.
  ctx.beginPath();
  ctx.arc(last.x, last.y, STROKE_WIDTH / 2, 0, Math.PI * 2);
  ctx.fillStyle = "#fff";
  ctx.fill();
});

pad.addEventListener("pointermove", (e) => {
  if (!drawing) return;
  const p = posOf(e);
  // Quadratic through the midpoint smooths the polyline; straight segments
  // between raw pointer samples produce visible corners at fast strokes.
  const mid = { x: (last.x + p.x) / 2, y: (last.y + p.y) / 2 };
  ctx.beginPath();
  ctx.moveTo(last.x, last.y);
  ctx.quadraticCurveTo(last.x, last.y, mid.x, mid.y);
  ctx.stroke();
  last = p;
});

for (const evt of ["pointerup", "pointercancel", "pointerleave"]) {
  pad.addEventListener(evt, () => { drawing = false; last = null; });
}

/* ---------------- API ---------------- */
const resultEl = document.getElementById("result");
const detailEl = document.getElementById("detail");
const barsEl = document.getElementById("bars");
const statusEl = document.getElementById("status");

async function classify() {
  const button = document.getElementById("predict");
  button.disabled = true;
  button.textContent = "…";
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: pad.toDataURL("image/png") }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    render(await res.json());
  } catch (err) {
    showError(err.message);
  } finally {
    button.disabled = false;
    button.textContent = "Classify";
  }
}

function showError(message) {
  detailEl.hidden = true;
  resultEl.className = "result empty";
  resultEl.innerHTML = `<p class="placeholder">${message}</p>`;
}

function render(p) {
  resultEl.className = "result";
  if (p.is_digit) {
    resultEl.innerHTML =
      `<p class="verdict digit">${p.label}</p>` +
      `<p class="confidence">${(p.confidence * 100).toFixed(1)}% confident</p>`;
  } else {
    resultEl.innerHTML =
      `<p class="verdict rejected">unclassifiable</p>` +
      `<p class="confidence">not a digit</p>`;
  }

  barsEl.innerHTML = "";
  const top = DIGIT_LABELS[
    DIGIT_LABELS.map((l) => p.probabilities[l])
      .reduce((best, v, i, a) => (v > a[best] ? i : best), 0)
  ];
  for (const label of DIGIT_LABELS) {
    const value = p.probabilities[label] ?? 0;
    const row = document.createElement("div");
    row.className = "bar-row"
      + (label === "unclassifiable" ? " unknown" : "")
      + (label === top ? " top" : "");
    row.innerHTML =
      `<span class="bar-label">${label}</span>` +
      `<span class="bar-track"><span class="bar-fill" style="width:${(value * 100).toFixed(2)}%"></span></span>` +
      `<span class="bar-value">${(value * 100).toFixed(1)}%</span>`;
    barsEl.appendChild(row);
  }

  document.getElementById("reason").textContent = p.reason;
  document.getElementById("energy").textContent = p.energy.toFixed(2);
  document.getElementById("punk").textContent =
    (p.unknown_probability * 100).toFixed(1) + "%";
  detailEl.hidden = false;
}

/* ---------------- sample generators ---------------- */
async function loadSample(kind) {
  const res = await fetch(`/api/sample/${kind}`);
  if (!res.ok) return showError(`could not generate a ${kind}`);
  const { image } = await res.json();
  const img = new Image();
  img.onload = () => {
    resetCanvas();
    ctx.drawImage(img, 0, 0, CANVAS_SIZE, CANVAS_SIZE);
    classify();
  };
  img.src = image;
}

/* ---------------- wiring ---------------- */
document.getElementById("predict").addEventListener("click", classify);
document.getElementById("clear").addEventListener("click", () => {
  resetCanvas();
  detailEl.hidden = true;
  resultEl.className = "result empty";
  resultEl.innerHTML = `<p class="placeholder">Draw something and press <strong>Classify</strong>.</p>`;
});

document.getElementById("samples").addEventListener("click", (e) => {
  const kind = e.target.dataset.kind;
  if (kind) loadSample(kind);
});

document.getElementById("file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const body = new FormData();
  body.append("file", file);
  try {
    const res = await fetch("/api/predict-file", { method: "POST", body });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    render(await res.json());
  } catch (err) {
    showError(err.message);
  }
  e.target.value = "";   // let the same file be picked again
});

/* Report model status up front: a missing checkpoint should say so plainly
   rather than failing on the first click. */
fetch("/api/health").then(async (res) => {
  const data = await res.json();
  if (res.ok) {
    const t = data.test_metrics || {};
    statusEl.textContent =
      `model ready — ${data.parameters.toLocaleString()} parameters` +
      (t.digit_accuracy
        ? ` · ${(t.digit_accuracy * 100).toFixed(1)}% digit accuracy · ` +
          `${(t.unknown_recall * 100).toFixed(1)}% rejection recall`
        : "");
  } else {
    statusEl.textContent = data.detail || "model not loaded";
  }
}).catch(() => { statusEl.textContent = "API unreachable"; });
