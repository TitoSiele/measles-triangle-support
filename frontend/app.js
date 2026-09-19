const API_BASE = (location.hostname === "127.0.0.1" || location.hostname === "localhost") ? "http://127.0.0.1:8000" : "";

// Numeric inputs: blank = unknown.
const NUM_FIELDS = {
  fever_temp_c: { label: "Fever temperature", min: 30, max: 45, int: false },
  fever_duration_days: { label: "Fever duration", min: 0, max: 60, int: true },
  rash_onset_day: { label: "Rash onset day", min: 0, max: 30, int: true },
  age_years: { label: "Age", min: 0, max: 120, int: false },
};

// Yes / No / Unknown selects (default Unknown).
const YN_FIELDS = [
  ["cough", "Cough"],
  ["coryza", "Runny nose (coryza)"],
  ["conjunctivitis", "Red/watery eyes (conjunctivitis)"],
  ["koplik_spots", "Koplik spots (white spots inside mouth)"],
  ["rash_present", "Rash present"],
  ["cephalocaudal_spread", "Rash spreading head-to-toe"],
  ["sore_throat", "Sore throat"],
  ["itchy_rash", "Itchy rash"],
  ["lymphadenopathy", "Swollen lymph nodes"],
  ["vaccinated", "Vaccinated against measles"],
];

const FIELD_LABELS = {};
Object.entries(NUM_FIELDS).forEach(([id, spec]) => { FIELD_LABELS[id] = spec.label; });
YN_FIELDS.forEach(([id, label]) => { FIELD_LABELS[id] = label; });

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function buildForm() {
  const box = document.getElementById("yn-fields");
  box.innerHTML = "";
  YN_FIELDS.forEach(([id, label]) => {
    const row = document.createElement("div");
    row.className = "sym-row";
    row.innerHTML =
      `<label for="${id}">${esc(label)}</label>` +
      `<select id="${id}">` +
      `<option value="">Unknown</option><option value="1">Yes</option><option value="0">No</option>` +
      `</select>`;
    box.appendChild(row);
    const sel = row.querySelector("select");
    sel.addEventListener("change", () => sel.classList.toggle("answered", sel.value !== ""));
  });
}

function readNumber(id) {
  const spec = NUM_FIELDS[id];
  const raw = document.getElementById(id).value.trim();
  if (raw === "") return null;
  const v = parseFloat(raw);
  if (Number.isNaN(v) || v < spec.min || v > spec.max) {
    throw new Error(`${spec.label} must be between ${spec.min} and ${spec.max}, or left blank if unknown.`);
  }
  return spec.int ? Math.round(v) : v;
}

function readYesNo(id) {
  const raw = document.getElementById(id).value;
  return raw === "" ? null : parseInt(raw, 10);
}

function collectValues() {
  const values = {};
  Object.keys(NUM_FIELDS).forEach(id => { values[id] = readNumber(id); });
  YN_FIELDS.forEach(([id]) => { values[id] = readYesNo(id); });
  return values;
}

function resetForm() {
  Object.keys(NUM_FIELDS).forEach(id => { document.getElementById(id).value = ""; });
  YN_FIELDS.forEach(([id]) => {
    const sel = document.getElementById(id);
    sel.value = "";
    sel.classList.remove("answered");
  });
  document.getElementById("imageFile").value = "";
  const resultDiv = document.getElementById("result");
  resultDiv.style.display = "none";
  resultDiv.innerHTML = "";
}

function pct(p) {
  return p === null || p === undefined ? null : Math.round(p * 100);
}

function renderResult(resultDiv, data, unknownCount) {
  const unknownNames = (data.unknown_key_fields || []).map(f => FIELD_LABELS[f] || f);
  const imagePct = pct(data.image_measles_probability);
  const disclaimer = `
    <p style="margin-top:16px; padding-top:12px; border-top:1px solid rgba(0,0,0,0.1);">
      <small><strong>${esc(data.disclaimer)}</strong></small>
    </p>`;

  if (data.insufficient_information) {
    resultDiv.className = "card flag-refer";
    resultDiv.innerHTML = `
      <h3>Insufficient information — clinical assessment recommended</h3>
      <p><span class="confidence-tag conf-Insufficient">No score given</span></p>
      <p>Key findings are unknown: <strong>${esc(unknownNames.join(", ") || "several")}</strong>.
      The tool does not give a score when too many key findings are missing, because the result would not be reliable.
      Please assess the patient clinically and record the missing findings if they become available.</p>
      ${imagePct !== null ? `<p><small>Image-based estimate (reference only, not used for triage): ${imagePct}%</small></p>` : ""}
      ${disclaimer}`;
    return;
  }

  const combined = data.combined_measles_probability;
  const score = pct(combined);
  let flagClass = "flag-low";
  if (data.signals_disagree || combined >= 0.6) flagClass = "flag-high";
  else if (combined >= 0.3) flagClass = "flag-moderate";

  const unknownNote = unknownCount > 0
    ? `<p><small>${unknownCount} finding${unknownCount === 1 ? " was" : "s were"} marked unknown` +
      (unknownNames.length ? ` (including key: ${esc(unknownNames.join(", "))})` : "") +
      `. Missing information makes the score less reliable.</small></p>`
    : "";

  resultDiv.className = "card " + flagClass;
  resultDiv.innerHTML = `
    <h3>${esc(data.triage_flag)}</h3>
    <p>
      <strong>Combined measles-consistency score: ${score}%</strong>
      <span class="confidence-tag conf-${esc(data.confidence)}">${esc(data.confidence)} confidence</span>
    </p>
    <div class="prob-bar"><div class="prob-fill" style="width:${score}%"></div></div>
    ${unknownNote}
    <p style="margin-top:16px"><small>
      Symptom-based estimate: ${pct(data.symptom_measles_probability) !== null ? pct(data.symptom_measles_probability) + "%" : "N/A"}<br>
      Image-based estimate: ${imagePct !== null ? imagePct + "%" : "N/A (no image provided or model unavailable)"}
    </small></p>
    ${disclaimer}`;
}

async function runPrediction() {
  const resultDiv = document.getElementById("result");
  const btn = document.getElementById("runBtn");
  resultDiv.style.display = "block";
  resultDiv.className = "card";

  let values;
  try {
    values = collectValues();
  } catch (e) {
    resultDiv.innerHTML = `<strong>Please check the form:</strong> ${esc(e.message)}`;
    return;
  }

  resultDiv.innerHTML = "Running assessment...";
  btn.disabled = true;

  // Unknown values are simply left out of the request; the API treats them as unknown.
  const params = new URLSearchParams();
  let unknownCount = 0;
  Object.entries(values).forEach(([field, v]) => {
    if (v === null) unknownCount += 1;
    else params.append(field, v);
  });

  const imageFile = document.getElementById("imageFile").files[0];
  const formData = new FormData();
  if (imageFile) formData.append("file", imageFile);

  try {
    const response = await fetch(`${API_BASE}/predict/combined?${params.toString()}`, {
      method: "POST",
      body: imageFile ? formData : undefined,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      const detail = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail || response.statusText);
      resultDiv.innerHTML = `<strong>Error:</strong> ${esc(detail)}`;
      return;
    }

    renderResult(resultDiv, await response.json(), unknownCount);
  } catch (e) {
    resultDiv.innerHTML = `<strong>Error:</strong> Could not reach the API. Is the server running at ${esc(API_BASE)}? (${esc(e.message)})`;
  } finally {
    btn.disabled = false;
  }
}

buildForm();
