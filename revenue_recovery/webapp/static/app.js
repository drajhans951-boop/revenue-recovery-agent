const ACTION_ICONS = {
  compliance_stop: "\u{1F6D1}",
  give_up_unlikely: "⚠️",
};

function iconFor(row) {
  if (row.action in ACTION_ICONS) return ACTION_ICONS[row.action];
  return row.recovered ? "✅" : "❌";
}

function fmtRs(n) {
  return "Rs." + Number(n).toLocaleString("en-IN", { maximumFractionDigits: 0 });
}

function setStatus(state, text) {
  const dot = document.getElementById("statusDot");
  dot.className = "status-dot" + (state ? " " + state : "");
  document.getElementById("statusText").textContent = text;
}

function appendTraceRow(row) {
  const trace = document.getElementById("trace");
  const hint = trace.querySelector(".empty-hint");
  if (hint) hint.remove();

  const el = document.createElement("div");
  el.className = "trace-row";
  el.innerHTML = `
    <span class="icon">${iconFor(row)}</span>
    <span class="txn">${row.txn_id}</span>
    <span class="amount">${fmtRs(row.amount)}</span>
    <span class="action">${row.action}</span>
    <span class="prob">${(row.predicted_prob * 100).toFixed(1)}%</span>
    <span class="ev">${fmtRs(row.expected_value)}</span>
    <span class="reason" title="${row.reasoning.replace(/"/g, "&quot;")}">${row.reasoning}${row.message ? " &mdash; “" + row.message + "”" : ""}</span>
  `;
  trace.appendChild(el);
  trace.scrollTop = trace.scrollHeight;
}

function renderBarChart(containerId, items, colorFor) {
  const container = document.getElementById(containerId);
  const max = Math.max(...items.map((i) => i.value), 1);
  container.innerHTML = items
    .map((item) => {
      const pct = Math.max(2, (item.value / max) * 100);
      return `
        <div class="bar-row">
          <span class="bar-label">${item.label}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${colorFor(item)}"></div></div>
          <span class="bar-value">${fmtRs(item.value)}</span>
        </div>`;
    })
    .join("");
}

function renderSummary(summary) {
  document.getElementById("statAtRisk").textContent = fmtRs(summary.total_at_risk);
  document.getElementById("statRecovered").textContent = fmtRs(summary.total_recovered);
  document.getElementById("statRate").textContent = (summary.recovery_rate * 100).toFixed(1) + "%";
  document.getElementById("statCount").textContent = summary.transactions;
  document.getElementById("statHardStop").textContent = summary.hard_stops;
  document.getElementById("statGiveUp").textContent = summary.soft_giveups;

  renderBarChart(
    "riskVsRecoveredChart",
    [
      { label: "Revenue at risk", value: summary.total_at_risk },
      { label: "Revenue recovered", value: summary.total_recovered },
    ],
    (item) => (item.label.includes("recovered") ? "var(--green)" : "var(--red)")
  );

  const byCode = Object.entries(summary.recovered_by_code)
    .sort((a, b) => b[1] - a[1])
    .map(([label, value]) => ({ label, value }));
  renderBarChart("byCodeChart", byCode, () => "var(--gold)");
}

function runAgent() {
  const trace = document.getElementById("trace");
  trace.innerHTML = "";
  setStatus("running", "running…");
  document.getElementById("runBtn").disabled = true;

  const source = new EventSource("/api/run");

  source.addEventListener("decision", (e) => {
    appendTraceRow(JSON.parse(e.data));
  });

  source.addEventListener("summary", (e) => {
    renderSummary(JSON.parse(e.data));
    setStatus("done", "done");
    document.getElementById("runBtn").disabled = false;
    source.close();
  });

  source.onerror = () => {
    setStatus("", "error");
    document.getElementById("runBtn").disabled = false;
    source.close();
  };
}

async function newBatch() {
  document.getElementById("newBatchBtn").disabled = true;
  setStatus("running", "generating batch…");
  const res = await fetch("/api/generate", { method: "POST" });
  const data = await res.json();
  setStatus("", `new batch: ${data.batch_rows} transactions, ${fmtRs(data.total_at_risk)} at risk`);
  document.getElementById("newBatchBtn").disabled = false;

  const trace = document.getElementById("trace");
  trace.innerHTML = '<div class="empty-hint">Click "Run Agent" to watch it process the batch, transaction by transaction.</div>';
}

function pct(x) {
  return x === null || x === undefined ? "—" : (x * 100).toFixed(1) + "%";
}

async function loadCalibration() {
  const el = document.getElementById("calibrationContent");
  el.innerHTML = '<div class="empty-hint">Running 5-fold cross-validation, refitting per-code models on each fold…</div>';
  const res = await fetch("/api/calibration");
  const data = await res.json();

  const bucketRows = data.buckets
    .map(
      (b) => `
      <tr>
        <td>${b.bucket}</td>
        <td>${b.avg_n.toFixed(1)}</td>
        <td>${pct(b.avg_predicted)}</td>
        <td>${pct(b.actual_rate)}</td>
      </tr>`
    )
    .join("");

  const codeRows = Object.entries(data.per_code)
    .map(
      ([code, s]) => `
      <tr>
        <td>${code}</td>
        <td>${pct(s.model_acc_mean)} &plusmn;${(s.model_acc_std * 100).toFixed(1)}</td>
        <td>${pct(s.naive_acc_mean)} &plusmn;${(s.naive_acc_std * 100).toFixed(1)}</td>
        <td style="color:${s.advantage > 0.001 ? "var(--green)" : "var(--text-dim)"}">${s.advantage >= 0 ? "+" : ""}${(s.advantage * 100).toFixed(1)}pp</td>
      </tr>`
    )
    .join("");

  el.innerHTML = `
    <p style="font-size:12px;color:var(--text-dim);font-family:-apple-system,sans-serif;">
      ${data.n_folds}-fold CV &middot; Model accuracy:
      <strong style="color:var(--gold)">${pct(data.model_acc_mean)} &plusmn;${(data.model_acc_std * 100).toFixed(1)}</strong>
      &middot; Naive per-code baseline: ${pct(data.naive_acc_mean)} &plusmn;${(data.naive_acc_std * 100).toFixed(1)}
      &middot; Advantage: <strong style="color:${data.advantage > 0.001 ? "var(--green)" : "var(--text-dim)"}">${data.advantage >= 0 ? "+" : ""}${(data.advantage * 100).toFixed(1)}pp</strong>
    </p>
    <table>
      <thead><tr><th>Failure code</th><th>Model acc</th><th>Naive acc</th><th>Advantage</th></tr></thead>
      <tbody>${codeRows}</tbody>
    </table>
    <p style="font-size:12px;color:var(--text-dim);font-family:-apple-system,sans-serif;margin-top:16px;">Calibration (predicted vs. actual, averaged across folds):</p>
    <table>
      <thead><tr><th>Predicted bucket</th><th>Avg n/fold</th><th>Avg predicted</th><th>Actual success rate</th></tr></thead>
      <tbody>${bucketRows}</tbody>
    </table>
  `;
}

async function loadGuardrailTests() {
  const el = document.getElementById("guardrailsContent");
  el.innerHTML = '<div class="empty-hint">Running pytest test_guardrails.py…</div>';
  const res = await fetch("/api/guardrail-tests");
  const data = await res.json();

  const rows = data.tests
    .map(
      (t) => `
      <tr>
        <td>${t.name}</td>
        <td><span class="badge ${t.passed ? "pass" : "fail"}">${t.passed ? "PASSED" : "FAILED"}</span></td>
      </tr>`
    )
    .join("");

  el.innerHTML = `
    <p style="font-size:12px;color:var(--text-dim);font-family:-apple-system,sans-serif;">
      <span class="badge ${data.all_passed ? "pass" : "fail"}">${data.passed_count}/${data.total_count} passed</span>
    </p>
    <table>
      <thead><tr><th>Test</th><th>Result</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

document.getElementById("runBtn").addEventListener("click", runAgent);
document.getElementById("newBatchBtn").addEventListener("click", newBatch);

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
    btn.classList.add("active");
    const tab = document.getElementById("tab-" + btn.dataset.tab);
    tab.classList.add("active");

    if (btn.dataset.tab === "calibration" && !tab.dataset.loaded) {
      tab.dataset.loaded = "1";
      loadCalibration();
    }
    if (btn.dataset.tab === "guardrails" && !tab.dataset.loaded) {
      tab.dataset.loaded = "1";
      loadGuardrailTests();
    }
  });
});
