document.addEventListener("DOMContentLoaded", () => {
    const statusEl = document.getElementById("status");
    const confidenceEl = document.getElementById("confidence");
  
    // Simulate a risk result from analysis
    const riskLevel = "danger"; // could be "safe", "warning", "danger"
    const confidenceScore = 82;
  
    // Update UI
    statusEl.textContent = riskLevel === "danger" ? "❌ Phishing" :
                           riskLevel === "warning" ? "⚠️ Suspicious" : "✅ Safe";
  
    statusEl.className = riskLevel === "danger" ? "danger" :
                         riskLevel === "warning" ? "warning" : "safe";
  
    confidenceEl.textContent = confidenceScore + "%";
  });
  