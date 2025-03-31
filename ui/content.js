function createSidebar(riskLevel, message, details = [], confidenceScore = null) {
    if (document.getElementById("email-risk-sidebar")) return;

    const sidebar = document.createElement("div");
    sidebar.id = "email-risk-sidebar";
    sidebar.classList.add(riskLevel); // 'low', 'medium', 'high'

    const title = document.createElement("h3");
    title.innerText = "Email Risk Report";

    const status = document.createElement("p");
    status.innerHTML = `<strong>Status:</strong> <span>${message}</span>`;

    const list = document.createElement("ul");
    for (let item of details) {
        const li = document.createElement("li");
        li.innerText = item;
        list.appendChild(li);
    }

    // ✅ Display actual score (if provided)
    if (confidenceScore !== null) {
        const confidence = document.createElement("p");
        confidence.innerHTML = `<strong>Confidence Score:</strong> ${confidenceScore}%`;
        sidebar.appendChild(confidence);
    }

    const closeBtn = document.createElement("button");
    closeBtn.innerText = "Close";
    closeBtn.onclick = () => {
        sidebar.classList.add("hide");
        setTimeout(() => sidebar.remove(), 300);
    };

    sidebar.appendChild(title);
    sidebar.appendChild(status);
    sidebar.appendChild(list);
    sidebar.appendChild(closeBtn);

    document.body.appendChild(sidebar);
    setTimeout(() => sidebar.classList.add("show"), 50);
}

  

function analyzeEmail() {
    const subjectElement = document.querySelector("h2.hP");
    const bodyElement = document.querySelector("div.a3s.aiL");
    
    const subjectText = subjectElement?.innerText || "⚠️ No subject found";
    const bodyText = bodyElement?.innerText || "⚠️ No body content found";
    
    console.log("📧 Subject:", subjectText);
    console.log("📧 Body:", bodyText);
    
    fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        subject: subjectText,
        body: bodyText
      })
    })    
    .then(res => res.json())
    .then(data => {
      const { label, score } = data;
      let risk = "low";
      let message = "✅ This email looks safe.";
      let reasons = [];
  
      if (label === "phishing") {
        risk = score > 80 ? "high" : "medium";
        message = "❌ This email may be a phishing attempt!";
        reasons.push("Flagged by phishing detection model.");
      } else {
        reasons.push("No major phishing indicators found.");
      }
  
      createSidebar(risk, message, reasons, score);
    })
    .catch(err => {
      console.error("❌ Prediction failed:", err);
      createSidebar("medium", "⚠️ Could not analyze email", ["Model may be offline or unreachable."], null);
    });
  }
  
  // ✅ Only run one version
  setTimeout(analyzeEmail, 3000);
  