function createSidebar(riskLevel, message, details = []) {
    if (document.getElementById("email-risk-sidebar")) return;
  
    const sidebar = document.createElement("div");
    sidebar.id = "email-risk-sidebar";
    sidebar.classList.add(riskLevel); // Adds 'high', 'medium', or 'low' class for styling
  
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
  
    const closeBtn = document.createElement("button");
    closeBtn.innerText = "Close";
    closeBtn.onclick = () => sidebar.remove();
  
    sidebar.appendChild(title);
    sidebar.appendChild(status);
    sidebar.appendChild(list);
    sidebar.appendChild(closeBtn);
  
    document.body.appendChild(sidebar);
  }
  

// Simulated email analysis
function analyzeEmail() {
    const emailText = document.body.innerText;

    let risk = "low";
    let message = "✅ This email looks safe.";
    let reasons = [];

    if (emailText.includes("urgent") || emailText.includes("click here")) {
        risk = "high";
        message = "❌ This email may be a phishing attempt!";
        reasons.push("Contains words like 'urgent' or 'click here'");
    } else if (emailText.includes("unsubscribe") || emailText.includes("limited offer")) {
        risk = "medium";
        message = "⚠️ This email might be promotional or spam.";
        reasons.push("Contains marketing keywords");
    } else {
        reasons.push("No major phishing indicators found");
    }

    createSidebar(risk, message, reasons);
}

setTimeout(analyzeEmail, 3000);  // Wait for Gmail to load
