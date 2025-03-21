function addWarningBanner(riskLevel, message) {
    const emailContainer = document.querySelector("div.adn.ads"); // Finds email body in Gmail
    if (!emailContainer) return;

    const existingBanner = document.getElementById("email-warning-banner");
    if (existingBanner) return; // Avoid duplicate banners

    const banner = document.createElement("div");
    banner.id = "email-warning-banner";
    banner.textContent = message;
    banner.style.position = "relative";
    banner.style.padding = "10px";
    banner.style.borderRadius = "5px";
    banner.style.marginBottom = "10px";
    banner.style.color = "#fff";
    banner.style.fontWeight = "bold";
    banner.style.textAlign = "center";

    if (riskLevel === "high") {
        banner.style.backgroundColor = "red";
    } else if (riskLevel === "medium") {
        banner.style.backgroundColor = "orange";
    } else {
        banner.style.backgroundColor = "green";
    }

    emailContainer.insertBefore(banner, emailContainer.firstChild);
}

// Simulate email analysis (in a real app, call your backend)
function analyzeEmail() {
    const emailBody = document.body.innerText;
    
    if (emailBody.includes("urgent") || emailBody.includes("click here")) {
        addWarningBanner("high", "⚠️ This email may be phishing! Be careful.");
    } else if (emailBody.includes("unsubscribe") || emailBody.includes("limited offer")) {
        addWarningBanner("medium", "⚠️ This email might be spam.");
    } else {
        addWarningBanner("low", "✅ This email looks safe.");
    }
}

// Wait for Gmail to load and analyze the email
setTimeout(analyzeEmail, 3000);
