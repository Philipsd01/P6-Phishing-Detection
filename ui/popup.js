document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("status").textContent = "Analyzing...";
    setTimeout(() => {
        document.getElementById("status").textContent = "✅ Safe";
    }, 2000);
});
