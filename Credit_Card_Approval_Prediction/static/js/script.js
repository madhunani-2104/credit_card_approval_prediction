document.querySelector("form")?.addEventListener("submit", () => {
  const btn = document.querySelector("button");
  if (btn) { btn.textContent = "Predicting..."; btn.disabled = true; }
});
