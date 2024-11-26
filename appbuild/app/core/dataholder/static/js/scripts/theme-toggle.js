const root = document.documentElement;
const themeToggle = document.getElementById("bd-theme");
const themeButtons = document.querySelectorAll("[data-bs-theme-value]");
const themeIconActive = document.querySelector(".theme-icon-active use");

function setTheme(theme) {
  root.setAttribute("data-bs-theme", theme);

  themeIconActive.setAttribute(
    "href",
    {
      light: "#sun-fill",
      dark: "#moon-stars-fill",
      auto: "#circle-half",
    }[theme],
  );

  themeButtons.forEach((button) => {
    const isActive = button.getAttribute("data-bs-theme-value") === theme;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-pressed", isActive);
    button
      .querySelector("svg:last-child")
      .classList.toggle("d-none", !isActive);
  });

  if (theme === "auto") {
    setSystemTheme();
  }
}

function setSystemTheme() {
  const isDarkMode = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const systemTheme = isDarkMode ? "dark" : "light";
  root.setAttribute("data-bs-theme", systemTheme);
  themeIconActive.setAttribute(
    "href",
    isDarkMode ? "#circle-half" : "#circle-half",
  );

  themeButtons.forEach((button) => {
    const isActive = button.getAttribute("data-bs-theme-value") === "auto";
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-pressed", isActive);
  });
}

const defaultTheme = "auto";
if (defaultTheme === "auto") {
  setSystemTheme();
} else {
  setTheme(defaultTheme);
}

window
  .matchMedia("(prefers-color-scheme: dark)")
  .addEventListener("change", () => {
    if (root.getAttribute("data-bs-theme") === "auto") {
      setSystemTheme();
    }
  });

themeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const selectedTheme = button.getAttribute("data-bs-theme-value");
    setTheme(selectedTheme);
  });
});
