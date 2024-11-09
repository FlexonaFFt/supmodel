const themeToggle = document.getElementById("theme-toggle");
const root = document.documentElement;

// Проверка системной темы и установка начальной темы
function setSystemTheme() {
  const isDarkMode = window.matchMedia("(prefers-color-scheme: dark)").matches;
  root.setAttribute("data-bs-theme", isDarkMode ? "dark" : "light");
  themeToggle.textContent = isDarkMode ? "Светлая тема" : "Темная тема";
}

// Установка начальной темы при загрузке
setSystemTheme();

// Слушаем изменения системной темы
window
  .matchMedia("(prefers-color-scheme: dark)")
  .addEventListener("change", setSystemTheme);

// Обработчик для переключения темы вручную
themeToggle.addEventListener("click", () => {
  const currentTheme = root.getAttribute("data-bs-theme");
  const newTheme = currentTheme === "dark" ? "light" : "dark";
  root.setAttribute("data-bs-theme", newTheme);
  themeToggle.textContent =
    newTheme === "dark" ? "Светлая тема" : "Темная тема";
});
