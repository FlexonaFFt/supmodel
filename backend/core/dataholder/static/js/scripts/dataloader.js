document.addEventListener("DOMContentLoaded", function () {
  fetch("http://127.0.0.1:8000/api/project-data/637206/")
    .then((response) => response.json())
    .then((data) => {
      // Извлекаем project_name из данных
      const projectName = data.project_name;
      // Обновляем title страницы
      document.getElementById("project-title").textContent = projectName;
    })
    .catch((error) => {
      console.error("Error fetching project data:", error);
      document.getElementById("project-title").textContent =
        "Error loading project data";
    });
});
