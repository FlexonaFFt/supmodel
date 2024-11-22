document.addEventListener("DOMContentLoaded", function () {
  fetch("http://127.0.0.1:8000/api/project-data/637206/")
    .then((response) => response.json())
    .then((data) => {
      const projectName = data.project_name;
      const description = data.description;
      const project_number = data.project_number;

      document.getElementById("project-title").textContent = projectName;
      document.getElementById("main-project-title").textContent = projectName;
      document.getElementById("main-description").textContent = description;
      document.getElementById("project-number").textContent =
        `project #${project_number}`;
    })
    .catch((error) => {
      console.error("Error fetching project data:", error);
      document.getElementById("project-title").textContent =
        "Error loading project data";
      document.getElementById("main-project-title").textContent =
        "Error loading project data";
      document.getElementById("main-description").textContent =
        "Error loading project description";
      document.getElementById("project-number").textContent = "project #Error";
    });
});
