document.addEventListener("DOMContentLoaded", function () {
  fetch("http://127.0.0.1:8000/api/project-data/664910/")
    .then((response) => response.json())
    .then((data) => {
      // Массивы для сопоставления ID с названиями
      const themes = {
        1: "Здравоохранение",
        2: "Образование",
        3: "Технологии",
        4: "Окружающая среда",
        5: "Финансы",
        6: "Развлечения",
        7: "Розничная торговля",
        8: "Транспорт",
        9: "Путешествия",
      };
      const categories = {
        1: "Медицина",
        2: "EdTech",
        3: "Искусственный интеллект",
        4: "Недвижимость",
        5: "GreenTech",
        6: "Пищевые технологии",
        7: "TravelTech",
        8: "Биотехнологии",
        9: "EnergyTech",
      };
      const projectName = data.project_name;
      const description = data.description;
      const project_number = data.project_number;
      const themeName = themes[data.user_input.theme_id];
      const categoryName = categories[data.user_input.category_id];

      document.getElementById("project-title").textContent = projectName;
      document.getElementById("main-project-title").textContent = projectName;
      document.getElementById("main-description").textContent = description;
      document.getElementById("project-number").textContent =
        `project #${project_number}`;
      document.getElementById("span-container-1").textContent = themeName;
      document.getElementById("span-container-2").textContent = categoryName;
    })
    .catch((error) => {
      console.error("Error fetching project data:", error);
    });
});
