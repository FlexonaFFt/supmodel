document.addEventListener("DOMContentLoaded", function () {
  const itemsPerPage = 9;
  const apiUrl = "http://127.0.0.1:8000/api/projects/";
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

  async function fetchProjects(page) {
    try {
      const response = await fetch(
        `${apiUrl}?page=${page}&limit=${itemsPerPage}`,
      );
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      console.log("Fetched projects:", data);
      return data;
    } catch (error) {
      console.error("Error fetching projects:", error);
      return [];
    }
  }

  async function fetchProjectData(projectNumber) {
    try {
      const response = await fetch(
        `http://127.0.0.1:8000/api/project-data/${projectNumber}/`,
      );
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      console.log(`Fetched project data for project ${projectNumber}:`, data); // Логирование данных для отладки
      return data;
    } catch (error) {
      console.error(
        `Error fetching project data for project ${projectNumber}:`,
        error,
      );
      return {};
    }
  }

  async function loadProjects(page, searchQuery = "") {
    const data = await fetchProjects(page);
    if (!Array.isArray(data)) {
      console.error("Projects data is missing or invalid:", data);
      return;
    }

    const projectsWithDetails = await Promise.all(
      data.map(async (project) => {
        const projectData = await fetchProjectData(project.project_number);
        return {
          ...project,
          team_name: projectData.user_input?.team_name || "N/A",
          theme: themes[projectData.user_input?.theme_id] || "N/A",
          category: categories[projectData.user_input?.category_id] || "N/A",
        };
      }),
    );

    const filteredProjects = projectsWithDetails.filter(
      (project) =>
        project.project_name
          .toLowerCase()
          .includes(searchQuery.toLowerCase()) ||
        project.team_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        project.description.toLowerCase().includes(searchQuery.toLowerCase()),
    );

    renderProjects(filteredProjects);
    renderPagination(Math.ceil(filteredProjects.length / itemsPerPage), page);
  }

  function truncateText(text, maxLength) {
    if (text.length <= maxLength) {
      return text;
    }
    return text.substring(0, maxLength) + "...";
  }

  function renderProjects(projects) {
    const projectList = document.getElementById("project-list");
    projectList.innerHTML = "";

    projects.forEach((project) => {
      const projectElement = document.createElement("div");
      projectElement.setAttribute("data-project-id", project.id);
      projectElement.className = "col-12 col-md-6 col-lg-4 mb-3 project-card";
      projectElement.innerHTML = `
                <div class="card border h-100">
                    <div class="card-header">${project.project_name}</div>
                    <div class="card-body text-secondary">
                        <h6 class="card-subtitle text-muted mb-2">${project.team_name}</h6>
                        <p class="card-text">${truncateText(project.description, 90)}</p>
                        <div class="text-muted">
                            <h6>Theme ID: ${project.theme}</h6>
                            <h6>Category ID: ${project.category}</h6>
                        </div>
                    </div>
                </div>
            `;
      projectElement.addEventListener("click", () => {
        window.location.href = `/project/${project.project_number}`;
      });
      projectList.appendChild(projectElement);
    });
  }

  function renderPagination(totalPages, currentPage) {
    const pagination = document.getElementById("pagination");
    pagination.innerHTML = "";

    for (let i = 1; i <= totalPages; i++) {
      const pageItem = document.createElement("li");
      pageItem.className = "page-item";
      if (i === currentPage) {
        pageItem.classList.add("active");
        pageItem.innerHTML = `<span class="page-link">${i}</span>`;
      } else {
        pageItem.innerHTML = `<a class="page-link" href="#">${i}</a>`;
        pageItem.querySelector("a").addEventListener("click", () => {
          loadProjects(i, document.getElementById("searchInput").value);
        });
      }
      pagination.appendChild(pageItem);
    }
  }

  // Initial load
  loadProjects(1);

  // Search input event listener
  document.getElementById("searchInput").addEventListener("input", function () {
    const searchQuery = this.value;
    loadProjects(1, searchQuery);
  });

  // Form submit event listener
  document.querySelector("form").addEventListener("submit", function (event) {
    event.preventDefault();
    const searchQuery = document.getElementById("searchInput").value;
    loadProjects(1, searchQuery);
  });
});
