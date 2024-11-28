document.addEventListener("DOMContentLoaded", function () {
  const itemsPerPage = 9;
  const projects = [
    {
      title: "Lil nas X project",
      author: "Lil nas X",
      description:
        "Тут идет описание проекта, его краткое description. Нужно сделать ограничение...",
      theme: "Тема стартапа",
      category: "Категория",
    },
    {
      title: "Lil nas X project",
      author: "Lil nas X",
      description:
        "Тут идет описание проекта, его краткое description. Нужно сделать ограничение...",
      theme: "Тема стартапа",
      category: "Категория",
    },
    {
      title: "Lil nas X project",
      author: "Lil nas X",
      description:
        "Тут идет описание проекта, его краткое description. Нужно сделать ограничение...",
      theme: "Тема стартапа",
      category: "Категория",
    },
    {
      title: "swag",
      author: "Lil nas X",
      description:
        "Тут идет описание проекта, его краткое description. Нужно сделать ограничение...",
      theme: "Тема стартапа",
      category: "Категория",
    },
    {
      title: "Lil nas X project",
      author: "Lil nas X",
      description:
        "Тут идет описание проекта, его краткое description. Нужно сделать ограничение...",
      theme: "Тема стартапа",
      category: "Категория",
    },
    {
      title: "Lil nas X project",
      author: "Lil nas X",
      description:
        "Тут идет описание проекта, его краткое description. Нужно сделать ограничение...",
      theme: "Тема стартапа",
      category: "Категория",
    },
    {
      title: "Lil nas X project",
      author: "Lil nas X",
      description:
        "Тут идет описание проекта, его краткое description. Нужно сделать ограничение...",
      theme: "Тема стартапа",
      category: "Категория",
    },
    {
      title: "Lil nas X project",
      author: "Lil nas X",
      description:
        "Тут идет описание проекта, его краткое description. Нужно сделать ограничение...",
      theme: "Тема стартапа",
      category: "Категория",
    },
    {
      title: "Lil nas X project",
      author: "Lil nas X",
      description:
        "Тут идет описание проекта, его краткое description. Нужно сделать ограничение...",
      theme: "Тема стартапа",
      category: "Категория",
    },
    {
      title: "Lil nas X project",
      author: "Lil nas X",
      description:
        "Тут идет описание проекта, его краткое description. Нужно сделать ограничение...",
      theme: "Тема стартапа",
      category: "Категория",
    },
    {
      title: "Lil nas X project",
      author: "Lil nas X",
      description:
        "Тут идет описание проекта, его краткое description. Нужно сделать ограничение...",
      theme: "Тема стартапа",
      category: "Категория",
    },
    {
      title: "Lil nas X project",
      author: "Lil nas X",
      description:
        "Тут идет описание проекта, его краткое description. Нужно сделать ограничение...",
      theme: "Тема стартапа",
      category: "Категория",
    },
    {
      title: "Lil nas X project",
      author: "Lil nas X",
      description:
        "Тут идет описание проекта, его краткое description. Нужно сделать ограничение...",
      theme: "Тема стартапа",
      category: "Категория",
    },
    {
      title: "Lil nas X project",
      author: "Lil nas X",
      description:
        "Тут идет описание проекта, его краткое description. Нужно сделать ограничение...",
      theme: "Тема стартапа",
      category: "Категория",
    },
    {
      title: "Lil nas X project",
      author: "Lil nas X",
      description:
        "Тут идет описание проекта, его краткое description. Нужно сделать ограничение...",
      theme: "Тема стартапа",
      category: "Категория",
    },
    {
      title: "Lil nas X project",
      author: "Lil nas X",
      description:
        "Тут идет описание проекта, его краткое description. Нужно сделать ограничение...",
      theme: "Тема стартапа",
      category: "Категория",
    },
    {
      title: "Lil nas X project",
      author: "Lil nas X",
      description:
        "Тут идет описание проекта, его краткое description. Нужно сделать ограничение...",
      theme: "Тема стартапа",
      category: "Категория",
    },
    {
      title: "fuck be ti",
      author: "Lil nas X",
      description:
        "Тут идет описание проекта, его краткое description. Нужно сделать ограничение...",
      theme: "Тема стартапа",
      category: "Категория",
    },
  ];

  function renderProjects(page) {
    const projectList = document.getElementById("project-list");
    projectList.innerHTML = "";
    const start = (page - 1) * itemsPerPage;
    const end = start + itemsPerPage;
    const paginatedProjects = projects.slice(start, end);

    paginatedProjects.forEach((project) => {
      const projectElement = document.createElement("div");
      projectElement.className = "col-12 col-md-6 col-lg-4 mb-3";
      projectElement.innerHTML = `
                <div class="card border h-100">
                    <div class="card-header">${project.title}</div>
                    <div class="card-body text-secondary">
                        <h6 class="card-subtitle text-muted mb-2">${project.author}</h6>
                        <p class="card-text">${project.description}</p>
                        <div class="text-muted">
                            <h6>${project.theme}</h6>
                            <h6>${project.category}</h6>
                        </div>
                    </div>
                </div>
            `;
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
          renderProjects(i);
          renderPagination(totalPages, i);
        });
      }
      pagination.appendChild(pageItem);
    }
  }

  const totalPages = Math.ceil(projects.length / itemsPerPage);
  renderProjects(1);
  renderPagination(totalPages, 1);
});
