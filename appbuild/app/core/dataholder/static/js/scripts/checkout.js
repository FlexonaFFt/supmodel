(() => {
  "use strict";
  const forms = document.querySelectorAll(".needs-validation");
  Array.from(forms).forEach((form) => {
    form.addEventListener(
      "submit",
      async (event) => {
        if (!form.checkValidity()) {
          event.preventDefault();
          event.stopPropagation();
        } else {
          event.preventDefault();

          const formData = {
            startup_name: document.getElementById("StartupName").value.trim(),
            team_name: document.getElementById("TeamName").value.trim(),
            theme_id: parseInt(document.getElementById("StartupTheme").value),
            category_id: parseInt(
              document.getElementById("StartupCathegory").value,
            ),
            description: document.getElementById("Description").value.trim(),
            start_m: parseInt(document.getElementById("Start_m").value),
            investments_m: parseInt(
              document.getElementById("Investments_m").value,
            ),
            crowdfunding_m: parseFloat(
              document.getElementById("Crowdfunding_m").value,
            ),
            team_size: parseInt(document.getElementById("TeamCount").value),
            team_index: parseInt(document.getElementById("TeamIndex").value),
            team_mapping: document.getElementById("TeamExp").value,
            tech_level: document.getElementById("Techonological_kef").value,
            tech_investment: parseInt(
              document.getElementById("Techological_m").value,
            ),
            competition_level: document.getElementById("Competition").value,
            competitor_count: parseInt(
              document.getElementById("CompetititorCount").value,
            ),
            demand_level: document.getElementById("DemandLevel").value,
            social_impact: document.getElementById("DemandIdx").value,
            audience_reach: parseInt(
              document.getElementById("TargetAudience").value,
            ),
            market_size: parseInt(document.getElementById("MarketSize").value),
          };

          console.log("Отправляемые данные:", formData);
          console.log(
            "Отправляемые данные:",
            JSON.stringify(formData, null, 2),
          );

          try {
            // Отправляем данные на API
            const response = await fetch(
              "http://127.0.0.1:8001/predict/all_full_form",
              {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                },
                body: JSON.stringify(formData),
              },
            );

            if (!response.ok) {
              throw new Error(`Ошибка: ${response.statusText}`);
              if (result.project_url) {
                window.location.href = result.project_url;
              } else {
                console.error("URL проекта не найден в ответе API.");
                alert("URL проекта не найден в ответе API.");
              }
            }

            const result = await response.json();
          } catch (error) {
            console.error("Ошибка при отправке запроса:", error);
          }
        }

        form.classList.add("was-validated");
      },
      false,
    );
  });
})();
