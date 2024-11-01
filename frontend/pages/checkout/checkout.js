(() => {
  "use strict";

  // Получаем все формы, к которым необходимо применить кастомные стили валидации Bootstrap
  const forms = document.querySelectorAll(".needs-validation");

  // Проходим по всем формам и добавляем событие отправки
  Array.from(forms).forEach((form) => {
    form.addEventListener(
      "submit",
      async (event) => {
        // Если форма не валидна, предотвращаем отправку
        if (!form.checkValidity()) {
          event.preventDefault();
          event.stopPropagation();
        } else {
          // Если форма валидна, выполняем отправку через API
          event.preventDefault(); // Отменяем стандартное поведение отправки

          // Собираем данные формы
          const formData = {
            startup_name: document.getElementById("StartupName").value.trim(),
            team_name: document.getElementById("TeamName").value.trim(),
            theme_id: parseInt(document.getElementById("StartupTheme").value), // Преобразуем в число
            category_id: parseInt(
              document.getElementById("StartupCathegory").value,
            ), // Преобразуем в число
            description: document.getElementById("Description").value.trim(),
            start_m: parseFloat(document.getElementById("Start_m").value),
            investments_m: parseFloat(
              document.getElementById("Investments_m").value,
            ),
            crowdfunding_m: parseFloat(
              document.getElementById("Crowdfunding_m").value,
            ),
            team_size: parseInt(document.getElementById("TeamCount").value),
            team_index: parseFloat(document.getElementById("TeamIndex").value),
            team_mapping: document.getElementById("TeamExp").value,
            tech_level: document.getElementById("Techonological_kef").value,
            tech_investment: parseFloat(
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
              "http://127.0.0.1:8000/predict/full_form",
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
            }

            const result = await response.json();
            console.log("Prediction:", result.prediction);
            console.log("calculate_indices:", result.calculate_indices);
            alert(
              `Результат предсказания: ${JSON.stringify(result.prediction)}`,
            );
          } catch (error) {
            console.error("Ошибка при отправке запроса:", error);
            alert(
              "Не удалось выполнить предсказание. Проверьте консоль для подробностей.",
            );
          }
        }

        form.classList.add("was-validated"); // Добавляем Bootstrap-стиль валидации
      },
      false,
    );
  });
})();
