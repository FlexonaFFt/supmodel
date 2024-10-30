// Example starter JavaScript for disabling form submissions if there are invalid fields
(() => {
  "use strict";

  // Преобразуем данные формы в объект
  const getFormData = (form) => {
    const formData = new FormData(form);
    const data = {};
    formData.forEach((value, key) => {
      data[key] = value;
    });
    return data;
  };

  // Функции преобразования для индексов
  const calculateTeamIdx = (teamDesc, experienceYears, teamSize) => {
    const teamMapping = { новички: 2, "средний опыт": 5, эксперты: 8 };
    const baseScore = teamMapping[teamDesc];
    return (
      (0.6 * experienceYears + 0.4 * teamSize) / 10 +
      baseScore / 10
    ).toFixed(1);
  };

  const calculateTechIdx = (techLevel, techInvestment) => {
    const techMapping = { низкий: 2, средний: 5, высокий: 8 };
    const baseScore = techMapping[techLevel];
    return (((0.5 * techInvestment) / 1000000 + 0.5 * baseScore) / 10).toFixed(
      1,
    );
  };

  const calculateCompIdx = (compLevel, competitors) => {
    const compMapping = {
      "низкая конкуренция": 8,
      "средняя конкуренция": 5,
      "высокая конкуренция": 2,
    };
    const baseScore = compMapping[compLevel];
    return (baseScore - competitors / 100).toFixed(1);
  };

  const calculateSocialIdx = (socialImpactDesc) => {
    const socialMapping = {
      "низкое влияние": 3,
      "среднее влияние": 6,
      "высокое влияние": 9,
    };
    return socialMapping[socialImpactDesc];
  };

  const calculateDemandIdx = (demandLevel, audienceReach, marketSize) => {
    const demandMapping = {
      "низкий спрос": 2,
      "средний спрос": 5,
      "высокий спрос": 8,
    };
    const baseScore = demandMapping[demandLevel];
    return (
      (baseScore + (audienceReach + marketSize) / (1000000 + 100000000)) *
      10
    ).toFixed(1);
  };

  // Подготовка данных для API
  const prepareDataForAPI = (formData) => {
    // Проверка, что значения существуют и не равны NaN
    const category_id = formData["category_id"] || 0;
    const comp_idx = calculateCompIdx(
      formData["comp_level"],
      parseFloat(formData["competitors"]) || 0,
    );
    const crowdfunding_m = parseFloat(formData["crowdfunding_m"]) || 0;
    const demand_idx = calculateDemandIdx(
      formData["demand_level"],
      parseFloat(formData["audience_reach"]) || 0,
      parseFloat(formData["market_size"]) || 0,
    );
    const investments_m = parseFloat(formData["investments_m"]) || 0;
    const social_idx = calculateSocialIdx(formData["social_impact_desc"]) || 0;
    const start_m = parseFloat(formData["start_m"]) || 0;
    const team_idx = calculateTeamIdx(
      formData["team_desc"],
      parseFloat(formData["experience_years"]) || 0,
      parseFloat(formData["team_size"]) || 0,
    );
    const tech_idx = calculateTechIdx(
      formData["tech_level"],
      parseFloat(formData["tech_investment"]) || 0,
    );
    const theme_id = formData["theme_id"] || 0;

    return {
      theme_id,
      category_id,
      comp_idx,
      start_m,
      investments_m,
      crowdfunding_m,
      team_idx,
      tech_idx,
      social_idx,
      demand_idx,
    };
  };

  // Функция отправки данных на API
  const sendDataToAPI = async (data) => {
    try {
      const response = await fetch("http://127.0.0.1:8000/predict/lstm3", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });
      const result = await response.json();
      console.log("Response from API:", result);
      alert("Данные успешно отправлены!");

      // Работа с ответом API
      const responseArray = [
        result.social_idx,
        result.investments_m,
        result.crowdfunding_m,
        result.demand_idx,
        result.comp_idx,
      ];
      console.log("Отформатированный ответ:", responseArray);
    } catch (error) {
      console.error("Ошибка отправки данных:", error);
      alert("Не удалось отправить данные.");
    }
  };

  // Основная функция для валидации формы и отправки данных
  const forms = document.querySelectorAll(".needs-validation");
  Array.from(forms).forEach((form) => {
    form.addEventListener("submit", async (event) => {
      if (!form.checkValidity()) {
        event.preventDefault();
        event.stopPropagation();
      } else {
        event.preventDefault();
        const formData = getFormData(form);
        const apiData = prepareDataForAPI(formData);
        console.log("Отправляемые данные:", apiData); // Проверка данных перед отправкой
        await sendDataToAPI(apiData);
      }
      form.classList.add("was-validated");
    });
  });
})();
