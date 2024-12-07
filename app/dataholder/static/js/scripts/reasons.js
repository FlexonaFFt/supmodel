const options = {
  0: [
    {
      range: [1, 2],
      text: "Начальный спрос находится на низком уровне. Вероятность дальнейшего роста не велика.",
      level: "low",
    },
    {
      range: [2.1, 3.4],
      text: "Начальный спрос находится на низком уровне. Но, возможно, спрос вырастет с течением времени.",
      level: "low",
    },
    {
      range: [3.5, 6],
      text: "Спрос находится на среднем уровне. Вероятность дальнейшего роста велика.",
      level: "mid",
    },
    {
      range: [6.1, 8],
      text: "Спрос находится на высоком уровне. С большей вероятностью спрос может продолжить расти.",
      level: "max",
    },
    {
      range: [8.1, 10],
      text: "Спрос находится на высоком уровне. Вероятнее всего спрос немного упадет.",
      level: "max",
    },
  ],
  1: [
    {
      range: [1, 4],
      text: "Стартап может испытывать трудности из-за конкуренции. Стоит обратить на это внимание.",
      level: "low",
    },
    {
      range: [4.1, 7],
      text: "Стартап имеет умеренную конкурентную среду. В такой ситуации он сможет уверенно существовать.",
      level: "mid",
    },
    {
      range: [7.1, 10],
      text: "Стартап находится в отличной конкурентной позиции. Это отличный показатель!",
      level: "max",
    },
  ],
  2: [
    {
      range: [1, 3],
      text: "Команда нуждается в срочном улучшении, иначе стартап может провалиться.",
      level: "low",
    },
    {
      range: [3.1, 6],
      text: "Команда работает на среднем уровне. Это может быть для проекта как хорошо, так и плохо.",
      level: "mid",
    },
    {
      range: [6.1, 10],
      text: "Команда демонстрирует высокий профессионализм, что очень хорошо для стартапа.",
      level: "max",
    },
  ],
  3: [
    {
      range: [1, 5],
      text: "Необходимы значительные технологические инвестиции.",
      level: "low",
    },
    {
      range: [5.1, 8],
      text: "Инвестиции в технологии достаточно стабильны.",
      level: "mid",
    },
    {
      range: [8.1, 10],
      text: "Технологические инвестиции превосходят ожидания.",
      level: "max",
    },
  ],
  4: [
    {
      range: [1, 3],
      text: "Общество может отреагировать нейтрально. Нужно обратить на это внимание!",
      level: "low",
    },
    {
      range: [3.1, 6],
      text: "Общество проявляет немалый интерес к стартапу, что может хорошо повлиять на проект.",
      level: "mid",
    },
    {
      range: [6.1, 10],
      text: "Общество активно проявляет интерес к стартапу! Успех обеспечен!",
      level: "max",
    },
  ],
};

function getTextAndLevelForValue(index, value) {
  // Корректировка значения, если оно выходит за пределы допустимого диапазона
  value = Math.max(1.0, Math.min(9.9, value));

  const option = options[index].find(
    ({ range }) => value >= range[0] && value <= range[1],
  );
  return option
    ? { text: option.text, level: option.level }
    : { text: "Текст не найден.", level: "unknown" };
}

document.addEventListener("DOMContentLoaded", function () {
  const projectNumber = document
    .getElementById("myChart")
    .getAttribute("data-project-number");

  fetch(`/api/project-data/${projectNumber}/`)
    .then((response) => response.json())
    .then((data) => {
      // Получаем первый элемент массива indeces
      const index = data.indeces[0];

      const values = [
        index.demand_idx,
        index.competition_idx,
        index.team_idx,
        index.tech_idx,
        index.social_idx,
      ];

      document.querySelectorAll("#dynamic-texts a").forEach((item, index) => {
        const value = values[index];
        const { text, level } = getTextAndLevelForValue(index, value);

        // Обновляем текст в <p>
        const p = item.querySelector("p");
        if (p) p.textContent = text;

        // Обновляем уровень в <small>
        const small = item.querySelector("small");
        if (small) small.textContent = level;
      });
    })
    .catch((error) => console.error("Error fetching data:", error));
});
