const values = [8.6, 8.3, 8.1, 8.2, 8.8];
const options = [
  {
    range: [1, 2],
    text: "Проект демонстрирует очень низкие показатели по основным параметрам. Требуется значительная доработка, чтобы повысить шансы на успех.",
  },
  {
    range: [2.1, 4],
    text: "Характеристики проекта находятся на уровне ниже среднего. Хотя есть потенциал, он ограничен, и потребуется много усилий для улучшения.",
  },
  {
    range: [4.1, 7],
    text: "Проект имеет средние показатели. Это неплохая отправная точка, но для достижения высоких результатов нужно продолжать работать над ключевыми аспектами.",
  },
  {
    range: [7.1, 10],
    text: "Проект демонстрирует отличные показатели! Это сильная позиция, которая обещает высокие шансы на успех и дальнейшее развитие.",
  },
];

function getTextForAverage(value) {
  const option = options.find(
    ({ range }) => value >= range[0] && value <= range[1],
  );
  return option ? option.text : "Текст не найден.";
}

const average = values.reduce((sum, value) => sum + value, 0) / values.length;
const conclusionText = getTextForAverage(average);
document.querySelector("#dynamic-conclusion p").textContent = conclusionText;
