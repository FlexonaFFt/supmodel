document.addEventListener("DOMContentLoaded", function () {
  const baseBlocks = document.querySelectorAll(".base-predictions");
  const timeSeriesBlocks = document.querySelectorAll(
    ".time-series-predictions",
  );
  const syntheticBlocks = document.querySelectorAll(".synthetic-prediction");
  const radioButtons = document.querySelectorAll(
    'input[name="listGroupCheckableRadios"]',
  );

  // Функция для скрытия всех блоков
  function hideAllBlocks() {
    baseBlocks.forEach((block) => (block.style.display = "none"));
    timeSeriesBlocks.forEach((block) => (block.style.display = "none"));
    syntheticBlocks.forEach((block) => (block.style.display = "none"));
  }

  radioButtons.forEach((radio) => {
    radio.addEventListener("change", function () {
      hideAllBlocks(); // Сначала скрываем все блоки

      // Затем показываем блоки, соответствующие выбранной радио-кнопке, сбрасывая `display`
      if (this.id === "listGroupCheckableRadios1") {
        baseBlocks.forEach((block) => (block.style.display = ""));
      } else if (this.id === "listGroupCheckableRadios2") {
        timeSeriesBlocks.forEach((block) => (block.style.display = ""));
      } else if (this.id === "listGroupCheckableRadios3") {
        syntheticBlocks.forEach((block) => (block.style.display = ""));
      }
    });
  });

  // Инициализация: скрываем все блоки, кроме базовых
  hideAllBlocks();
  baseBlocks.forEach((block) => (block.style.display = ""));
});
