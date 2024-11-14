document.addEventListener("DOMContentLoaded", function () {
  const baseBlocks = document.querySelectorAll(".base-predictions");
  const timeSeriesBlocks = document.querySelectorAll(
    ".time-series-predictions",
  );
  const radioButtons = document.querySelectorAll(
    'input[name="listGroupCheckableRadios"]',
  );

  radioButtons.forEach((radio) => {
    radio.addEventListener("change", function () {
      if (this.id === "listGroupCheckableRadios1") {
        baseBlocks.forEach((block) => (block.style.display = "block"));
        timeSeriesBlocks.forEach((block) => (block.style.display = "none"));
      } else if (this.id === "listGroupCheckableRadios2") {
        baseBlocks.forEach((block) => (block.style.display = "none"));
        timeSeriesBlocks.forEach((block) => (block.style.display = "block"));
      }
    });
  });
});
