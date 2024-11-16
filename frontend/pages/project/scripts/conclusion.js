const values = [5.6];
const options = {
  0: [
    { range: [1, 3], text: "Спрос находится на низком уровне." },
    {
      range: [3.1, 6],
      text: "Короче, воалыоаоыво аыва ыв аывлоаылвоа лоывало ывлоаыов лаоылво алывоалыво алывоалывоа ылвоаылв оаылаолы ",
    },
    { range: [6.1, 10], text: "Спрос находится на высоком уровне." },
  ],
};
function getTextForValue(index, value) {
  const option = options[index].find(
    ({ range }) => value >= range[0] && value <= range[1],
  );
  return option ? option.text : "Текст не найден.";
}
document.querySelectorAll("#dynamic-conclusion p").forEach((p, index) => {
  const value = values[index];
  p.textContent = getTextForValue(index, value);
});
