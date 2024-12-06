document
  .getElementById("checkoutButton")
  .addEventListener("click", function () {
    window.location.href = "http://localhost:8000/checkout";
  });

document.getElementById("mainButton").addEventListener("click", function () {
  window.location.href = "http://localhost:8000/";
});
