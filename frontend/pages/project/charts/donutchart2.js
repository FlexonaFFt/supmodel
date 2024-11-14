// Получите элемент canvas
const ctx = document.getElementById("donutChart2").getContext("2d");

// Создайте график
const myChart = new Chart(ctx, {
  type: "doughnut",
  data: {
    labels: ["investments", "crowdfunding"],
    datasets: [
      {
        label: "Предсказанные характеристики за 5 промежуток",
        data: [10000, 28900],
        backgroundColor: ["rgba(12, 110, 253, 0.6)", "rgba(12, 110, 253, 0.9)"],
        borderColor: ["rgba(12, 110, 253, 1)", "rgba(12, 110, 253, 1)"],
        borderWidth: 0,
      },
      {
        label: "Предсказанные характеристики за 4 промежуток",
        data: [8000, 43000],
        backgroundColor: ["rgba(12, 110, 253, 0.3)", "rgba(12, 110, 253, 0.1)"],
        borderColor: ["rgba(12, 110, 253, 1)", "rgba(12, 110, 253, 1)"],
        borderWidth: 0,
      },
      {
        label: "Предсказанные характеристики за 3 промежуток",
        data: [7600, 34500],
        backgroundColor: ["rgba(12, 110, 253, 0.7)", "rgba(12, 110, 253, 0.2)"],
        borderColor: ["rgba(12, 110, 253, 1)", "rgba(12, 110, 253, 1)"],
        borderWidth: 0,
      },
      {
        label: "Предсказанные характеристики за 2 промежуток",
        data: [14387, 12345],
        backgroundColor: ["rgba(12, 110, 253, 0.3)", "rgba(12, 110, 253, 0.8)"],
        borderColor: ["rgba(12, 110, 253, 1)", "rgba(12, 110, 253, 1)"],
        borderWidth: 0,
      },
      {
        label: "Предсказанные характеристики за 1 промежуток",
        data: [16770, 28900],
        backgroundColor: ["rgba(12, 110, 253, 0.9)", "rgba(12, 110, 253, 0.4)"],
        borderColor: ["rgba(12, 110, 253, 1)", "rgba(12, 110, 253, 1)"],
        borderWidth: 0,
      },
      {
        label: "Базовые характеристики",
        data: [22000, 43210],
        backgroundColor: ["rgba(12, 110, 253, 0.5)", "rgba(12, 110, 253, 1)"],
        borderColor: ["rgba(12, 110, 253, 1)", "rgba(12, 110, 253, 1)"],
        borderWidth: 0,
      },
    ],
  },
  options: {
    responsive: true,
    plugins: {
      tooltip: {
        enabled: true,
      },
      legend: {
        display: false,
      },
    },
    animation: {
      animateRotate: true,
      animateScale: true,
      duration: 2000,
    },
    cutout: "20%",
  },
});

/*
// Получите элемент canvas
const ctx = document.getElementById("donutChart2").getContext("2d");

// Создайте график
const myChart = new Chart(ctx, {
  type: "doughnut",
  data: {
    labels: ["investments", "crowdfunding"],
    datasets: [
      {
        label: "Предсказанные характеристики за 5 промежуток",
        data: [10000, 28900],
        backgroundColor: ["rgba(189, 151, 243, 0.2)", "rgba(189, 151, 243, 1)"],
        borderColor: ["rgba(12, 110, 253, 1)", "rgba(12, 110, 253, 1)"],
        borderWidth: 0,
      },
      {
        label: "Предсказанные характеристики за 4 промежуток",
        data: [8000, 43000],
        backgroundColor: ["rgba(12, 110, 253, 0.2)", "rgba(12, 110, 253, 1)"],
        borderColor: ["rgba(12, 110, 253, 1)", "rgba(12, 110, 253, 1)"],
        borderWidth: 0,
      },
      {
        label: "Предсказанные характеристики за 3 промежуток",
        data: [7600, 34500],
        backgroundColor: ["rgba(54, 162, 235, 0.2)", "rgba(54, 162, 235, 1)"],
        borderColor: ["rgba(12, 110, 253, 1)", "rgba(12, 110, 253, 1)"],
        borderWidth: 0,
      },
      {
        label: "Предсказанные характеристики за 2 промежуток",
        data: [14387, 12345],
        backgroundColor: ["rgba(255, 99, 132, 0.2)", "rgba(255, 99, 132, 1)"],
        borderColor: ["rgba(12, 110, 253, 1)", "rgba(12, 110, 253, 1)"],
        borderWidth: 0,
      },
      {
        label: "Предсказанные характеристики за 1 промежуток",
        data: [16770, 28900],
        backgroundColor: ["rgba(75, 192, 192, 0.2)", "rgba(75, 192, 192, 1)"],
        borderColor: ["rgba(12, 110, 253, 1)", "rgba(12, 110, 253, 1)"],
        borderWidth: 0,
      },
      {
        label: "Базовые характеристики",
        data: [22000, 43210],
        backgroundColor: ["rgba(12, 110, 253, 0.2)", "rgba(12, 110, 253, 1)"],
        borderColor: ["rgba(12, 110, 253, 1)", "rgba(12, 110, 253, 1)"],
        borderWidth: 0,
      },
    ],
  },
  options: {
    responsive: true,
    plugins: {
      tooltip: {
        enabled: true,
      },
      legend: {
        display: false,
      },
    },
    animation: {
      animateRotate: true,
      animateScale: true,
      duration: 2000,
    },
    cutout: "20%",
  },
  }); */
