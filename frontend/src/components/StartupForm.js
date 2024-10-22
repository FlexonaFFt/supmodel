// src/components/StartupForm.js
import React, { useState } from "react";
import "../styles/StartupForm.css";

const StartupForm = () => {
  const [formData, setFormData] = useState({
    name: "",
    description: "",
    theme: "",
    category: "",
    capital: "",
    investments: "",
    crowdfunding: "",
    teamSize: "",
    teamExperience: "",
    fieldExperience: "",
    techLevel: "",
    techInvestments: "",
    competitionLevel: "",
    competitorsCount: "",
    demandLevel: "",
    targetAudience: "",
    marketSize: "",
  });

  const themes = [
    { id: 1, name: "Healthcare" },
    { id: 2, name: "Education" },
    { id: 3, name: "Technology" },
    { id: 4, name: "Environment" },
    { id: 5, name: "Finance" },
    { id: 6, name: "Entertainment" },
    { id: 7, name: "Retail" },
    { id: 8, name: "Transportation" },
    { id: 9, name: "Travel" },
  ];

  const categories = [
    { id: 1, name: "Medical" },
    { id: 2, name: "EdTech" },
    { id: 3, name: "AI" },
    { id: 4, name: "GreenTech" },
    { id: 5, name: "Property" },
    { id: 6, name: "FoodTech" },
    { id: 7, name: "TravelTech" },
    { id: 8, name: "SpaceTech" },
    { id: 9, name: "Biotech" },
    { id: 10, name: "EnergyTech" },
  ];

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log(formData);
  };

  return (
    <form onSubmit={handleSubmit}>
      <h2>Startup Information</h2>

      <div>
        <label>Название стартапа:</label>
        <input
          type="text"
          name="name"
          value={formData.name}
          onChange={handleChange}
          required
        />
      </div>

      <div>
        <label>Краткое описание:</label>
        <textarea
          name="description"
          value={formData.description}
          onChange={handleChange}
          maxLength="250"
          required
        />
      </div>

      <div>
        <label>Тема стартапа:</label>
        <select
          name="theme"
          value={formData.theme}
          onChange={handleChange}
          required
        >
          <option value="">Выбрать тему</option>
          {themes.map((theme) => (
            <option key={theme.id} value={theme.id}>
              {theme.name}
            </option>
          ))}
        </select>
      </div>

      <div>
        <label>Категория стартапа:</label>
        <select
          name="category"
          value={formData.category}
          onChange={handleChange}
          required
        >
          <option value="">Выбрать категорию</option>
          {categories.map((category) => (
            <option key={category.id} value={category.id}>
              {category.name}
            </option>
          ))}
        </select>
      </div>

      <div>
        <label>Стартовый капитал (USD):</label>
        <input
          type="number"
          name="capital"
          value={formData.capital}
          onChange={handleChange}
          required
        />
      </div>

      <div>
        <label>Инвестиции (USD):</label>
        <input
          type="number"
          name="investments"
          value={formData.investments}
          onChange={handleChange}
          required
        />
      </div>

      <div>
        <label>Краудфандинговые сборы (USD):</label>
        <input
          type="number"
          name="crowdfunding"
          value={formData.crowdfunding}
          onChange={handleChange}
          required
        />
      </div>

      <div>
        <label>Размер команды:</label>
        <input
          type="number"
          name="teamSize"
          value={formData.teamSize}
          onChange={handleChange}
          required
        />
      </div>

      <div>
        <label>Средняя оценка опыта команды (1-10):</label>
        <input
          type="number"
          name="teamExperience"
          value={formData.teamExperience}
          onChange={handleChange}
          min="1"
          max="10"
          required
        />
      </div>

      <div>
        <label>Средний опыт в данной области (1-10):</label>
        <input
          type="number"
          name="fieldExperience"
          value={formData.fieldExperience}
          onChange={handleChange}
          min="1"
          max="10"
          required
        />
      </div>

      <div>
        <label>Технологический уровень (1-10):</label>
        <input
          type="number"
          name="techLevel"
          value={formData.techLevel}
          onChange={handleChange}
          min="1"
          max="10"
          required
        />
      </div>

      <div>
        <label>Технологические инвестиции (USD):</label>
        <input
          type="number"
          name="techInvestments"
          value={formData.techInvestments}
          onChange={handleChange}
          required
        />
      </div>

      <div>
        <label>Уровень конкуренции (1-10):</label>
        <input
          type="number"
          name="competitionLevel"
          value={formData.competitionLevel}
          onChange={handleChange}
          min="1"
          max="10"
          required
        />
      </div>

      <div>
        <label>Примерное количество конкурентов:</label>
        <input
          type="number"
          name="competitorsCount"
          value={formData.competitorsCount}
          onChange={handleChange}
          required
        />
      </div>

      <div>
        <label>Уровень спроса (1-10):</label>
        <input
          type="number"
          name="demandLevel"
          value={formData.demandLevel}
          onChange={handleChange}
          min="1"
          max="10"
          required
        />
      </div>

      <div>
        <label>Целевая аудитория (человек):</label>
        <input
          type="number"
          name="targetAudience"
          value={formData.targetAudience}
          onChange={handleChange}
          required
        />
      </div>

      <div>
        <label>Примерный размер рынка (человек):</label>
        <input
          type="number"
          name="marketSize"
          value={formData.marketSize}
          onChange={handleChange}
          required
        />
      </div>

      <button type="submit">Отправить данные</button>
    </form>
  );
};

export default StartupForm;
