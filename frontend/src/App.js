import React from "react";
import StartupForm from "./components/StartupForm";

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Startup Success Prediction</h1>
      </header>
      <main>
        <StartupForm /> {/* Используем компонент формы */}
      </main>
    </div>
  );
}

export default App;
