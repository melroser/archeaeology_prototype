import { useState, useEffect } from "react";
import "./App.css";

export default function App() {
  const [chops, setChops] = useState([]);

  useEffect(() => {
    fetch("/api/chops")
      .then(res => res.json())
      .then(setChops);
  }, []);

  return (
    <div className="App">
      <h1>ARCHAEOLOG1ST — Chops</h1>
      <div className="grid">
        {chops.map((c, i) => (
          <div key={i} className="card">
            <p>{c.file}</p>
            <audio controls src={c.path}></audio>
          </div>
        ))}
      </div>
    </div>
  );
}




