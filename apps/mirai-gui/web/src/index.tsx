import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { WelcomeScreen } from "./welcome-screen";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("Root element is missing");

createRoot(root).render(
  <StrictMode>
    <WelcomeScreen />
  </StrictMode>,
);
