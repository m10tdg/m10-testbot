import express from "express";
import "dotenv/config";
import { createProject, listProjects } from "./routes/createProject.js";

const app = express();
app.use(express.json());

// Simple health check - always add one of these to every service. It's the first
// thing you curl to confirm "is this process even running and reachable".
app.get("/health", (req, res) => res.json({ status: "ok", service: "project-service" }));

app.post("/projects", createProject);
app.get("/projects", listProjects);

const PORT = process.env.PORT || 4001;
app.listen(PORT, () => {
  console.log(`[project-service] listening on http://localhost:${PORT}`);
});
