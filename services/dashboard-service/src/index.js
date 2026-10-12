import express from "express";
import "dotenv/config";
import { pool } from "./db.js";
import {
  getDashboard,
  getProject,
  getReport,
  getRun,
  listProjectDocuments,
  listProjectRuns,
  listReports,
  listRunArtifacts,
} from "./routes/dashboard.js";

const app = express();

app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", process.env.CORS_ORIGIN || "*");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
  if (req.method === "OPTIONS") return res.sendStatus(204);
  next();
});

function asyncRoute(handler) {
  return (req, res, next) => Promise.resolve(handler(req, res, next)).catch(next);
}

app.get("/health", (req, res) => res.json({ status: "ok", service: "dashboard-service" }));
app.get("/dashboard", asyncRoute(getDashboard));
app.get("/projects/:projectId", asyncRoute(getProject));
app.get("/projects/:projectId/documents", asyncRoute(listProjectDocuments));
app.get("/projects/:projectId/runs", asyncRoute(listProjectRuns));
app.get("/runs/:runId", asyncRoute(getRun));
app.get("/runs/:runId/artifacts", asyncRoute(listRunArtifacts));
app.get("/reports", asyncRoute(listReports));
app.get("/reports/:reportId", asyncRoute(getReport));

app.use((err, req, res, next) => {
  if (res.headersSent) return next(err);
  const statusCode = Number.isInteger(err.statusCode) ? err.statusCode : 500;
  if (statusCode >= 500) {
    console.error("[dashboard-service] request failed:", err);
  }
  res.status(statusCode).json({
    error: statusCode >= 500 ? "Request failed" : err.message,
  });
});

const PORT = process.env.PORT || 4004;
const server = app.listen(PORT, () => {
  console.log(`[dashboard-service] listening on http://localhost:${PORT}`);
});

async function shutdown() {
  server.close(async () => {
    await pool.end();
    process.exit(0);
  });
}

process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);
