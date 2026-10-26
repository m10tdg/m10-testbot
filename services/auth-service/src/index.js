import express from "express";
import { config } from "./config.js";
import { pool } from "./db.js";
import { saveOwner } from "./lib/owners.js";
import { Audience } from "./lib/tokens.js";
import { requireAuth } from "./middleware/requireAuth.js";
import { getCurrentUser, login } from "./routes/auth.js";
import {
  createTenant,
  getCurrentOwner,
  listTenants,
  ownerLogin,
} from "./routes/owner.js";

const app = express();
app.use(express.json());

app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", config.corsOrigin);
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  if (req.method === "OPTIONS") return res.sendStatus(204);
  next();
});

function asyncRoute(handler) {
  return (req, res, next) => Promise.resolve(handler(req, res, next)).catch(next);
}

const requireOwner = requireAuth(Audience.OWNER);
const requireTenantUser = requireAuth(Audience.TENANT);

app.get("/health", (req, res) => res.json({ status: "ok", service: "auth-service" }));

// Tenant users
app.post("/auth/login", asyncRoute(login));
app.get("/auth/me", requireTenantUser, asyncRoute(getCurrentUser));

// Platform owners
app.post("/owner/login", asyncRoute(ownerLogin));
app.get("/owner/me", requireOwner, asyncRoute(getCurrentOwner));
app.get("/owner/tenants", requireOwner, asyncRoute(listTenants));
app.post("/owner/tenants", requireOwner, asyncRoute(createTenant));

app.use((err, req, res, next) => {
  if (res.headersSent) return next(err);
  // express.json() reports malformed bodies with a 4xx `status`.
  const statusCode = Number.isInteger(err.statusCode) ? err.statusCode : err.status || 500;
  if (statusCode >= 500) {
    console.error("[auth-service] request failed:", err);
  }
  res.status(statusCode).json({
    error: statusCode >= 500 ? "Request failed" : err.message,
  });
});

async function ensureInitialOwner() {
  if (!config.initialOwner) return;
  const created = await saveOwner(config.initialOwner);
  if (created) console.log(`[auth-service] created initial platform owner ${created.email}`);
}

await ensureInitialOwner();

const server = app.listen(config.port, () => {
  console.log(`[auth-service] listening on http://localhost:${config.port}`);
});

async function shutdown() {
  server.close(async () => {
    await pool.end();
    process.exit(0);
  });
}

process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);
