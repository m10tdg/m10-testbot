// Sign-in for tenant users (the normal M10 sign-in page).
import { pool } from "../db.js";
import { unauthorized } from "../lib/httpError.js";
import { verifyPassword } from "../lib/passwords.js";
import { Audience, signToken } from "../lib/tokens.js";
import { requireEmail, requireString } from "../lib/validation.js";

const USER_COLUMNS = `
  u.id, u.name, u.email, u.role, u.password_hash,
  t.id AS tenant_id, t.name AS tenant_name, t.slug AS tenant_slug`;

function toUserResponse(row) {
  return {
    id: row.id,
    name: row.name,
    email: row.email,
    role: row.role,
    tenant: { id: row.tenant_id, name: row.tenant_name, slug: row.tenant_slug },
  };
}

export async function login(req, res) {
  const email = requireEmail(req.body?.email);
  const password = requireString(req.body?.password, "password");

  const result = await pool.query(
    `SELECT ${USER_COLUMNS}
     FROM users u JOIN tenants t ON t.id = u.tenant_id
     WHERE u.email = $1`,
    [email]
  );
  const user = result.rows[0];
  const passwordMatches = await verifyPassword(password, user?.password_hash);
  if (!user || !passwordMatches) throw unauthorized();

  const token = signToken(Audience.TENANT, user.id, {
    tenantId: user.tenant_id,
    role: user.role,
  });
  res.json({ token, user: toUserResponse(user) });
}

export async function getCurrentUser(req, res) {
  const result = await pool.query(
    `SELECT ${USER_COLUMNS}
     FROM users u JOIN tenants t ON t.id = u.tenant_id
     WHERE u.id = $1`,
    [req.auth.sub]
  );
  if (!result.rows[0]) throw unauthorized("Account no longer exists");
  res.json({ user: toUserResponse(result.rows[0]) });
}
