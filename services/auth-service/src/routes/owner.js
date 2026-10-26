// Platform-owner panel: owner sign-in and tenant provisioning.
import { pool, withTransaction } from "../db.js";
import { conflict, unauthorized } from "../lib/httpError.js";
import { hashPassword, verifyPassword } from "../lib/passwords.js";
import { Audience, signToken } from "../lib/tokens.js";
import {
  requireEmail,
  requirePassword,
  requireSlug,
  requireString,
} from "../lib/validation.js";

const UNIQUE_VIOLATION = "23505";

function toOwnerResponse(row) {
  return { id: row.id, name: row.name, email: row.email };
}

export async function ownerLogin(req, res) {
  const email = requireEmail(req.body?.email);
  const password = requireString(req.body?.password, "password");

  const result = await pool.query(
    `SELECT id, name, email, password_hash FROM platform_owners WHERE email = $1`,
    [email]
  );
  const owner = result.rows[0];
  const passwordMatches = await verifyPassword(password, owner?.password_hash);
  if (!owner || !passwordMatches) throw unauthorized();

  const token = signToken(Audience.OWNER, owner.id);
  res.json({ token, owner: toOwnerResponse(owner) });
}

export async function getCurrentOwner(req, res) {
  const result = await pool.query(
    `SELECT id, name, email FROM platform_owners WHERE id = $1`,
    [req.auth.sub]
  );
  if (!result.rows[0]) throw unauthorized("Account no longer exists");
  res.json({ owner: toOwnerResponse(result.rows[0]) });
}

export async function listTenants(req, res) {
  const result = await pool.query(
    `SELECT t.id, t.name, t.slug, t.created_at,
            COUNT(u.id)::int AS user_count,
            (ARRAY_AGG(u.email ORDER BY u.created_at)
               FILTER (WHERE u.role = 'admin'))[1] AS admin_email
     FROM tenants t
     LEFT JOIN users u ON u.tenant_id = t.id
     GROUP BY t.id
     ORDER BY t.created_at DESC`
  );
  res.json({ items: result.rows });
}

// Creates a tenant together with its first admin account, atomically.
export async function createTenant(req, res) {
  const name = requireString(req.body?.name, "name");
  const slug = requireSlug(req.body?.slug);
  const adminName = requireString(req.body?.admin?.name, "admin.name");
  const adminEmail = requireEmail(req.body?.admin?.email, "admin.email");
  const adminPassword = requirePassword(req.body?.admin?.password, "admin.password");

  const passwordHash = await hashPassword(adminPassword);

  try {
    const created = await withTransaction(async (client) => {
      const tenant = await client.query(
        `INSERT INTO tenants (name, slug) VALUES ($1, $2)
         RETURNING id, name, slug, created_at`,
        [name, slug]
      );
      const admin = await client.query(
        `INSERT INTO users (tenant_id, name, email, password_hash, role)
         VALUES ($1, $2, $3, $4, 'admin')
         RETURNING id, name, email, role`,
        [tenant.rows[0].id, adminName, adminEmail, passwordHash]
      );
      return { tenant: tenant.rows[0], admin: admin.rows[0] };
    });
    res.status(201).json(created);
  } catch (err) {
    if (err.code === UNIQUE_VIOLATION) {
      throw conflict(
        err.table === "users"
          ? "A user with this email already exists"
          : "A tenant with this slug already exists"
      );
    }
    throw err;
  }
}
