// services/user-service/src/middleware/syncUser.js
export async function syncCognitoUser(req, res, next) {
  const { sub, email, "custom:tenant_id": tenantId, "custom:role": role } = req.auth.claims; // from Cognito JWT
  const existing = await db.query(`SELECT id FROM users WHERE cognito_sub = $1`, [sub]);
  if (existing.rows.length === 0) {
    await db.query(
      `INSERT INTO users (tenant_id, cognito_sub, email, role) VALUES ($1,$2,$3,$4)`,
      [tenantId, sub, email, role]
    );
  }
  req.tenantContext = { tenantId, userId: existing.rows[0]?.id, role };
  next();
}