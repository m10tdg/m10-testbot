// libs/tenant-context.js
export function setTenantContext(pool) {
  return async (req, res, next) => {
    const client = await pool.connect();
    await client.query(`SET app.current_tenant = $1`, [req.tenantContext.tenantId]);
    req.db = client; // downstream handlers use req.db, not a shared pool, for this request
    res.on("finish", () => client.release());
    next();
  };
}