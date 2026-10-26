import { pool } from "../db.js";
import { hashPassword } from "./passwords.js";
import { requireEmail, requirePassword, requireString } from "./validation.js";

// Creates a platform owner. With `resetIfExists`, an existing owner with the same
// email gets the new name and password; otherwise it is left untouched.
// Returns the owner, or null when it already existed and was not reset.
export async function saveOwner({ email, password, name }, { resetIfExists = false } = {}) {
  const values = [
    requireString(name, "name"),
    requireEmail(email),
    await hashPassword(requirePassword(password)),
  ];
  const onConflict = resetIfExists
    ? "DO UPDATE SET name = EXCLUDED.name, password_hash = EXCLUDED.password_hash"
    : "DO NOTHING";

  const result = await pool.query(
    `INSERT INTO platform_owners (name, email, password_hash)
     VALUES ($1, $2, $3)
     ON CONFLICT (email) ${onConflict}
     RETURNING id, email`,
    values
  );
  return result.rows[0] ?? null;
}
