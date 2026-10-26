// Creates (or resets the password of) an M10 platform owner.
// Owners cannot sign up through the API on purpose; they are provisioned here
// or through the INITIAL_OWNER_* variables at startup.
//
//   npm run create-owner -- <email> <password> "<name>"
import { pool } from "../src/db.js";
import { saveOwner } from "../src/lib/owners.js";

async function main() {
  const [email, password, name = "Platform Owner"] = process.argv.slice(2);
  const owner = await saveOwner({ email, password, name }, { resetIfExists: true });
  console.log(`[auth-service] platform owner ready: ${owner.email} (${owner.id})`);
}

main()
  .catch((err) => {
    console.error(`[auth-service] could not create owner: ${err.message}`);
    console.error('usage: npm run create-owner -- <email> <password> "<name>"');
    process.exitCode = 1;
  })
  .finally(() => pool.end());
