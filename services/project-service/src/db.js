import pg from "pg";
import "dotenv/config";

// One shared connection pool for this service. Every route imports this same `pool`.
export const pool = new pg.Pool({
  connectionString: process.env.DATABASE_URL,
});
