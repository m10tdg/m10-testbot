import "dotenv/config";

function required(name) {
  const value = process.env[name];
  if (!value) throw new Error(`[auth-service] missing required env var ${name}`);
  return value;
}

export const config = {
  port: Number(process.env.PORT || 4005),
  databaseUrl: required("DATABASE_URL"),
  jwtSecret: required("JWT_SECRET"),
  jwtExpiresIn: process.env.JWT_EXPIRES_IN || "8h",
  corsOrigin: process.env.CORS_ORIGIN || "*",
  // Optional: the first platform owner, created at startup if it does not exist yet.
  initialOwner: process.env.INITIAL_OWNER_EMAIL
    ? {
        email: process.env.INITIAL_OWNER_EMAIL,
        password: process.env.INITIAL_OWNER_PASSWORD,
        name: process.env.INITIAL_OWNER_NAME || "Platform Owner",
      }
    : null,
};
