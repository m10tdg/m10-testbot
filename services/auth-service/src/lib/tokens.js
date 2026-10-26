import jwt from "jsonwebtoken";
import { config } from "../config.js";

// Each kind of session gets its own audience, so a tenant user's token can never
// be accepted by owner-only routes (and vice versa).
export const Audience = Object.freeze({
  OWNER: "m10-owner",
  TENANT: "m10-tenant",
});

const ISSUER = "m10-auth-service";

export function signToken(audience, subject, claims = {}) {
  return jwt.sign(claims, config.jwtSecret, {
    audience,
    subject,
    issuer: ISSUER,
    expiresIn: config.jwtExpiresIn,
  });
}

// Throws if the token is invalid, expired or issued for another audience.
export function verifyToken(token, audience) {
  return jwt.verify(token, config.jwtSecret, { audience, issuer: ISSUER });
}
