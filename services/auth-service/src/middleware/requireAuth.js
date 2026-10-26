import { unauthorized } from "../lib/httpError.js";
import { verifyToken } from "../lib/tokens.js";

// Rejects the request unless it carries a valid `Bearer` token for `audience`.
// On success the decoded claims are available as `req.auth`.
export function requireAuth(audience) {
  return (req, res, next) => {
    const [scheme, token] = (req.headers.authorization || "").split(" ");
    if (scheme !== "Bearer" || !token) {
      return next(unauthorized("Authentication required"));
    }
    try {
      req.auth = verifyToken(token, audience);
      next();
    } catch {
      next(unauthorized("Session is invalid or has expired"));
    }
  };
}
