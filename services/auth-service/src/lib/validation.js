import { badRequest } from "./httpError.js";

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const SLUG_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
export const MIN_PASSWORD_LENGTH = 8;

export function requireString(value, name) {
  if (typeof value !== "string" || !value.trim()) {
    throw badRequest(`${name} is required`);
  }
  return value.trim();
}

export function requireEmail(value, name = "email") {
  const email = requireString(value, name).toLowerCase();
  if (!EMAIL_PATTERN.test(email)) throw badRequest(`${name} must be a valid email address`);
  return email;
}

export function requirePassword(value, name = "password") {
  if (typeof value !== "string" || value.length < MIN_PASSWORD_LENGTH) {
    throw badRequest(`${name} must be at least ${MIN_PASSWORD_LENGTH} characters`);
  }
  return value;
}

export function requireSlug(value, name = "slug") {
  const slug = requireString(value, name).toLowerCase();
  if (!SLUG_PATTERN.test(slug)) {
    throw badRequest(`${name} may only contain lowercase letters, numbers and single dashes`);
  }
  return slug;
}
