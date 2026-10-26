import bcrypt from "bcryptjs";

const SALT_ROUNDS = 12;

// Compared against when an email is unknown, so a failed sign-in takes the same
// time whether or not the account exists (prevents email enumeration by timing).
const DUMMY_HASH = bcrypt.hashSync("m10-dummy-password", SALT_ROUNDS);

export function hashPassword(password) {
  return bcrypt.hash(password, SALT_ROUNDS);
}

export function verifyPassword(password, passwordHash) {
  return bcrypt.compare(password, passwordHash ?? DUMMY_HASH);
}
