export class HttpError extends Error {
  constructor(statusCode, message) {
    super(message);
    this.statusCode = statusCode;
  }
}

export const badRequest = (message) => new HttpError(400, message);
export const unauthorized = (message = "Invalid email or password") => new HttpError(401, message);
export const forbidden = (message = "Forbidden") => new HttpError(403, message);
export const conflict = (message) => new HttpError(409, message);
