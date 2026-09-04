import { createHash, createHmac, randomBytes, randomInt, timingSafeEqual } from 'node:crypto';

export function sha256Hex(input: string): string {
  return createHash('sha256').update(input, 'utf8').digest('hex');
}

export function hmacHex(secret: string, message: string): string {
  return createHmac('sha256', secret).update(message, 'utf8').digest('hex');
}

export function safeEqualHex(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  return timingSafeEqual(Buffer.from(a, 'hex'), Buffer.from(b, 'hex'));
}

export function randomToken(bytes = 32): string {
  return randomBytes(bytes).toString('base64url');
}

/** 8 haneli, okunması kolay aktivasyon kodu (ör. 4831-9027). */
export function generateActivationCode(): string {
  const n = randomInt(0, 100_000_000);
  const s = String(n).padStart(8, '0');
  return `${s.slice(0, 4)}-${s.slice(4)}`;
}

export function normalizeActivationCode(code: string): string {
  return code.replace(/\D/g, '');
}

/**
 * Mobil istek imzası: HMAC-SHA256(deviceSecret, `${timestamp}.${method}.${path}.${sha256(body)}`).
 * Gövde yoksa boş dizenin özeti kullanılır.
 */
export function signRequest(secret: string, timestamp: string, method: string, path: string, rawBody: string): string {
  const canonical = `${timestamp}.${method.toUpperCase()}.${path}.${sha256Hex(rawBody)}`;
  return hmacHex(secret, canonical);
}

/** T.C. Kimlik Numarası algoritmik doğrulaması. */
export function isValidTurkishNationalId(id: string): boolean {
  if (!/^[1-9]\d{10}$/.test(id)) return false;
  const d = id.split('').map(Number);
  const odd = d[0] + d[2] + d[4] + d[6] + d[8];
  const even = d[1] + d[3] + d[5] + d[7];
  const tenth = ((odd * 7 - even) % 10 + 10) % 10;
  if (tenth !== d[9]) return false;
  const sum10 = d.slice(0, 10).reduce((a, b) => a + b, 0);
  return sum10 % 10 === d[10];
}
