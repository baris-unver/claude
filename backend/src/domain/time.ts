/** Saat dilimi yardımcıları. Türkiye (Europe/Istanbul) UTC+3, yaz saati uygulanmaz. */

export interface LocalParts {
  year: number;
  month: number; // 1-12
  day: number;
  hour: number;
  minute: number;
  second: number;
  weekday: number; // 0 = Pazar
}

const cache = new Map<string, Intl.DateTimeFormat>();

function formatter(tz: string): Intl.DateTimeFormat {
  let f = cache.get(tz);
  if (!f) {
    f = new Intl.DateTimeFormat('en-US', {
      timeZone: tz,
      hourCycle: 'h23',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      weekday: 'short',
    });
    cache.set(tz, f);
  }
  return f;
}

const WEEKDAYS: Record<string, number> = { Sun: 0, Mon: 1, Tue: 2, Wed: 3, Thu: 4, Fri: 5, Sat: 6 };

export function localParts(tz: string, at: Date): LocalParts {
  const parts: Record<string, string> = {};
  for (const p of formatter(tz).formatToParts(at)) parts[p.type] = p.value;
  return {
    year: Number(parts.year),
    month: Number(parts.month),
    day: Number(parts.day),
    hour: Number(parts.hour) % 24,
    minute: Number(parts.minute),
    second: Number(parts.second),
    weekday: WEEKDAYS[parts.weekday] ?? 0,
  };
}

export function tzOffsetMinutes(tz: string, at: Date): number {
  const lp = localParts(tz, at);
  const asUtc = Date.UTC(lp.year, lp.month - 1, lp.day, lp.hour, lp.minute, lp.second);
  return Math.round((asUtc - Math.floor(at.getTime() / 1000) * 1000) / 60_000);
}

/** Yerel tarih/saat bileşenlerinden UTC anı üretir. */
export function zonedToUtc(tz: string, year: number, month: number, day: number, hour: number, minute: number): Date {
  const guess = Date.UTC(year, month - 1, day, hour, minute);
  const offset = tzOffsetMinutes(tz, new Date(guess));
  return new Date(guess - offset * 60_000);
}

export function parseHHMM(s: string): { h: number; m: number } {
  const match = /^([01]?\d|2[0-3]):([0-5]\d)$/.exec(s);
  if (!match) throw new Error(`Geçersiz saat biçimi: ${s}`);
  return { h: Number(match[1]), m: Number(match[2]) };
}

export function isHHMM(s: unknown): s is string {
  return typeof s === 'string' && /^([01]?\d|2[0-3]):([0-5]\d)$/.test(s);
}

export function dayKey(tz: string, at: Date): string {
  const lp = localParts(tz, at);
  return `${lp.year}-${String(lp.month).padStart(2, '0')}-${String(lp.day).padStart(2, '0')}`;
}

export function parseDayKey(key: string): { year: number; month: number; day: number } {
  const [y, m, d] = key.split('-').map(Number);
  return { year: y, month: m, day: d };
}

export function addMinutes(d: Date, minutes: number): Date {
  return new Date(d.getTime() + minutes * 60_000);
}
