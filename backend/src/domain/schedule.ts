import type { CheckinSchedule, Curfew } from './types.js';
import { localParts, parseDayKey, parseHHMM, zonedToUtc } from './time.js';

export interface CheckinWindow {
  dueStart: Date;
  dueEnd: Date;
}

/**
 * Bir takvim için verilen yerel gün içindeki yoklama pencerelerini üretir.
 * `fixed`: pencere boyunca tek yoklama (karakolda imza günü mantığı).
 * `random`: pencere içinde rastgele anlarda `timesPerDay` adet kısa süreli yoklama.
 */
export function checkinWindowsForDay(
  schedule: CheckinSchedule,
  dayKey: string,
  tz: string,
  rng: () => number = Math.random,
): CheckinWindow[] {
  const { year, month, day } = parseDayKey(dayKey);
  const weekday = new Date(Date.UTC(year, month - 1, day)).getUTCDay();
  if (!schedule.daysOfWeek.includes(weekday)) return [];

  const ws = parseHHMM(schedule.windowStart);
  const we = parseHHMM(schedule.windowEnd);
  const start = zonedToUtc(tz, year, month, day, ws.h, ws.m);
  const end = zonedToUtc(tz, year, month, day, we.h, we.m);
  if (end <= start) return [];

  if (schedule.kind === 'fixed') return [{ dueStart: start, dueEnd: end }];

  const responseMs = schedule.responseMinutes * 60_000;
  const span = end.getTime() - start.getTime() - responseMs;
  if (span <= 0) return [{ dueStart: start, dueEnd: end }];

  const count = Math.max(1, schedule.timesPerDay);
  const starts: number[] = [];
  for (let i = 0; i < count; i++) starts.push(start.getTime() + Math.floor(rng() * span));
  starts.sort((a, b) => a - b);
  return starts.map((s) => ({ dueStart: new Date(s), dueEnd: new Date(s + responseMs) }));
}

/** Verilen anda ev hapsi (curfew) kuralının etkin olup olmadığı. Gece yarısını aşan aralıkları destekler. */
export function isInCurfew(curfew: Curfew, now: Date, tz: string): boolean {
  const lp = localParts(tz, now);
  const nowMin = lp.hour * 60 + lp.minute;
  const s = parseHHMM(curfew.startTime);
  const e = parseHHMM(curfew.endTime);
  const startMin = s.h * 60 + s.m;
  const endMin = e.h * 60 + e.m;

  if (startMin < endMin) {
    return curfew.daysOfWeek.includes(lp.weekday) && nowMin >= startMin && nowMin < endMin;
  }
  // Gece yarısını aşan aralık: akşam bölümü bugünün, sabah bölümü dünün gününe aittir.
  if (nowMin >= startMin) return curfew.daysOfWeek.includes(lp.weekday);
  if (nowMin < endMin) return curfew.daysOfWeek.includes((lp.weekday + 6) % 7);
  return false;
}

export function missedAfter(window: CheckinWindow, graceMinutes: number): Date {
  return new Date(window.dueEnd.getTime() + graceMinutes * 60_000);
}
