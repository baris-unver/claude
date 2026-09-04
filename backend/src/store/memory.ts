import { randomUUID } from 'node:crypto';
import type {
  ActivationCode,
  AuditEntry,
  Checkin,
  CheckinRequest,
  CheckinSchedule,
  Curfew,
  Device,
  LocationSample,
  Officer,
  RefreshToken,
  Subject,
  Violation,
  Zone,
} from '../domain/types.js';
import type { Store } from './types.js';

class Table<T extends { id: string }> {
  private rows = new Map<string, T>();

  insert(row: Omit<T, 'id'>): T {
    const full = { ...(row as T), id: randomUUID() };
    this.rows.set(full.id, full);
    return full;
  }
  get(id: string): T | null {
    return this.rows.get(id) ?? null;
  }
  all(): T[] {
    return [...this.rows.values()];
  }
  find(pred: (r: T) => boolean): T[] {
    return this.all().filter(pred);
  }
  first(pred: (r: T) => boolean): T | null {
    return this.all().find(pred) ?? null;
  }
  patch(id: string, patch: Partial<T>): T | null {
    const cur = this.rows.get(id);
    if (!cur) return null;
    const next = { ...cur, ...patch, id };
    this.rows.set(id, next);
    return next;
  }
  delete(id: string): boolean {
    return this.rows.delete(id);
  }
  deleteWhere(pred: (r: T) => boolean): number {
    let n = 0;
    for (const [id, r] of this.rows) {
      if (pred(r)) {
        this.rows.delete(id);
        n++;
      }
    }
    return n;
  }
}

const byDateDesc = <T>(pick: (r: T) => Date) => (a: T, b: T) => pick(b).getTime() - pick(a).getTime();

/** Bellek içi depo: testler ve veritabanısız demo çalıştırma için. */
export class MemoryStore implements Store {
  private t = {
    officers: new Table<Officer>(),
    subjects: new Table<Subject>(),
    activationCodes: new Table<ActivationCode>(),
    devices: new Table<Device>(),
    refreshTokens: new Table<RefreshToken>(),
    zones: new Table<Zone>(),
    curfews: new Table<Curfew>(),
    schedules: new Table<CheckinSchedule>(),
    checkinRequests: new Table<CheckinRequest>(),
    checkins: new Table<Checkin>(),
    locations: new Table<LocationSample>(),
    violations: new Table<Violation>(),
    audit: new Table<AuditEntry>(),
  };

  async init(): Promise<void> {}
  async close(): Promise<void> {}

  officers: Store['officers'] = {
    create: async (o) => this.t.officers.insert({ ...o, createdAt: new Date() }),
    getById: async (id) => this.t.officers.get(id),
    getByUsername: async (u) => this.t.officers.first((o) => o.username === u),
    list: async () => this.t.officers.all(),
  };

  subjects: Store['subjects'] = {
    create: async (s) => {
      const now = new Date();
      return this.t.subjects.insert({ ...s, createdAt: now, updatedAt: now });
    },
    getById: async (id) => this.t.subjects.get(id),
    getByNationalId: async (nid) => this.t.subjects.first((s) => s.nationalId === nid),
    list: async (filter) =>
      this.t.subjects
        .find((s) => (!filter?.status || s.status === filter.status) && (!filter?.officerId || s.officerId === filter.officerId))
        .sort(byDateDesc((s) => s.createdAt)),
    update: async (id, patch) => this.t.subjects.patch(id, { ...patch, updatedAt: new Date() }),
  };

  activationCodes: Store['activationCodes'] = {
    create: async (c) => this.t.activationCodes.insert({ ...c, createdAt: new Date() }),
    findUsable: async (subjectId, codeHash, now) =>
      this.t.activationCodes.first(
        (c) => c.subjectId === subjectId && c.codeHash === codeHash && c.usedAt === null && c.expiresAt > now,
      ),
    markUsed: async (id, at) => {
      this.t.activationCodes.patch(id, { usedAt: at });
    },
  };

  devices: Store['devices'] = {
    create: async (d) => this.t.devices.insert(d),
    getById: async (id) => this.t.devices.get(id),
    getActiveBySubject: async (subjectId) =>
      this.t.devices
        .find((d) => d.subjectId === subjectId && d.revokedAt === null)
        .sort(byDateDesc((d) => d.registeredAt))[0] ?? null,
    update: async (id, patch) => this.t.devices.patch(id, patch),
    revokeAllForSubject: async (subjectId, at) => {
      for (const d of this.t.devices.find((d) => d.subjectId === subjectId && d.revokedAt === null)) {
        this.t.devices.patch(d.id, { revokedAt: at });
      }
    },
  };

  refreshTokens: Store['refreshTokens'] = {
    create: async (t) => this.t.refreshTokens.insert(t),
    getByHash: async (hash) => this.t.refreshTokens.first((t) => t.tokenHash === hash),
    revoke: async (id, at) => {
      this.t.refreshTokens.patch(id, { revokedAt: at });
    },
    revokeForDevice: async (deviceId, at) => {
      for (const t of this.t.refreshTokens.find((t) => t.deviceId === deviceId && t.revokedAt === null)) {
        this.t.refreshTokens.patch(t.id, { revokedAt: at });
      }
    },
  };

  zones: Store['zones'] = {
    create: async (z) => this.t.zones.insert({ ...z, createdAt: new Date() }),
    getById: async (id) => this.t.zones.get(id),
    listBySubject: async (subjectId) => this.t.zones.find((z) => z.subjectId === subjectId),
    delete: async (id) => this.t.zones.delete(id),
  };

  curfews: Store['curfews'] = {
    create: async (c) => this.t.curfews.insert({ ...c, createdAt: new Date() }),
    listBySubject: async (subjectId) => this.t.curfews.find((c) => c.subjectId === subjectId),
    delete: async (id) => this.t.curfews.delete(id),
  };

  schedules: Store['schedules'] = {
    create: async (s) => this.t.schedules.insert({ ...s, createdAt: new Date() }),
    getById: async (id) => this.t.schedules.get(id),
    listBySubject: async (subjectId) => this.t.schedules.find((s) => s.subjectId === subjectId),
    listActive: async () => this.t.schedules.find((s) => s.active),
    update: async (id, patch) => this.t.schedules.patch(id, patch),
    delete: async (id) => this.t.schedules.delete(id),
  };

  checkinRequests: Store['checkinRequests'] = {
    create: async (r) => this.t.checkinRequests.insert({ ...r, createdAt: new Date() }),
    getById: async (id) => this.t.checkinRequests.get(id),
    existsForScheduleDay: async (scheduleId, dayKey) =>
      this.t.checkinRequests.first((r) => r.scheduleId === scheduleId && r.dayKey === dayKey) !== null,
    listPending: async () => this.t.checkinRequests.find((r) => r.status === 'pending'),
    listBySubject: async (subjectId, limit) =>
      this.t.checkinRequests
        .find((r) => r.subjectId === subjectId)
        .sort(byDateDesc((r) => r.dueStart))
        .slice(0, limit),
    update: async (id, patch) => this.t.checkinRequests.patch(id, patch),
  };

  checkins: Store['checkins'] = {
    create: async (c) => this.t.checkins.insert(c),
    getById: async (id) => this.t.checkins.get(id),
    listBySubject: async (subjectId, limit) =>
      this.t.checkins
        .find((c) => c.subjectId === subjectId)
        .sort(byDateDesc((c) => c.submittedAt))
        .slice(0, limit),
    listForReview: async () =>
      this.t.checkins.find((c) => c.result === 'manual_review' && c.reviewerId === null).sort(byDateDesc((c) => c.submittedAt)),
    update: async (id, patch) => this.t.checkins.patch(id, patch),
  };

  locations: Store['locations'] = {
    insertMany: async (samples) => samples.map((s) => this.t.locations.insert(s)),
    latestBySubject: async (subjectId) =>
      this.t.locations
        .find((l) => l.subjectId === subjectId)
        .sort(byDateDesc((l) => l.recordedAt))[0] ?? null,
    listBySubject: async (subjectId, from, to, limit) =>
      this.t.locations
        .find((l) => l.subjectId === subjectId && l.recordedAt >= from && l.recordedAt <= to)
        .sort((a, b) => a.recordedAt.getTime() - b.recordedAt.getTime())
        .slice(-limit),
    latestForAll: async () => {
      const latest = new Map<string, LocationSample>();
      for (const l of this.t.locations.all()) {
        const cur = latest.get(l.subjectId);
        if (!cur || cur.recordedAt < l.recordedAt) latest.set(l.subjectId, l);
      }
      return [...latest.values()];
    },
    deleteOlderThan: async (before) => this.t.locations.deleteWhere((l) => l.recordedAt < before),
  };

  violations: Store['violations'] = {
    create: async (v) => this.t.violations.insert(v),
    getById: async (id) => this.t.violations.get(id),
    findOpen: async (subjectId, type) =>
      this.t.violations.first((v) => v.subjectId === subjectId && v.type === type && v.status === 'open'),
    list: async (filter) =>
      this.t.violations
        .find((v) => (!filter.subjectId || v.subjectId === filter.subjectId) && (!filter.status || v.status === filter.status))
        .sort(byDateDesc((v) => v.lastSeenAt))
        .slice(0, filter.limit ?? 200),
    update: async (id, patch) => this.t.violations.patch(id, patch),
  };

  audit: Store['audit'] = {
    append: async (e) => {
      this.t.audit.insert(e);
    },
    list: async (limit) => this.t.audit.all().sort(byDateDesc((e) => e.at)).slice(0, limit),
  };
}
