import pg from 'pg';
import { runMigrations } from '../db/migrate.js';
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

const JSON_COLUMNS = new Set(['polygon', 'challenge', 'details', 'frames']);

const snake = (s: string) => s.replace(/[A-Z]/g, (c) => '_' + c.toLowerCase());
const camel = (s: string) => s.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase());

function toRow(obj: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) {
    if (v === undefined) continue;
    const col = snake(k);
    out[col] = JSON_COLUMNS.has(col) && v !== null ? JSON.stringify(v) : v;
  }
  return out;
}

function fromRow<T>(row: Record<string, unknown>): T {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(row)) out[camel(k)] = v;
  return out as T;
}

function reviveRequest(r: Record<string, unknown>): CheckinRequest {
  const req = fromRow<CheckinRequest>(r);
  if (req.challenge) {
    req.challenge = {
      ...req.challenge,
      issuedAt: new Date(req.challenge.issuedAt),
      expiresAt: new Date(req.challenge.expiresAt),
    };
  }
  return req;
}

/** PostgreSQL deposu. Sorgular tek tablo düzeyinde tutulmuştur; ölçekleme için PostGIS eklenebilir. */
export class PgStore implements Store {
  private pool: pg.Pool;

  constructor(connectionString: string) {
    this.pool = new pg.Pool({ connectionString, max: 10 });
  }

  async init(): Promise<void> {
    await runMigrations(this.pool);
  }

  async close(): Promise<void> {
    await this.pool.end();
  }

  private async insert<T>(table: string, obj: Record<string, unknown>, map: (r: Record<string, unknown>) => T = fromRow): Promise<T> {
    const row = toRow(obj);
    const cols = Object.keys(row);
    const params = cols.map((_, i) => `$${i + 1}`);
    const res = await this.pool.query(
      `INSERT INTO ${table} (${cols.join(', ')}) VALUES (${params.join(', ')}) RETURNING *`,
      Object.values(row),
    );
    return map(res.rows[0]);
  }

  private async patch<T>(
    table: string,
    id: string,
    obj: Record<string, unknown>,
    map: (r: Record<string, unknown>) => T = fromRow,
  ): Promise<T | null> {
    const row = toRow(obj);
    const cols = Object.keys(row);
    if (cols.length === 0) return this.one(table, id, map);
    const sets = cols.map((c, i) => `${c} = $${i + 2}`);
    const res = await this.pool.query(`UPDATE ${table} SET ${sets.join(', ')} WHERE id = $1 RETURNING *`, [
      id,
      ...Object.values(row),
    ]);
    return res.rows[0] ? map(res.rows[0]) : null;
  }

  private async one<T>(table: string, id: string, map: (r: Record<string, unknown>) => T = fromRow): Promise<T | null> {
    const res = await this.pool.query(`SELECT * FROM ${table} WHERE id = $1`, [id]);
    return res.rows[0] ? map(res.rows[0]) : null;
  }

  private async many<T>(sql: string, params: unknown[] = [], map: (r: Record<string, unknown>) => T = fromRow): Promise<T[]> {
    const res = await this.pool.query(sql, params);
    return res.rows.map(map);
  }

  private async first<T>(sql: string, params: unknown[] = [], map: (r: Record<string, unknown>) => T = fromRow): Promise<T | null> {
    const rows = await this.many(sql, params, map);
    return rows[0] ?? null;
  }

  officers: Store['officers'] = {
    create: (o) => this.insert<Officer>('officers', o as Record<string, unknown>),
    getById: (id) => this.one<Officer>('officers', id),
    getByUsername: (u) => this.first<Officer>('SELECT * FROM officers WHERE username = $1', [u]),
    list: () => this.many<Officer>('SELECT * FROM officers ORDER BY created_at'),
  };

  subjects: Store['subjects'] = {
    create: (s) => this.insert<Subject>('subjects', s as Record<string, unknown>),
    getById: (id) => this.one<Subject>('subjects', id),
    getByNationalId: (nid) => this.first<Subject>('SELECT * FROM subjects WHERE national_id = $1', [nid]),
    list: (filter) => {
      const where: string[] = [];
      const params: unknown[] = [];
      if (filter?.status) {
        params.push(filter.status);
        where.push(`status = $${params.length}`);
      }
      if (filter?.officerId) {
        params.push(filter.officerId);
        where.push(`officer_id = $${params.length}`);
      }
      const sql = `SELECT * FROM subjects ${where.length ? 'WHERE ' + where.join(' AND ') : ''} ORDER BY created_at DESC`;
      return this.many<Subject>(sql, params);
    },
    update: (id, patch) => this.patch<Subject>('subjects', id, { ...patch, updatedAt: new Date() }),
  };

  activationCodes: Store['activationCodes'] = {
    create: (c) => this.insert<ActivationCode>('activation_codes', c as Record<string, unknown>),
    findUsable: (subjectId, codeHash, now) =>
      this.first<ActivationCode>(
        'SELECT * FROM activation_codes WHERE subject_id = $1 AND code_hash = $2 AND used_at IS NULL AND expires_at > $3 LIMIT 1',
        [subjectId, codeHash, now],
      ),
    markUsed: async (id, at) => {
      await this.pool.query('UPDATE activation_codes SET used_at = $2 WHERE id = $1', [id, at]);
    },
  };

  devices: Store['devices'] = {
    create: (d) => this.insert<Device>('devices', d as Record<string, unknown>),
    getById: (id) => this.one<Device>('devices', id),
    getActiveBySubject: (subjectId) =>
      this.first<Device>(
        'SELECT * FROM devices WHERE subject_id = $1 AND revoked_at IS NULL ORDER BY registered_at DESC LIMIT 1',
        [subjectId],
      ),
    update: (id, patch) => this.patch<Device>('devices', id, patch as Record<string, unknown>),
    revokeAllForSubject: async (subjectId, at) => {
      await this.pool.query('UPDATE devices SET revoked_at = $2 WHERE subject_id = $1 AND revoked_at IS NULL', [subjectId, at]);
    },
  };

  refreshTokens: Store['refreshTokens'] = {
    create: (t) => this.insert<RefreshToken>('refresh_tokens', t as Record<string, unknown>),
    getByHash: (hash) => this.first<RefreshToken>('SELECT * FROM refresh_tokens WHERE token_hash = $1', [hash]),
    revoke: async (id, at) => {
      await this.pool.query('UPDATE refresh_tokens SET revoked_at = $2 WHERE id = $1', [id, at]);
    },
    revokeForDevice: async (deviceId, at) => {
      await this.pool.query('UPDATE refresh_tokens SET revoked_at = $2 WHERE device_id = $1 AND revoked_at IS NULL', [deviceId, at]);
    },
  };

  zones: Store['zones'] = {
    create: (z) => this.insert<Zone>('zones', z as Record<string, unknown>),
    getById: (id) => this.one<Zone>('zones', id),
    listBySubject: (subjectId) => this.many<Zone>('SELECT * FROM zones WHERE subject_id = $1 ORDER BY created_at', [subjectId]),
    delete: async (id) => (await this.pool.query('DELETE FROM zones WHERE id = $1', [id])).rowCount === 1,
  };

  curfews: Store['curfews'] = {
    create: (c) => this.insert<Curfew>('curfews', c as Record<string, unknown>),
    listBySubject: (subjectId) => this.many<Curfew>('SELECT * FROM curfews WHERE subject_id = $1 ORDER BY created_at', [subjectId]),
    delete: async (id) => (await this.pool.query('DELETE FROM curfews WHERE id = $1', [id])).rowCount === 1,
  };

  schedules: Store['schedules'] = {
    create: (s) => this.insert<CheckinSchedule>('checkin_schedules', s as Record<string, unknown>),
    getById: (id) => this.one<CheckinSchedule>('checkin_schedules', id),
    listBySubject: (subjectId) =>
      this.many<CheckinSchedule>('SELECT * FROM checkin_schedules WHERE subject_id = $1 ORDER BY created_at', [subjectId]),
    listActive: () => this.many<CheckinSchedule>('SELECT * FROM checkin_schedules WHERE active = true'),
    update: (id, patch) => this.patch<CheckinSchedule>('checkin_schedules', id, patch as Record<string, unknown>),
    delete: async (id) => (await this.pool.query('DELETE FROM checkin_schedules WHERE id = $1', [id])).rowCount === 1,
  };

  checkinRequests: Store['checkinRequests'] = {
    create: (r) => this.insert<CheckinRequest>('checkin_requests', r as Record<string, unknown>, reviveRequest),
    getById: (id) => this.one<CheckinRequest>('checkin_requests', id, reviveRequest),
    existsForScheduleDay: async (scheduleId, dayKey) =>
      (await this.pool.query('SELECT 1 FROM checkin_requests WHERE schedule_id = $1 AND day_key = $2 LIMIT 1', [scheduleId, dayKey]))
        .rowCount === 1,
    listPending: () => this.many<CheckinRequest>("SELECT * FROM checkin_requests WHERE status = 'pending'", [], reviveRequest),
    listBySubject: (subjectId, limit) =>
      this.many<CheckinRequest>(
        'SELECT * FROM checkin_requests WHERE subject_id = $1 ORDER BY due_start DESC LIMIT $2',
        [subjectId, limit],
        reviveRequest,
      ),
    update: (id, patch) => this.patch<CheckinRequest>('checkin_requests', id, patch as Record<string, unknown>, reviveRequest),
  };

  checkins: Store['checkins'] = {
    create: (c) => this.insert<Checkin>('checkins', c as Record<string, unknown>),
    getById: (id) => this.one<Checkin>('checkins', id),
    listBySubject: (subjectId, limit) =>
      this.many<Checkin>('SELECT * FROM checkins WHERE subject_id = $1 ORDER BY submitted_at DESC LIMIT $2', [subjectId, limit]),
    listForReview: () =>
      this.many<Checkin>("SELECT * FROM checkins WHERE result = 'manual_review' AND reviewer_id IS NULL ORDER BY submitted_at DESC"),
    update: (id, patch) => this.patch<Checkin>('checkins', id, patch as Record<string, unknown>),
  };

  locations: Store['locations'] = {
    insertMany: async (samples) => {
      if (samples.length === 0) return [];
      const cols = ['subject_id', 'device_id', 'recorded_at', 'received_at', 'lat', 'lng', 'accuracy', 'speed', 'is_mock', 'battery'];
      const values: unknown[] = [];
      const tuples = samples.map((s, i) => {
        values.push(s.subjectId, s.deviceId, s.recordedAt, s.receivedAt, s.lat, s.lng, s.accuracy, s.speed, s.isMock, s.battery);
        const base = i * cols.length;
        return `(${cols.map((_, j) => `$${base + j + 1}`).join(', ')})`;
      });
      return this.many<LocationSample>(
        `INSERT INTO location_samples (${cols.join(', ')}) VALUES ${tuples.join(', ')} RETURNING *`,
        values,
      );
    },
    latestBySubject: (subjectId) =>
      this.first<LocationSample>('SELECT * FROM location_samples WHERE subject_id = $1 ORDER BY recorded_at DESC LIMIT 1', [subjectId]),
    listBySubject: (subjectId, from, to, limit) =>
      this.many<LocationSample>(
        `SELECT * FROM (
           SELECT * FROM location_samples WHERE subject_id = $1 AND recorded_at BETWEEN $2 AND $3
           ORDER BY recorded_at DESC LIMIT $4
         ) t ORDER BY recorded_at ASC`,
        [subjectId, from, to, limit],
      ),
    latestForAll: () =>
      this.many<LocationSample>('SELECT DISTINCT ON (subject_id) * FROM location_samples ORDER BY subject_id, recorded_at DESC'),
    deleteOlderThan: async (before) =>
      (await this.pool.query('DELETE FROM location_samples WHERE recorded_at < $1', [before])).rowCount ?? 0,
  };

  violations: Store['violations'] = {
    create: (v) => this.insert<Violation>('violations', v as Record<string, unknown>),
    getById: (id) => this.one<Violation>('violations', id),
    findOpen: (subjectId, type) =>
      this.first<Violation>("SELECT * FROM violations WHERE subject_id = $1 AND type = $2 AND status = 'open' LIMIT 1", [
        subjectId,
        type,
      ]),
    list: (filter) => {
      const where: string[] = [];
      const params: unknown[] = [];
      if (filter.subjectId) {
        params.push(filter.subjectId);
        where.push(`subject_id = $${params.length}`);
      }
      if (filter.status) {
        params.push(filter.status);
        where.push(`status = $${params.length}`);
      }
      params.push(filter.limit ?? 200);
      const sql = `SELECT * FROM violations ${where.length ? 'WHERE ' + where.join(' AND ') : ''} ORDER BY last_seen_at DESC LIMIT $${params.length}`;
      return this.many<Violation>(sql, params);
    },
    update: (id, patch) => this.patch<Violation>('violations', id, patch as Record<string, unknown>),
  };

  audit: Store['audit'] = {
    append: async (e) => {
      await this.insert<AuditEntry>('audit_log', e as Record<string, unknown>);
    },
    list: (limit) => this.many<AuditEntry>('SELECT * FROM audit_log ORDER BY at DESC LIMIT $1', [limit]),
  };
}
