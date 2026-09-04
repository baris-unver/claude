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
  SubjectStatus,
  Violation,
  ViolationStatus,
  ViolationType,
  Zone,
} from '../domain/types.js';

export type NewSubject = Omit<Subject, 'id' | 'createdAt' | 'updatedAt'>;
export type NewOfficer = Omit<Officer, 'id' | 'createdAt'>;
export type NewActivationCode = Omit<ActivationCode, 'id' | 'createdAt'>;
export type NewDevice = Omit<Device, 'id'>;
export type NewRefreshToken = Omit<RefreshToken, 'id'>;
export type NewZone = Omit<Zone, 'id' | 'createdAt'>;
export type NewCurfew = Omit<Curfew, 'id' | 'createdAt'>;
export type NewSchedule = Omit<CheckinSchedule, 'id' | 'createdAt'>;
export type NewCheckinRequest = Omit<CheckinRequest, 'id' | 'createdAt'>;
export type NewCheckin = Omit<Checkin, 'id'>;
export type NewLocationSample = Omit<LocationSample, 'id'>;
export type NewViolation = Omit<Violation, 'id'>;
export type NewAuditEntry = Omit<AuditEntry, 'id'>;

/**
 * Depolama arayüzü. `MemoryStore` (test/demo) ve `PgStore` (üretim) bu sözleşmeyi uygular.
 */
export interface Store {
  init(): Promise<void>;
  close(): Promise<void>;

  officers: {
    create(o: NewOfficer): Promise<Officer>;
    getById(id: string): Promise<Officer | null>;
    getByUsername(username: string): Promise<Officer | null>;
    list(): Promise<Officer[]>;
  };

  subjects: {
    create(s: NewSubject): Promise<Subject>;
    getById(id: string): Promise<Subject | null>;
    getByNationalId(nationalId: string): Promise<Subject | null>;
    list(filter?: { status?: SubjectStatus; officerId?: string }): Promise<Subject[]>;
    update(id: string, patch: Partial<NewSubject>): Promise<Subject | null>;
  };

  activationCodes: {
    create(c: NewActivationCode): Promise<ActivationCode>;
    findUsable(subjectId: string, codeHash: string, now: Date): Promise<ActivationCode | null>;
    markUsed(id: string, at: Date): Promise<void>;
  };

  devices: {
    create(d: NewDevice): Promise<Device>;
    getById(id: string): Promise<Device | null>;
    getActiveBySubject(subjectId: string): Promise<Device | null>;
    update(id: string, patch: Partial<NewDevice>): Promise<Device | null>;
    revokeAllForSubject(subjectId: string, at: Date): Promise<void>;
  };

  refreshTokens: {
    create(t: NewRefreshToken): Promise<RefreshToken>;
    getByHash(tokenHash: string): Promise<RefreshToken | null>;
    revoke(id: string, at: Date): Promise<void>;
    revokeForDevice(deviceId: string, at: Date): Promise<void>;
  };

  zones: {
    create(z: NewZone): Promise<Zone>;
    getById(id: string): Promise<Zone | null>;
    listBySubject(subjectId: string): Promise<Zone[]>;
    delete(id: string): Promise<boolean>;
  };

  curfews: {
    create(c: NewCurfew): Promise<Curfew>;
    listBySubject(subjectId: string): Promise<Curfew[]>;
    delete(id: string): Promise<boolean>;
  };

  schedules: {
    create(s: NewSchedule): Promise<CheckinSchedule>;
    getById(id: string): Promise<CheckinSchedule | null>;
    listBySubject(subjectId: string): Promise<CheckinSchedule[]>;
    listActive(): Promise<CheckinSchedule[]>;
    update(id: string, patch: Partial<NewSchedule>): Promise<CheckinSchedule | null>;
    delete(id: string): Promise<boolean>;
  };

  checkinRequests: {
    create(r: NewCheckinRequest): Promise<CheckinRequest>;
    getById(id: string): Promise<CheckinRequest | null>;
    existsForScheduleDay(scheduleId: string, dayKey: string): Promise<boolean>;
    listPending(): Promise<CheckinRequest[]>;
    listBySubject(subjectId: string, limit: number): Promise<CheckinRequest[]>;
    update(id: string, patch: Partial<NewCheckinRequest>): Promise<CheckinRequest | null>;
  };

  checkins: {
    create(c: NewCheckin): Promise<Checkin>;
    getById(id: string): Promise<Checkin | null>;
    listBySubject(subjectId: string, limit: number): Promise<Checkin[]>;
    listForReview(): Promise<Checkin[]>;
    update(id: string, patch: Partial<NewCheckin>): Promise<Checkin | null>;
  };

  locations: {
    insertMany(samples: NewLocationSample[]): Promise<LocationSample[]>;
    latestBySubject(subjectId: string): Promise<LocationSample | null>;
    listBySubject(subjectId: string, from: Date, to: Date, limit: number): Promise<LocationSample[]>;
    latestForAll(): Promise<LocationSample[]>;
    deleteOlderThan(before: Date): Promise<number>;
  };

  violations: {
    create(v: NewViolation): Promise<Violation>;
    getById(id: string): Promise<Violation | null>;
    findOpen(subjectId: string, type: ViolationType): Promise<Violation | null>;
    list(filter: { subjectId?: string; status?: ViolationStatus; limit?: number }): Promise<Violation[]>;
    update(id: string, patch: Partial<NewViolation>): Promise<Violation | null>;
  };

  audit: {
    append(e: NewAuditEntry): Promise<void>;
    list(limit: number): Promise<AuditEntry[]>;
  };
}
