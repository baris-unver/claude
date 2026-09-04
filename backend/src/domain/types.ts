/**
 * Alan (domain) tipleri. Tüm tarih alanları UTC `Date` nesnesidir.
 */

export interface LatLng {
  lat: number;
  lng: number;
}

/** GeoJSON Polygon: koordinatlar [lng, lat] sırasındadır. */
export interface GeoPolygon {
  type: 'Polygon';
  coordinates: number[][][];
}

export type SubjectStatus = 'pending_activation' | 'active' | 'suspended' | 'completed';

export interface Subject {
  id: string;
  nationalId: string; // T.C. Kimlik No
  fullName: string;
  phone: string | null;
  caseNumber: string; // Denetimli serbestlik dosya numarası
  officerId: string;
  status: SubjectStatus;
  notes: string | null;
  faceReference: string | null; // base64 referans yüz görüntüsü (üretimde nesne deposu anahtarı)
  createdAt: Date;
  updatedAt: Date;
}

export interface ActivationCode {
  id: string;
  subjectId: string;
  codeHash: string;
  expiresAt: Date;
  usedAt: Date | null;
  createdAt: Date;
}

export interface Device {
  id: string;
  subjectId: string;
  platform: 'android' | 'ios';
  model: string;
  osVersion: string;
  appVersion: string;
  pushToken: string | null;
  secret: string; // İstek imzalama anahtarı (HMAC). Veritabanı seviyesinde şifrelenmelidir.
  isRooted: boolean;
  locationServicesEnabled: boolean;
  backgroundPermission: boolean;
  battery: number | null;
  registeredAt: Date;
  lastSeenAt: Date;
  revokedAt: Date | null;
}

export interface RefreshToken {
  id: string;
  deviceId: string;
  tokenHash: string;
  expiresAt: Date;
  revokedAt: Date | null;
}

export type ZoneKind = 'allowed' | 'forbidden' | 'home';

export interface Zone {
  id: string;
  subjectId: string;
  name: string;
  kind: ZoneKind;
  polygon: GeoPolygon;
  activeFrom: Date | null;
  activeTo: Date | null;
  createdAt: Date;
}

/** Gece/ev hapsi saatleri (ör. 22:00-06:00 evde bulunma zorunluluğu). */
export interface Curfew {
  id: string;
  subjectId: string;
  daysOfWeek: number[]; // 0 = Pazar ... 6 = Cumartesi (yerel saat)
  startTime: string; // "HH:MM"
  endTime: string; // "HH:MM" (başlangıçtan küçükse gece yarısını aşar)
  createdAt: Date;
}

export type ScheduleKind = 'fixed' | 'random';

export interface CheckinSchedule {
  id: string;
  subjectId: string;
  kind: ScheduleKind;
  daysOfWeek: number[];
  windowStart: string; // "HH:MM"
  windowEnd: string; // "HH:MM"
  timesPerDay: number; // random için günlük yoklama sayısı
  responseMinutes: number; // random yoklamada cevap süresi
  graceMinutes: number; // pencere bittikten sonra tolerans
  active: boolean;
  createdAt: Date;
}

export type CheckinRequestStatus = 'pending' | 'completed' | 'missed' | 'failed';

export type LivenessAction = 'look_straight' | 'turn_left' | 'turn_right' | 'blink' | 'smile' | 'nod';

export interface LivenessChallenge {
  nonce: string;
  actions: LivenessAction[];
  issuedAt: Date;
  expiresAt: Date;
}

export interface CheckinRequest {
  id: string;
  subjectId: string;
  scheduleId: string | null;
  dayKey: string; // "YYYY-MM-DD" (yerel gün)
  dueStart: Date;
  dueEnd: Date;
  graceMinutes: number;
  status: CheckinRequestStatus;
  challenge: LivenessChallenge | null;
  notifiedAt: Date | null;
  attempts: number;
  createdAt: Date;
}

export type CheckinResult = 'verified' | 'rejected' | 'manual_review';

export interface Checkin {
  id: string;
  requestId: string;
  subjectId: string;
  deviceId: string;
  submittedAt: Date;
  lat: number;
  lng: number;
  accuracy: number;
  faceScore: number;
  livenessPassed: boolean;
  result: CheckinResult;
  reviewerId: string | null;
  reviewNote: string | null;
  frames: Array<{ action: LivenessAction; image: string }>; // base64 kareler (üretimde nesne deposu)
}

export interface LocationSample {
  id: string;
  subjectId: string;
  deviceId: string;
  recordedAt: Date;
  receivedAt: Date;
  lat: number;
  lng: number;
  accuracy: number;
  speed: number | null;
  isMock: boolean;
  battery: number | null;
}

export type ViolationType =
  | 'ZONE_EXIT'
  | 'FORBIDDEN_ZONE_ENTRY'
  | 'CURFEW_VIOLATION'
  | 'MOCK_LOCATION'
  | 'DEVICE_COMPROMISED'
  | 'LOCATION_SERVICES_OFF'
  | 'HEARTBEAT_MISSED'
  | 'CHECKIN_MISSED'
  | 'CHECKIN_VERIFICATION_FAILED';

export type Severity = 'low' | 'medium' | 'high' | 'critical';
export type ViolationStatus = 'open' | 'acknowledged' | 'resolved' | 'dismissed';

export interface Violation {
  id: string;
  subjectId: string;
  type: ViolationType;
  severity: Severity;
  detectedAt: Date;
  lastSeenAt: Date;
  occurrences: number;
  details: Record<string, unknown>;
  status: ViolationStatus;
  resolvedBy: string | null;
  resolvedAt: Date | null;
  note: string | null;
}

export type OfficerRole = 'officer' | 'admin';

export interface Officer {
  id: string;
  username: string;
  passwordHash: string;
  fullName: string;
  role: OfficerRole;
  unit: string; // Denetimli serbestlik müdürlüğü
  createdAt: Date;
}

export interface AuditEntry {
  id: string;
  actorType: 'officer' | 'subject' | 'system';
  actorId: string;
  action: string;
  targetType: string;
  targetId: string | null;
  ip: string | null;
  at: Date;
  details: Record<string, unknown>;
}
