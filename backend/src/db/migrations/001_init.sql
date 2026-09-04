-- Denetimli serbestlik mobil takip sistemi - başlangıç şeması
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS officers (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  username text UNIQUE NOT NULL,
  password_hash text NOT NULL,
  full_name text NOT NULL,
  role text NOT NULL CHECK (role IN ('officer', 'admin')),
  unit text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS subjects (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  national_id text UNIQUE NOT NULL,
  full_name text NOT NULL,
  phone text,
  case_number text NOT NULL,
  officer_id uuid NOT NULL REFERENCES officers(id),
  status text NOT NULL CHECK (status IN ('pending_activation', 'active', 'suspended', 'completed')),
  notes text,
  face_reference text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS activation_codes (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  subject_id uuid NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
  code_hash text NOT NULL,
  expires_at timestamptz NOT NULL,
  used_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS activation_codes_subject_idx ON activation_codes(subject_id);

CREATE TABLE IF NOT EXISTS devices (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  subject_id uuid NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
  platform text NOT NULL CHECK (platform IN ('android', 'ios')),
  model text NOT NULL,
  os_version text NOT NULL,
  app_version text NOT NULL,
  push_token text,
  secret text NOT NULL,
  is_rooted boolean NOT NULL DEFAULT false,
  location_services_enabled boolean NOT NULL DEFAULT true,
  background_permission boolean NOT NULL DEFAULT true,
  battery double precision,
  registered_at timestamptz NOT NULL DEFAULT now(),
  last_seen_at timestamptz NOT NULL DEFAULT now(),
  revoked_at timestamptz
);
CREATE INDEX IF NOT EXISTS devices_subject_idx ON devices(subject_id) WHERE revoked_at IS NULL;

CREATE TABLE IF NOT EXISTS refresh_tokens (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  device_id uuid NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
  token_hash text UNIQUE NOT NULL,
  expires_at timestamptz NOT NULL,
  revoked_at timestamptz
);

CREATE TABLE IF NOT EXISTS zones (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  subject_id uuid NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
  name text NOT NULL,
  kind text NOT NULL CHECK (kind IN ('allowed', 'forbidden', 'home')),
  polygon jsonb NOT NULL,
  active_from timestamptz,
  active_to timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS zones_subject_idx ON zones(subject_id);

CREATE TABLE IF NOT EXISTS curfews (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  subject_id uuid NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
  days_of_week int[] NOT NULL,
  start_time text NOT NULL,
  end_time text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS checkin_schedules (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  subject_id uuid NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
  kind text NOT NULL CHECK (kind IN ('fixed', 'random')),
  days_of_week int[] NOT NULL,
  window_start text NOT NULL,
  window_end text NOT NULL,
  times_per_day int NOT NULL DEFAULT 1,
  response_minutes int NOT NULL DEFAULT 30,
  grace_minutes int NOT NULL DEFAULT 15,
  active boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS checkin_schedules_active_idx ON checkin_schedules(active);

CREATE TABLE IF NOT EXISTS checkin_requests (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  subject_id uuid NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
  schedule_id uuid REFERENCES checkin_schedules(id) ON DELETE SET NULL,
  day_key text NOT NULL,
  due_start timestamptz NOT NULL,
  due_end timestamptz NOT NULL,
  grace_minutes int NOT NULL DEFAULT 15,
  status text NOT NULL CHECK (status IN ('pending', 'completed', 'missed', 'failed')),
  challenge jsonb,
  notified_at timestamptz,
  attempts int NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS checkin_requests_pending_idx ON checkin_requests(status) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS checkin_requests_schedule_day_idx ON checkin_requests(schedule_id, day_key);

CREATE TABLE IF NOT EXISTS checkins (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  request_id uuid NOT NULL REFERENCES checkin_requests(id) ON DELETE CASCADE,
  subject_id uuid NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
  device_id uuid NOT NULL REFERENCES devices(id),
  submitted_at timestamptz NOT NULL,
  lat double precision NOT NULL,
  lng double precision NOT NULL,
  accuracy double precision NOT NULL,
  face_score double precision NOT NULL,
  liveness_passed boolean NOT NULL,
  result text NOT NULL CHECK (result IN ('verified', 'rejected', 'manual_review')),
  reviewer_id uuid REFERENCES officers(id),
  review_note text,
  frames jsonb NOT NULL DEFAULT '[]'::jsonb
);
CREATE INDEX IF NOT EXISTS checkins_subject_idx ON checkins(subject_id, submitted_at DESC);

CREATE TABLE IF NOT EXISTS location_samples (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  subject_id uuid NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
  device_id uuid NOT NULL REFERENCES devices(id),
  recorded_at timestamptz NOT NULL,
  received_at timestamptz NOT NULL DEFAULT now(),
  lat double precision NOT NULL,
  lng double precision NOT NULL,
  accuracy double precision NOT NULL,
  speed double precision,
  is_mock boolean NOT NULL DEFAULT false,
  battery double precision
);
CREATE INDEX IF NOT EXISTS location_samples_subject_time_idx ON location_samples(subject_id, recorded_at DESC);

CREATE TABLE IF NOT EXISTS violations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  subject_id uuid NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
  type text NOT NULL,
  severity text NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
  detected_at timestamptz NOT NULL,
  last_seen_at timestamptz NOT NULL,
  occurrences int NOT NULL DEFAULT 1,
  details jsonb NOT NULL DEFAULT '{}'::jsonb,
  status text NOT NULL CHECK (status IN ('open', 'acknowledged', 'resolved', 'dismissed')),
  resolved_by uuid REFERENCES officers(id),
  resolved_at timestamptz,
  note text
);
CREATE INDEX IF NOT EXISTS violations_open_idx ON violations(subject_id, type) WHERE status = 'open';
CREATE INDEX IF NOT EXISTS violations_last_seen_idx ON violations(last_seen_at DESC);

CREATE TABLE IF NOT EXISTS audit_log (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  actor_type text NOT NULL,
  actor_id text NOT NULL,
  action text NOT NULL,
  target_type text NOT NULL,
  target_id text,
  ip text,
  at timestamptz NOT NULL DEFAULT now(),
  details jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS audit_log_at_idx ON audit_log(at DESC);
