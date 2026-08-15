-- TESTBOT database schema
-- This file runs AUTOMATICALLY the first time the Postgres container starts,
-- because docker-compose mounts this folder into Postgres's auto-init directory.
-- (Explained in the README - Step 1.)

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE tenants (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT NOT NULL,
  slug TEXT UNIQUE NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE projects (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id),
  name TEXT NOT NULL,
  slug TEXT NOT NULL,
  qdrant_collection TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (tenant_id, slug)
);

CREATE TABLE documents (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id),
  project_id UUID NOT NULL REFERENCES projects(id),
  name TEXT NOT NULL,
  doc_type TEXT NOT NULL DEFAULT 'requirement',
  s3_path TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'uploaded',
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE test_runs (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id),
  project_id UUID NOT NULL REFERENCES projects(id),
  run_source TEXT NOT NULL DEFAULT 'ui',
  url TEXT NOT NULL,
  prompt TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'queued',
  created_at TIMESTAMPTZ DEFAULT now(),
  completed_at TIMESTAMPTZ
);

CREATE TABLE reports (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id),
  project_id UUID NOT NULL REFERENCES projects(id),
  run_id UUID NOT NULL REFERENCES test_runs(id),
  status TEXT NOT NULL DEFAULT 'completed',
  severity TEXT,
  report_url TEXT NOT NULL,
  s3_path TEXT NOT NULL,
  format TEXT NOT NULL DEFAULT 'html',
  critical_count INT DEFAULT 0,
  warning_count INT DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Seed one tenant so you have a real tenantId to test with immediately.
-- We deliberately do NOT seed a project here: creating a project is a real API
-- call (Step 4 in the README) that also creates its Qdrant collection - a
-- project row without a matching Qdrant collection would be confusing to debug.
INSERT INTO tenants (id, name, slug)
VALUES ('11111111-1111-1111-1111-111111111111', 'Demo Company', 'demo-co');
