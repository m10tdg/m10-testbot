-- Authentication tables used by auth-service.
-- Applied automatically by the db-migrate service on every `docker compose up`.

-- M10 platform owners: they sign in to the owner panel and provision tenants.
-- They do not belong to any tenant.
CREATE TABLE IF NOT EXISTS platform_owners (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT NOT NULL,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Tenant users. Emails are unique platform-wide so the normal sign-in form only
-- needs email + password to resolve the tenant.
CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id),
  name TEXT NOT NULL,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'standard_user'
    CHECK (role IN ('admin', 'test_engineer', 'standard_user', 'developer')),
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS users_tenant_id_idx ON users (tenant_id);
