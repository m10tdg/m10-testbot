ALTER TABLE test_runs
  ADD COLUMN IF NOT EXISTS baseline_s3_path TEXT;
