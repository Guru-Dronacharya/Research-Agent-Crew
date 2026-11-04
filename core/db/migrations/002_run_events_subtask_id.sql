-- 002_run_events_subtask_id.sql
-- Add subtask_id column to run_events for direct subtask tagging.
ALTER TABLE run_events ADD COLUMN subtask_id TEXT;
CREATE INDEX IF NOT EXISTS idx_run_events_seq ON run_events(seq);
