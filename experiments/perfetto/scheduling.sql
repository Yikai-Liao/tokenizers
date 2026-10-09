-- App annotations and Linux scheduler events are aligned by CLOCK_MONOTONIC.
-- Task intervals never overlap on one worker. SPAN_JOIN computes exact overlap.
CREATE PERFETTO TABLE bpe_tasks AS
SELECT s.id AS task_id, s.ts, s.dur, tt.utid, s.name,
       CAST(EXTRACT_ARG(s.arg_set_id, 'debug.round') AS INT) AS round_id
FROM slice s JOIN thread_track tt ON s.track_id = tt.id
WHERE s.name IN ('prepare.job', 'commit.owner', 'prepare.aa_validate', 'prepare.aa_choose', 'prepare.aa_job', 'apply.job');
CREATE PERFETTO TABLE bpe_sched AS
SELECT sc.id AS sched_id, sc.ts, sc.dur, sc.utid, sc.cpu
FROM sched sc
WHERE sc.dur > 0 AND sc.utid IN (SELECT DISTINCT utid FROM bpe_tasks);
CREATE VIRTUAL TABLE task_cpu USING SPAN_JOIN(bpe_tasks PARTITIONED utid, bpe_sched PARTITIONED utid);
SELECT name, COUNT(DISTINCT task_id) AS tasks, ROUND(SUM(dur)/1e6,6) AS on_cpu_ms
FROM task_cpu GROUP BY name ORDER BY name;

CREATE PERFETTO TABLE bpe_states AS
SELECT st.id AS state_id, st.ts, st.dur, st.utid, st.state
FROM thread_state st
WHERE st.dur > 0 AND st.utid IN (SELECT DISTINCT utid FROM bpe_tasks);
CREATE VIRTUAL TABLE task_states USING SPAN_JOIN(bpe_tasks PARTITIONED utid, bpe_states PARTITIONED utid);
SELECT name, state, ROUND(SUM(dur)/1e6,6) AS ms
FROM task_states GROUP BY name,state ORDER BY name,state;

-- A phase is global to the pool, so intersect every worker's states with it.
CREATE PERFETTO TABLE bpe_phases AS
SELECT s.id AS phase_id, s.ts, s.dur, s.name AS phase
FROM slice s WHERE s.name IN ('vocabulary','corpus_plan','cleanup','output_model','initial_index','materialize','select','prepare','apply','commit','release_candidates','release_events');
CREATE VIRTUAL TABLE phase_states USING SPAN_JOIN(bpe_phases, bpe_states PARTITIONED utid);
SELECT phase, state, ROUND(SUM(dur)/1e6,6) AS ms
FROM phase_states GROUP BY phase,state ORDER BY phase,state;

SELECT name, COUNT(*) AS slices, ROUND(SUM(dur)/1e6,6) AS wall_ms
FROM slice GROUP BY name ORDER BY name;
SELECT name, severity, value FROM stats WHERE value > 0 AND severity IN ('error', 'data_loss');
SELECT COUNT(*) AS incomplete_slices FROM slice WHERE dur < 0;
