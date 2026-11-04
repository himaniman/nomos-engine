# Cloud study — what makes an AI scientist effective

Six parallel cloud-lab runs (`docs/cloud-lab.md` — read it first, every rule there holds)
that differ in ONE thing each, so the morning can answer a question one night cannot:
does the agent's success come from its EYES (pictures of lives), its MEMORY (one mind
across the whole night), or just from the model?

| arm | who answers the mailbox | observer | question |
|-----|-------------------------|----------|----------|
| FULL | you, the session's agent, all night | on (default) | baseline |
| BLIND | you, all night | OFF: `--observe-every 0` | do pictures of lives help? |
| FRESH | a NEW subagent per request, no history | on | does one remembering mind beat a fresh mind with only the lab's notes (night 1's setup, API model)? |

Each arm runs twice, on different selection worlds; the FINAL worlds (36-67) are the same
for all six, so every champion and every control meets the same exam:

| replicate | flags |
|-----------|-------|
| 1 | defaults: `--train 0-19 --holdout 20-35 --seed 0` |
| 2 | `--train 100-119 --holdout 120-135 --seed 1` |

Everything else is identical: `--agent --hours 12 --jobs 0 --control-law
laws/atlas_confine.nomos --anchor laws/atlas_ghost.nomos`, 8000 ticks, 32 FINAL worlds.

## Your session

Your arm and replicate are given in the task, e.g. `BLIND-2`. Then:

- branch: `git checkout -b cloud-lab-<arm>-<rep>` (lowercase, e.g. `cloud-lab-blind-2`);
  commit and push ONLY there
- run dir: `runs/cloud-lab/<arm>-<rep>` (e.g. `runs/cloud-lab/blind-2`)
- the command from `docs/cloud-lab.md` with the flags of your arm and replicate

## FRESH: how to answer

You are the dispatcher, not the scientist. For EVERY request:

1. Spawn a NEW subagent (never reuse one, never continue one) with this prompt, paths
   filled in:
   > You are the model of an artificial-life lab. Read `<run>/mailbox/system-XXXX.md`
   > (the rules and the DSL) and `<run>/mailbox/NNNN.request.md` (the question)
   > and, if the request names a picture, look at `<run>/mailbox/NNNN.png`. You may run
   > `.venv/bin/python -m cortex.mailbox check <file>` on a draft law. Read nothing else
   > in the repository or the run directory. Answer exactly in the shape the request
   > asks and write ONLY the answer to `<run>/mailbox/NNNN.reply.md`.
2. Check the reply file exists and is non-empty; do not edit, improve or add to it.
3. Do not think about the science yourself and do not pass the subagent your own notes —
   whatever it knows must come from the lab's request. That is the whole point of the arm.

## FULL and BLIND

Answer yourself, as `docs/cloud-lab.md` says. BLIND: there will be no OBSERVE requests;
do not look at lives any other way (no GIFs, no track pictures — rule 2 of cloud-lab).

## The morning

As in `docs/cloud-lab.md`, but the journal entry is short and titled with the arm and
replicate (`## 2026-11-05 — cloud study BLIND-2`): the FINAL table, champion vs control
on FINAL, the train -> holdout -> final drops, the prediction-error line, and one
paragraph on what this arm's agent did differently. The cross-arm comparison is written
afterwards from all six branches.
