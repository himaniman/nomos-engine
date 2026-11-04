# Cloud lab — the session's own model is the physicist

A run of `cortex/night_lab.py` (the hours-long lab: the model designs STRUCTURE, a machine
tunes CONSTANTS, a control arm keeps both honest; see its docstring and the 2026-09-23
entry in `docs/lab-journal.md`) where **the model is you**, the agent of this session,
not an API. The lab writes each question to a file in a mailbox and waits; you read it,
answer into a file, and the lab goes on. Tuning, the control arm and holdout checks run
in the background while you think, exactly as they did while a slow local model thought.

You play three roles, and each request names which one it is:

| kind | what to write into `NNNN.reply.md` |
|------|------------------------------------|
| DESIGN | `HYPOTHESIS:` 2-3 sentences, `PREDICTION:` one number (mean meals on train), then the whole law in one ```` ```nomos ```` block |
| REPAIR | the same law, fixed exactly as the lab asked, whole, in one block |
| OBSERVE | look at the PNG named in the request, at most five plain sentences: what the agent does, where it fails, one mechanism that would fix it |
| THEORY | at most 150 words of plain text, written FROM THE EVIDENCE in the request |

Read the system text (`mailbox/system-*.md`, written once) before your first answer:
it is the DSL spec, the syntax traps and the known cliffs.

## Rules — what keeps the result worth anything

1. **You write structure, the machine writes constants.** Never hand-tune numbers; a law
   that only changes constants is caught as a twin and wastes a turn.
2. **All evaluation goes through the lab.** Do not live laws yourself (no
   `cortex.atlas_law --lives`, no `A.lives`, no scripts that run worlds): that is
   private tuning the control arm does not get. The one free tool is
   `python -m cortex.mailbox check <file>`: parse + the same value-field diagnosis the
   lab gives, on a test map, no world lived.
3. **FINAL worlds are sealed.** Seeds 36-67 are lived once, by the lab, at the end.
   Never run anything on them, never read `state.json["final"]` before the end.
4. **Do not touch the exam:** `cortex/world.py`, `cortex/tiles.py`, `cortex/atlas*.py`,
   `cortex/law_lab.py`, `cortex/night_lab.py` scoring, `laws/atlas_*.nomos`, `tests/`.
   If the INSTRUMENT is broken (the lab crashes, the observer reports nonsense), stop,
   fix only bookkeeping, write in the journal what and why, and resume.
   `tests/test_cortex_atlas.py::test_the_wave_routes_around_a_known_wall` was already
   failing before this run (stale since `leak` became 0.5) — leave it.
5. **The theory follows the evidence, not the conversation.** Night 1's last theory
   contradicted its own champion. Before a THEORY answer, re-read the evidence lines and
   the board in that request; say which result each claim rests on.
6. **Diversity over depth on one world.** Night 1 lost ~15 turns to variations of one
   idea aimed at one world where every law scores 0. When the lab says your law is a
   vocabulary twin, move along a different axis (memory, commitment, exploration,
   reading the map's geometry, the far-vs-near trade-off, ...).

## Run it

Setup (Ubuntu VM, Python 3.12):

```bash
python3 -m venv .venv && .venv/bin/pip install -q -e '.[dev]' pygame
export SDL_VIDEODRIVER=dummy      # pygame draws the champion GIF's frames (cortex/app.py)
.venv/bin/python -m pytest tests/test_night_lab.py -q        # ~1 min
nproc                                                         # jobs = this
```

Start the lab in the BACKGROUND (it runs for hours; a foreground command is killed after
~40 min). The run directory is TRACKED (`runs/`, not `out/`), so a commit is a full
snapshot and a fresh VM resumes from it:

```bash
RUN=runs/cloud-lab/$(date +%F)
mkdir -p $RUN
SDL_VIDEODRIVER=dummy nohup .venv/bin/python -m cortex.night_lab --agent --hours 10 --dir $RUN --jobs 0 \
  --control-law laws/atlas_confine.nomos --anchor laws/atlas_ghost.nomos \
  > $RUN/lab.log 2>&1 &
```

- `--control-law laws/atlas_confine.nomos`: night 1's champion is anchor #1 and the bar —
  the control arm tunes IT with the same budget, so a win means beating the best known
  structure, not the ghost. The ghost and the reference wave stay on the board.
- `--agent` = mailbox model, no embeddings (novelty is judged by the law's vocabulary),
  and PACING: the next DESIGN request waits until the last idea had its tuning budget
  (a law lives 20 worlds x 8000 ticks in ~1.5-2 min on 4 vCPU; ~15-20 min per idea).

The loop (repeat until it prints `NIGHT DONE`):

```bash
.venv/bin/python -m cortex.mailbox next $RUN/mailbox --wait 540
# -> REQUEST 0007, kind: DESIGN ... ; write $RUN/mailbox/0007.reply.md
```

`no request yet` = the lab is tuning; just call `next` again. Use `--wait 540` so each
call stays under the 10-minute foreground limit and the session never sits idle (an idle
session's VM pauses and can be reclaimed).

Do not shrink the worlds to go faster. The 2026-11-04 pilot (8 train worlds x 4000
ticks) showed tuning overfitting badly: the control went 17.4 train -> 11.9 holdout ->
13.1 FINAL while the same law untuned held 15.2, and FINAL differences on 8 worlds were
noise. The defaults (20 train, 16 holdout, 32 FINAL, 8000 ticks) are the experiment.

## Keep it alive and saved

- **Every ~30 minutes** (or after each holdout checkpoint in `journal.md`): commit and
  push the run directory to this session's branch:
  `git add $RUN && git commit -qm "chore(lab): cloud lab snapshot" && git push -q`.
  Pushing THIS session's branch is allowed; never push `main` or other branches.
- `mailbox status $RUN/mailbox` shows the heartbeat. If `status.json` "now" is more than
  ~10 minutes old and no request is pending, the lab died: look at `lab.log`, then start
  the SAME command again — it resumes from `state.json` (elapsed time is kept, an
  unanswered request is marked stale and asked afresh).
- On a fresh VM (the old one was reclaimed): pull the branch, set up, same command.

## The morning

When `next` prints `NIGHT DONE`, the lab has lived the FINAL worlds and written
`summary.md`, `champion.nomos`, `control.nomos`, `champion.gif`. Then:

1. Commit and push the whole run directory.
2. Add an entry to `docs/lab-journal.md` in the style of the 2026-09-23 one: the FINAL
   table, whether the model's structure beat the control (tuned confinement) on FINAL,
   train -> holdout -> final drops, the champion's mechanism, what the theory ended as and
   whether it agrees with the evidence, and what the run taught about the lab itself.
   Compare with night 1 (deepseek, 6 h): did a stronger model buy better structure?
3. One line under "State of play" in `CLAUDE.md`. Commit, push.
