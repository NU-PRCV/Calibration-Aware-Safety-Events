# Final submission tasks

Report, presentation and code are due **Dec 14**. Target: **everything finished and submitted by Dec 6** (Dhanya flies Dec 9).

Click a checkbox on GitHub to mark a task done. Add notes after the task with ` — note: ...`. Keep owners and dates up to date.

**Owners:** D = Dhanya, T = Tejaswini

---

## Report (Overleaf)

- [ ] **T** · Nov 30 — Abstract (~150 words; check numbers against the final paper)
- [ ] **T** · Nov 30 — Related Work (at least 6 papers; notes are in the section's comments)
- [ ] **T** · Nov 30 — Review and expand "Perception stack" (Section 3.5)
- [ ] **T** · Nov 30 — Review the perception error budget (Section 4.6)
- [ ] **T** · Nov 30 — Pose-based foot points: results paragraph (optional; if not done, move to future work)
- [ ] **T** · Dec 1 — Conclusion: discussion and future work
- [ ] **D** · Dec 1 — Fill in the repo link in the open-source paragraph
- [ ] **D + T** · Dec 2 — Check all references against the original papers (split the list in half)
- [ ] **D + T** · Dec 3 — Trim to 8 pages, excluding references (after all text is in)
- [ ] **D + T** · Dec 3 — Final figure and table captions; no TODOs left
- [ ] **D + T** · Dec 4 — Read through the whole paper for flow and wording
- [ ] **T** · Dec 5 — Hide the red owner markers (`\newcommand{\owner}[1]{}`)
- [ ] **D + T** · Dec 5 — Add an AI-assistance note if the course policy requires it (check the syllabus)
- [ ] **D** · Dec 6 — Download the final PDF and submit

## Presentation

- [ ] **D** · Nov 27 — Check presentation requirements (length, format, recorded or live)
- [ ] **D** · Dec 3 — Slides: problem, method, calibration results, mitigations, guideline (reuse report figures)
- [ ] **T** · Dec 3 — Slides: detection, tracking and perception results
- [ ] **D** · Dec 4 — Demo video of the Streamlit app, 2–3 min (`streamlit run app/demo.py`)
- [ ] **D** · Dec 5 — Record the presentation
- [ ] **D** · Dec 6 — Submit the recording

## Code and repository

- [ ] **D** · Dec 1 — Pin exact versions: `pip freeze > requirements-lock.txt`; note the Python version in the README
- [ ] **D** · Dec 1 — README: repo link, `_loose` track files note, MOTA 0.185, Demo section, `run_all.sh` run time
- [ ] **T** · Nov 30 — Pose-based foot points code: PR, do not merge without review (optional)
- [ ] **D** · Dec 1 — Review the pose PR and run it on the dataset (only if the PR arrives)
- [ ] **D** · Dec 2 — GitHub Actions: run ruff and pytest on every push; add the badge to the README (optional)
- [ ] **D** · Dec 2 — Copy the final report figures to `docs/figures/`
- [ ] **D** · Dec 3 — Fresh-clone check: clone, new venv, install, `pytest` passes
- [ ] **D + T** · Dec 3 — All PRs merged; `main` is final
- [ ] **D** · Dec 3 — Repo accessible to graders
- [ ] **D** · Dec 6 — Tag the submitted version: `git tag -a v1.0` and push

## Coordination

- [x] **D** · Nov 24 — Share the Overleaf project with Tejaswini as editor
- [x] **D** · Nov 24 — Send Tejaswini this list and the deadline
- [ ] **D + T** · Nov 28 — Check in on progress