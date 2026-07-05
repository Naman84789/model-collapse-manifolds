# Publication route — from this folder to a submitted paper

You have: `main.tex`, `refs.bib`, `figs/` (6 figures), compiled `main.pdf`.
This is the exact sequence I recommend. Total elapsed to first public artifact: ~1 week.

---

## Step 0 — decisions only you can make (30 minutes)

1. **Name & email.** The paper currently uses your name + your gmail. If you want a
   different contact address (many people make a firstname.lastname@gmail for papers),
   change it in `main.tex` line ~40.
2. **Read the paper once, fully.** You are the sole author and take responsibility for
   every claim. The Acknowledgments section discloses AI assistance — venues (NeurIPS,
   ICML, ICLR) require disclosure of substantial LLM use and prohibit listing an AI as
   an author; the current wording is compliant. Do not remove it.

## Step 1 — put the code on GitHub (half a day)

Referees and readers will look. Create a public repo (e.g. `manifold-collapse-anchor`):

- Copy the ~14 load-bearing scripts listed in `metric-audit/README.md` (core pipeline
  table) + the jsonl result logs + `figures/` + that README itself as the repo README.
- Do NOT copy the whole metric-audit folder (it has 30+ exploratory scripts and an
  unrelated lm-evaluation-harness checkout — the README's quarantine list says what to
  skip).
- Add an MIT license file.
- Then in `main.tex`, Section "Reproducibility", add the repo URL.

## Step 2 — arXiv preprint (do this BEFORE any venue submission)

Why first: it timestamps priority (this is a hot area — the Cambridge group is active
RIGHT NOW), it's citable immediately, and every serious ML venue allows prior arXiv.

Mechanics:
1. Make an account at arxiv.org with your real name.
2. **Endorsement:** first-time submitters to cs.LG / stat.ML need an endorsement.
   After you start a submission, arXiv shows an endorsement code; a qualified endorser
   (anyone with a few prior arXiv papers in the category) enters it. If you know any
   researcher, ask them; otherwise arXiv sometimes waives it based on your email/history.
   This is the one step that can take a few days — start it early.
3. Upload the LaTeX source (`main.tex`, `refs.bib`, `figs/*.png`) — arXiv compiles
   LaTeX itself; upload the `.bbl` file too (produced next to main.pdf) since arXiv
   does not run BibTeX. Category: **cs.LG (primary), stat.ML (secondary)**.
4. Announcement takes 1-2 business days. You then have an arXiv ID forever.

## Step 3 — pick the venue (my concrete recommendation)

You said "top-20 AI publisher". The realistic ladder for this paper, given the July 2026
calendar:

| Venue | Deadline | Fit | Notes |
|---|---|---|---|
| **TMLR** (Transactions on ML Research) | rolling, any time | **Excellent — my recommendation** | Journal of the ICML/NeurIPS community. Reviews for *correctness and clarity*, not hype/novelty-lottery. No page limit (your appendix proofs fit naturally). Decision ~2 months. Accepted TMLR papers are widely cited and respected; certification (e.g. "Featured") adds prestige. |
| **ICLR 2027** | abstracts ~Sept 19, papers ~Sept 26, 2026 | Excellent | Top-tier conference (top-5 in AI by any ranking). Double-blind; 9-page body + unlimited appendix. Your body is ~11 pages single-column → reformat with the ICLR style file (they provide a template; mostly mechanical). |
| AISTATS 2027 | ~early Oct 2026 | Very good | Theory-friendly; slightly less visible than ICLR. Backup if you miss ICLR. |
| NeurIPS 2026 | passed (May 2026) | — | Next cycle May 2027. |
| JMLR | rolling | Good | Slower (6-12 months); use if you want a "real journal" line. |

**My recommendation: submit to arXiv now, then TMLR.** Reasons: (a) sole-author,
first-paper, no lab backing — TMLR's correctness-focused review is the fairest venue
for you; (b) no deadline pressure means you can respond to reviews carefully; (c) the
paper's strength is rigor + honesty, exactly what TMLR rewards; (d) if it lands well
you can still present it at a conference later via TMLR-to-conference tracks. If you
specifically want the conference brand, target **ICLR 2027 (Sept deadline)** — you have
2.5 months, which is comfortable.

## Step 4 — venue formatting (1 day, when you've chosen)

- **TMLR:** download the TMLR LaTeX style from jmlr.org/tmlr; move `\documentclass` and
  header; body text transfers unchanged. Submission is via OpenReview (make an account
  with your real identity; TMLR is double-blind — remove your name from the submitted
  copy, keep it on arXiv, that's allowed and normal).
- **ICLR:** same via the ICLR 2027 style file + OpenReview; compress body to 9 pages by
  pushing Section 7's table details and Section 8 expansions into the appendix.

## Step 5 — pre-submission checklist (referee-proofing)

- [ ] Every number in the paper matches `metric-audit/README.md` headline list.
- [ ] Repo URL live and README run-order works on a fresh clone (test once).
- [ ] The `.bbl` uploaded to arXiv; PDF renders figures correctly (check fig0/fig5).
- [ ] Double-blind copy: name/email/acknowledgments removed for OpenReview version
      (venues instruct exactly what to strip; arXiv version keeps them).
- [ ] Optional but high-value: email a preprint courtesy copy to the Cambridge authors
      (Khelifa, Turner, Venkataramanan) — "we answer your open Case 2" is the kind of
      email researchers *want* to receive, it can convert the most likely referees into
      informed readers, and Turner is exactly the right endorser profile for arXiv.

## What I'd do in your shoes, in order

1. Today: read `main.pdf` end to end; fix anything that isn't your voice.
2. Tomorrow: GitHub repo up; URL into the paper; recompile.
3. This week: arXiv account + endorsement request; submit preprint.
4. Next week: TMLR OpenReview submission (or start ICLR reformat if you prefer the
   conference route).
5. After arXiv announcement: the courtesy email to the Cambridge group.
