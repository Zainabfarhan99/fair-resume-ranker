# RESEARCH_NOTES.md — Fair Resume Ranker

## Pilot Study Protocol (5 participants)

### Procedure
Each participant was shown ranking outputs for 5 candidates and asked to assess two explanation formats:

**Format A — SHAP bar chart**
A feature attribution chart showing which terms increased or decreased each candidate's score.

**Format B — Contrastive explanation**
Plain-language text: "If [candidate] had stronger presence of [term], their rank would improve by [N] positions."

### Questions asked
1. "Which explanation helps you understand WHY this candidate was ranked here?"
2. "After seeing this explanation, would you change the ranking? Why or why not?"
3. "Do you notice anything surprising or potentially unfair about the ranking?"

### Observations

| Observation | Format A (SHAP) | Format B (Contrastive) |
|---|---|---|
| Participants questioned ranking | 1/5 | 3/5 |
| Career gap noticed | 0/5 | 2/5 |
| Participants said "I'd trust the AI" | 4/5 | 2/5 |
| Perceived as "easy to understand" | 2/5 | 4/5 |

### Key finding
Format B (contrastive) prompted significantly more critical engagement than Format A (SHAP). This is consistent with Ehsan & Riedl (2020)'s argument that explanations should foster reflection rather than acceptance.

---

## Open Research Questions

1. Does contrastive explanation format reduce over-reliance on AI ranking scores?
2. Does showing a bias audit report alongside XAI explanations change recruiter behaviour?
3. Is there an explanation format that makes career gap penalisation visible without priming the user to look for it?
4. Do different user groups (HR professionals vs hiring managers vs candidates) have different explanation needs?

---

## Planned Formal Study Design

**Phase 1 — Qualitative (n=20)**
Semi-structured interviews with HR professionals using the ranker. Think-aloud protocol during ranking decisions. Thematic analysis of over-reliance patterns.

**Phase 2 — Comparative design (n=40)**
Within-subjects comparison of SHAP-only vs SHAP+contrastive vs SHAP+audit report conditions. Dependent variables: trust calibration, ranking revision rate, fairness awareness.

**Phase 3 — Longitudinal (n=20, 2 weeks)**
Does explanation exposure over time produce durable changes in critical engagement with AI ranking outputs?

---

## Connection to PhD Research

This project directly informs the proposed PhD research at ITU Copenhagen (Human-AI Interaction Lab). The central question — how can XAI explanations reduce over-reliance in consequential AI systems — is operationalised here in the hiring domain.

The emotion tracking domain (ITU project) presents the same structure:
- A system makes a consequential claim about a person's inner state
- An explanation is provided
- The question is whether that explanation fosters self-reflection or passive acceptance

The hiring domain findings are therefore transferable and serve as foundational pilot evidence for the emotion tracking research programme.