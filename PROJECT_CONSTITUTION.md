# Project Cortana Constitution

## Mission

Project Cortana is an AI-powered cybersecurity assistant built using modern software engineering principles.

Every feature must prioritize:

- Security
- Reliability
- Modularity
- Readability
- Testability
- Performance

---

## Architecture

Project structure:

project-cortana/
│
├── src/
├── tests/
├── docs/
├── config/
├── requirements.txt
├── README.md

---

## Coding Standards

- Use Python 3.13+
- Use type hints everywhere
- Use docstrings
- Follow PEP8
- Functions should do ONE job.
- Avoid duplicated code.
- Prefer composition over large classes.

---

## Security Rules

Never hardcode:

- passwords
- API keys
- tokens
- secrets

Always validate user input.

Never trust external data.

---

## Git Rules

Every completed feature requires:

- commit
- push to GitHub

Commit messages should clearly describe the work completed.

---

## Testing

Every new feature requires testing before commit.

Future unit tests will be stored in:

tests/

---

## AI Development Workflow

1. ChatGPT designs architecture.
2. Claude reviews prompts and code.
3. Cursor implements code.
4. ChatGPT reviews architecture.
5. Claude performs independent review.
6. Approved changes are committed.
7. Push to GitHub.

---

## Goal

Build a professional AI cybersecurity platform suitable for production deployment.
