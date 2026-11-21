# Movie Recommendation Agent frontend

React 19, TypeScript and Vite. The interface supports movie search and ratings,
clarifying questions, recommendations with expandable explanations, and history.
It uses the existing Service A contract; see [application setup](../README.md)
for database migrations, catalog seeding, and starting the API.

## Local development

```bash
cd src/app/frontend
npm ci
cp .env.example .env
npm run dev
```

`VITE_API_BASE_URL` selects Service A. `VITE_SERVICE_B_MODE=mock` displays the
sample-output notice (also the default when unset). This setting is display-only:
keep it aligned with the backend's `SERVICE_B_MODE`. Changing it does not connect
a real model. All Vite environment variables are public browser configuration.

## Verification

```bash
npm test
npm run lint
npm run build
npx playwright install chromium
npm run test:e2e
```

Browser tests require the repository-root `.venv` with the application Python
dependencies installed. They start an isolated in-memory SQLite API with 13
fixture movies on port 8017 and Vite on port 4173. They do not touch PostgreSQL,
download models, or assess recommendation quality. Both ports must be free.
Set `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` to use an existing Chromium executable.
Screenshots and failure traces are written to ignored `test-results/`.

## Behavior and limits

- Ratings and the current conversation survive view changes within the page.
  They are not restored after a full reload. There is no login or account recovery.
- Mutations run from user actions, never component mount effects. An in-flight
  guard prevents double submissions within this tab.
- Search ignores stale responses and supports keyboard selection and retry.
- Requests time out after 45 seconds. A timeout does not undo server writes.
  Failed turn retries read session status first; completed recommendations can
  be recovered without repeating the answer.
- If a lost response leaves an ambiguous pending question, the UI preserves the
  draft and asks the user to start a new recommendation. The current API cannot
  retrieve the pending question, and has no idempotency key contract. Session
  creation with a lost response can still leave an orphan session on the server.
- Movie metadata and explanation text come from the API. The UI hides raw model
  scores. Mock explanations remain explicitly sample output, not model evidence.
