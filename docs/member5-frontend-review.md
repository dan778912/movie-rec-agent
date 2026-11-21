# Member 5 frontend review handoff

Date: 2026-11-12
Branch: `member5-service-a`
Review base: `1592f7c33f770686a659a43319b18add29f5c90d`
Status: independent review P2 addressed; approved for commit and branch publication.
No merge or new PR is included in this change.

## Review scope

Review the branch diff against the base above with `git diff 1592f7c..HEAD`.
The scope includes the new hook, shared recommendation component, tests,
configuration, and this document.

The first screen now centers movie search, ratings, and one primary submission
button. Secondary navigation appears after a session exists. The editorial
list layout uses responsive typography, restrained colors, and expandable
recommendation explanations. There are no new external metadata dependencies.

## Functional changes

- `src/app/frontend/src/App.tsx`: persistent rating form and conversation state,
  simplified navigation, skip link, and explicit mock-output notice.
- `src/app/frontend/src/useRecommendation.ts`: event-driven mutations, duplicate
  submission guard, retained answer draft, and status-based failure recovery.
  This removes the previous mount-effect request, which repeated under
  StrictMode and navigation.
- `src/app/frontend/src/components/MoviePicker.tsx`: debounced search, stale
  response rejection, visible failure/retry, keyboard selection, and dismissal.
- `RatingForm.tsx`: duplicate-movie validation and disabled submitting state.
- `Conversation.tsx`, `HistoryPanel.tsx`, `RecommendationList.tsx`: shared movie
  rows, expandable explanations, loading/empty/error states, and no raw scores.
- `src/app/frontend/src/api.ts`: readable structured validation errors, session
  status lookup, and a 45-second request deadline.
- `App.css` and `index.css`: desktop/mobile layout, focus states, reduced-motion
  handling, and restrained editorial styling.
- Added Vitest and Playwright tests, isolated Python browser-test API, test
  scripts/dependencies, environment example, and frontend operating instructions.

## Verification

- 14 frontend tests pass, including StrictMode/navigation request counts,
  retained drafts, failed answer retry, ambiguous response recovery, stale
  search responses, keyboard selection, duplicate selection/submission,
  structured API errors, and request timeout.
- 4 Chromium browser tests pass: full rating/clarification/recommendation/history
  flow at 1440px and 390px, plus direct recommendations with three ratings and
  injected search/session-creation failures. Browser tests check for JavaScript
  page errors and horizontal overflow; screenshots were inspected at both sizes.
- Frontend production build and lint pass. All 16 existing backend/data tests pass.
- `git diff --check` passes. All 24 changed/new text files were scanned; no Han
  characters were found. The existing Starlette/httpx deprecation warning remains.
- The browser fixture uses real Service A routes with SQLite and mock Service B.
  It is not a PostgreSQL deployment test or a real-model integration test.

## Review priorities and remaining limits

1. Inspect recovery transitions in `useRecommendation.ts`. Frontend guards are
   not server idempotency. A deadline cannot cancel committed server work;
   concurrent tabs and response loss during session creation remain limitations.
2. The API cannot read the pending clarification. An ambiguous response that
   leaves `awaiting_clarification` cannot safely be retried with the old answer;
   the UI preserves input and suggests starting a new session instead.
3. State survives navigation, not browser reload. No authentication was added.
4. Demo visibility is configured at build time and must match backend mode.
5. Service B integration, model quality, deployment, and CI remain separate work.

## Reproduce

```bash
.venv/bin/python -m unittest discover -s tests -v
cd src/app/frontend
npm test
npm run lint
npm run build
npm run test:e2e
```

See `src/app/frontend/README.md` for browser installation and fixture requirements.

## Independent review follow-up: session creation error isolation

The reported P2 issue was reproduced with two failing regression cases before
changing production code. Session creation and conversation errors previously
shared state, so a failed replacement session incorrectly forced the existing
conversation into recovery. It also overwrote an existing validation retry.

`useRecommendation.ts` now exposes a separate `creationError`. A failed
`createSession` updates only that error; the current conversation, answer,
and retry metadata remain intact. Conversation recovery resets only after
creation succeeds. `App.tsx` shows creation errors only on the ratings view.

Validation after the fix: 14 frontend tests, 4 Chromium browser tests, and all
16 backend/data tests pass; build, lint, and diff whitespace checks pass. The
new browser test reproduces the original 503/navigation/answer sequence using
real Service A routes with the isolated SQLite fixture. The user subsequently
authorized committing and pushing the reviewed changes to `member5-service-a`.
