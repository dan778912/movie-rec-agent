import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  workers: 1,
  use: {
    baseURL: 'http://127.0.0.1:4173',
    trace: 'retain-on-failure',
    launchOptions: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
      ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH } : {},
  },
  webServer: [
    { command: '../../../.venv/bin/python ../../../tests/ui_server.py', url: 'http://127.0.0.1:8017/openapi.json', reuseExistingServer: false },
    { command: 'npm run dev -- --host 127.0.0.1 --port 4173 --strictPort', url: 'http://127.0.0.1:4173', reuseExistingServer: false,
      env: { VITE_API_BASE_URL: 'http://127.0.0.1:8017', VITE_SERVICE_B_MODE: 'mock' } },
  ],
});
