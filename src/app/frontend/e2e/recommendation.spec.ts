import { expect, test, type Page } from '@playwright/test';

async function pick(page: Page, title: string) {
  const input = page.getByRole('searchbox', { name: 'Search a movie title' });
  await input.fill(title);
  await page.getByRole('button', { name: new RegExp(title) }).click();
}

for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
  test(`rate, clarify, recommend and return at ${viewport.width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize(viewport);
    const errors: string[] = [];
    page.on('pageerror', err => errors.push(err.message));
    let turns = 0;
    page.on('request', req => { if (/\/sessions\/\d+\/turn$/.test(req.url()) && req.method() === 'POST') turns++; });
    await page.goto('/');
    await expect(page.getByRole('heading', { name: /Start with what/ })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Get recommendations', exact: true })).toBeDisabled();
    await page.screenshot({ path: testInfo.outputPath('home.png'), fullPage: true });
    const search = page.getByRole('searchbox');
    await search.fill('Arrival');
    await expect(page.getByRole('button', { name: /Arrival/ })).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath('search.png'), fullPage: true });
    await search.press('ArrowDown');
    await page.keyboard.press('Enter');
    await page.getByRole('combobox', { name: /rating/i }).selectOption('4.5');
    await page.getByRole('button', { name: 'Get recommendations', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'One more thing.' })).toBeVisible();
    await page.getByRole('textbox', { name: 'Your answer' }).fill('Thoughtful science fiction');
    await page.getByRole('button', { name: 'History', exact: true }).click();
    await page.getByRole('button', { name: 'Current recommendations', exact: true }).click();
    await expect(page.getByRole('textbox', { name: 'Your answer' })).toHaveValue('Thoughtful science fiction');
    await page.getByRole('button', { name: 'Send', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Something to watch.' })).toBeVisible();
    await expect(page.locator('.recommendation-row')).toHaveCount(10);
    await expect(page.locator('.recommendation-row').filter({ hasText: 'Arrival' })).toHaveCount(0);
    await page.locator('.recommendation-row').first().click();
    await expect(page.locator('.recommendation-detail').first()).toBeVisible();
    await expect(page.getByText(/match 0/)).toHaveCount(0);
    await page.screenshot({ path: testInfo.outputPath('recommendations.png'), fullPage: true });
    await page.getByRole('button', { name: 'History', exact: true }).click();
    await expect(page.locator('.history-row')).toHaveCount(1);
    await page.locator('.history-row').click();
    await expect(page.locator('.recommendation-row')).toHaveCount(10);
    await page.screenshot({ path: testInfo.outputPath('history.png'), fullPage: true });
    await page.getByRole('button', { name: 'Edit ratings', exact: true }).click();
    await expect(page.getByText('Arrival (2016)', { exact: true })).toBeVisible();
    await expect(page.getByRole('combobox', { name: /rating/i })).toHaveValue('4.5');
    expect(turns).toBe(1);
    expect(errors).toEqual([]);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  });
}

test('three ratings recommend directly, search failure recovers, and create failure preserves the draft', async ({ page }) => {
  await page.goto('/');
  await page.route('**/movies/search?**', route => route.fulfill({ status: 503, contentType: 'application/json', body: '{"detail":"Search unavailable"}' }));
  await page.getByRole('searchbox').fill('Arrival');
  await expect(page.getByRole('alert')).toBeVisible();
  await page.unroute('**/movies/search?**');
  await page.getByRole('button', { name: 'Retry search' }).click();
  await page.getByRole('button', { name: /Arrival/ }).click();
  for (const title of ['Contact', 'Moon']) {
    await page.getByRole('button', { name: /Add another/ }).click();
    await pick(page, title);
  }
  await page.route('**/sessions', route => route.fulfill({ status: 503, contentType: 'application/json', body: '{"detail":"Temporarily unavailable"}' }));
  await page.getByRole('button', { name: 'Get recommendations', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('Temporarily unavailable');
  await expect(page.getByText('Arrival (2016)', { exact: true })).toBeVisible();
  await page.unroute('**/sessions');
  await page.getByRole('button', { name: 'Get recommendations', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Something to watch.' })).toBeVisible();
  await expect(page.getByRole('textbox', { name: 'Your answer' })).toHaveCount(0);
});

test('failed new session does not prevent answering the existing conversation', async ({ page }) => {
  await page.goto('/');
  await pick(page, 'Contact');
  await page.getByRole('button', { name: 'Get recommendations', exact: true }).click();
  await expect(page.getByRole('textbox', { name: 'Your answer' })).toBeVisible();
  await page.getByRole('button', { name: 'Edit ratings', exact: true }).click();
  await page.route('**/sessions', route => route.fulfill({ status: 503, contentType: 'application/json', body: '{"detail":"Creation unavailable"}' }));
  await page.getByRole('button', { name: 'Get recommendations', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('Creation unavailable');
  await page.getByRole('button', { name: 'Current recommendations', exact: true }).click();
  await expect(page.getByRole('alert')).toHaveCount(0);
  await page.getByRole('textbox', { name: 'Your answer' }).fill('Thoughtful science fiction');
  await page.getByRole('button', { name: 'Send', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Something to watch.' })).toBeVisible();
  await expect(page.locator('.recommendation-row')).toHaveCount(10);
});
