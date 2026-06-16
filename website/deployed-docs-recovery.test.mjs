import test from 'node:test';
import assert from 'node:assert/strict';
import { access, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const websiteRoot = path.dirname(fileURLToPath(import.meta.url));

async function readSource(relativePath) {
  return readFile(path.join(websiteRoot, relativePath), 'utf8');
}

test('homepage matches the intended current docs IA and comparison layout', async () => {
  const source = await readSource("app/(home)/page.tsx");

  assert.match(source, /export const metadata/);
  assert.match(source, /HomeCodeTerminal/);
  // Logo is now in the masthead (beside the wordmark) rather than a middle slot
  // of the metric row — guard that the brush mark still renders on the page.
  assert.match(source, /withBasePath\('\/logo\.png'\)/);
  assert.match(source, /numpyCode/);
  assert.match(source, /flopscopeCode/);

  assert.match(source, /\/docs\/guides\/migrate-from-numpy/);
  assert.match(source, /\/docs\/understanding\/flop-counting-model/);
  assert.match(source, /\/docs\/api\/for-agents/);

  assert.doesNotMatch(source, /\/docs\/how-to\/migrate-from-numpy/);
  assert.doesNotMatch(source, /\/docs\/concepts\/flop-counting-model/);
  assert.doesNotMatch(source, /\/docs\/reference\/for-agents/);
  assert.doesNotMatch(source, /Runtime overhead/);
});

test('global CSS keeps Fumadocs defaults and only scopes the homepage seam fix locally', async () => {
  const source = await readSource('app/global.css');

  assert.match(source, /#nd-home-layout\s+#nd-nav\s*>\s*div\s*>\s*div:last-child/);
  assert.doesNotMatch(source, /main\s*\{\s*padding-top:\s*1rem;?\s*\}/);
  assert.doesNotMatch(source, /\.prose\s+code:not\(pre code\)/);
});

test('calibration page is retired and absent from all docs nav', async () => {
  // The standalone calibration page was removed; its content was consolidated
  // into docs/reference/empirical-weights.md (linked from the cost-model page).
  // Guard that neither location serves it and neither nav lists it.
  const understandingCalibration = path.join(
    websiteRoot,
    'content',
    'docs',
    'understanding',
    'calibration.mdx',
  );
  const developmentCalibration = path.join(
    websiteRoot,
    'content',
    'docs',
    'development',
    'calibration.mdx',
  );
  const understandingMeta = await readSource('content/docs/understanding/meta.json');
  const developmentMeta = await readSource('content/docs/development/meta.json');

  await assert.rejects(access(understandingCalibration));
  await assert.rejects(access(developmentCalibration));

  assert.doesNotMatch(understandingMeta, /"calibration"/);
  assert.doesNotMatch(developmentMeta, /"calibration"/);
});

test('machine-readable links use plain anchors instead of framework navigation', async () => {
  const agentsSource = await readSource('content/docs/api/for-agents.mdx');
  const opPageSource = await readSource('components/api-reference/OperationDocPage.tsx');

  assert.match(agentsSource, /<StaticFileLink href="\/llms\.txt">llms\.txt<\/StaticFileLink>/);
  assert.match(agentsSource, /<StaticFileLink href="\/llms-full\.txt">llms-full\.txt<\/StaticFileLink>/);
  assert.match(agentsSource, /<StaticFileLink href="\/ops\.json">ops\.json<\/StaticFileLink>/);
  assert.match(agentsSource, /\[Exploit Symmetry]\(\/docs\/guides\/symmetry\/\)/);
  assert.doesNotMatch(agentsSource, /\.\.\/\.\.\/\.\.\/llms\.txt/);
  assert.doesNotMatch(agentsSource, /\(\/?\.\//);

  assert.doesNotMatch(opPageSource, /detail_json_href/);
  assert.doesNotMatch(opPageSource, /json payload/i);
});
