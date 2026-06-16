#!/usr/bin/env node
/**
 * Post-build script: generates llms.txt and llms-full.txt from Next.js static export.
 *
 * Reads HTML files from out/, extracts text content, and produces:
 *   out/llms.txt       - index with section headers and links
 *   out/llms-full.txt  - all page content concatenated
 *   public/llms.txt    - dev/runtime copy served by Next.js
 *   public/llms-full.txt - dev/runtime copy served by Next.js
 *
 * Run: node scripts/generate-llmstxt.mjs
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { JSDOM } from 'jsdom';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT_DIR = path.resolve(__dirname, '..', 'out');
const PUBLIC_DIR = path.resolve(__dirname, '..', 'public');
const SITE_URL = 'https://aicrowd.github.io';
const BASE_PATH = '/flopscope';

/**
 * Section configuration mirroring the Docusaurus llmstxt plugin.
 * Each key is a section header; value is {slug: description}.
 */
const SECTIONS = {
  'Getting Started': {
    'getting-started/installation': 'Install flopscope and verify setup',
    'getting-started/quickstart': 'Run your first FLOP-counted computation',
    'getting-started/competition': 'Everything you need to compete within a FLOP budget',
  },
  'Guides': {
    'guides/migrate-from-numpy': 'Migrate existing NumPy code to flopscope',
    'guides/einsum': 'Einsum patterns, costs, and optimization',
    'guides/symmetry': 'Reduce FLOP costs with symmetric tensors',
    'guides/linalg': 'Linear algebra operations and costs',
    'guides/fft': 'FFT operations, real vs complex, and costs',
    'guides/budget-planning': 'Estimate costs and diagnose budget overruns',
  },
  'Understanding flopscope': {
    'understanding/how-flopscope-works': 'How Flopscope wraps NumPy to count every FLOP',
    'understanding/flop-counting-model': 'Analytical FLOP cost formulas and conventions',
    'understanding/operation-categories': 'Free vs counted vs blocked operations',
    'understanding/symmetry-detection': 'Subgraph symmetry detection algorithm deep dive',
  },
  'API Reference': {
    'api': 'Interactive searchable reference for all 508 operations',
    'api/for-agents': 'Machine-readable resources and rules for AI coding assistants',
  },
  'Infrastructure': {
    'infrastructure/client-server': 'Client-server architecture for competition evaluation',
    'infrastructure/docker': 'Running flopscope with Docker',
  },
  'Development': {
    'development/contributing': 'Repository layout, local workflows, and test suite',
  },
  'Changelog': {
    'changelog': 'Release history and breaking changes',
  },
};

/**
 * Extract plain text from an HTML file, stripping nav/script/style/header/footer.
 */
function extractText(htmlPath) {
  const html = fs.readFileSync(htmlPath, 'utf-8');
  const dom = new JSDOM(html);
  const doc = dom.window.document;

  // Remove elements that are not content
  const removeSelectors = [
    'script',
    'style',
    'nav',
    'header',
    'footer',
    // Docusaurus selectors (kept for compatibility)
    '.navbar',
    '.footer',
    '.pagination-nav',
    '.table-of-contents',
    '.theme-doc-toc-mobile',
    '.theme-doc-toc-desktop',
    '.theme-doc-breadcrumbs',
    '.theme-doc-sidebar-container',
    // Fumadocs selectors
    '[data-sidebar]',
    '[data-toc]',
    '.fd-toc',
    '.fd-sidebar',
    '.fd-nav',
    '[role="navigation"]',
  ];

  for (const sel of removeSelectors) {
    for (const el of doc.querySelectorAll(sel)) {
      el.remove();
    }
  }

  // Try to get the main article content first
  const article =
    doc.querySelector('article') ||
    doc.querySelector('.markdown') ||
    doc.querySelector('main') ||
    doc.body;

  const text = (article.textContent || '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

  return text;
}

/**
 * Resolve a doc slug to its HTML file path in the build directory.
 * Fumadocs puts doc pages under docs/ prefix, so we check both
 * <outDir>/docs/<slug>/index.html and <outDir>/<slug>/index.html.
 */
function resolveHtmlPath(slug) {
  // Fumadocs routes: try docs/<slug>/index.html first
  const docsIndexPath = path.join(OUT_DIR, 'docs', slug, 'index.html');
  if (fs.existsSync(docsIndexPath)) {
    return docsIndexPath;
  }
  // Try <outDir>/<slug>/index.html
  const indexPath = path.join(OUT_DIR, slug, 'index.html');
  if (fs.existsSync(indexPath)) {
    return indexPath;
  }
  // Try <outDir>/<slug>.html
  const directPath = path.join(OUT_DIR, slug + '.html');
  if (fs.existsSync(directPath)) {
    return directPath;
  }
  return null;
}

// --- Main ---

const siteBase = `${SITE_URL}${BASE_PATH}/`;

const llmsTxtLines = [];
const llmsFullLines = [];

// Header
llmsTxtLines.push('# flopscope');
llmsTxtLines.push('');
llmsTxtLines.push('> NumPy-compatible math primitives with FLOP counting');
llmsTxtLines.push('');

let found = 0;
let missing = 0;

for (const [section, pages] of Object.entries(SECTIONS)) {
  llmsTxtLines.push(`## ${section}`);
  llmsTxtLines.push('');

  llmsFullLines.push('='.repeat(60));
  llmsFullLines.push(section);
  llmsFullLines.push('='.repeat(60));
  llmsFullLines.push('');

  for (const [slug, description] of Object.entries(pages)) {
    // Determine the actual URL based on where the HTML was found
    const htmlPath = resolveHtmlPath(slug);
    const isDocsRoute = htmlPath && htmlPath.includes(path.join('out', 'docs'));
    const urlSlug = isDocsRoute ? `docs/${slug}` : slug;
    const pageUrl = `${siteBase}${urlSlug}`;
    llmsTxtLines.push(`- [${description}](${pageUrl})`);

    if (htmlPath) {
      const text = extractText(htmlPath);
      llmsFullLines.push(`--- ${slug} ---`);
      llmsFullLines.push(`URL: ${pageUrl}`);
      llmsFullLines.push('');
      llmsFullLines.push(text);
      llmsFullLines.push('');
      found++;
    } else {
      llmsFullLines.push(`--- ${slug} ---`);
      llmsFullLines.push(`URL: ${pageUrl}`);
      llmsFullLines.push('');
      llmsFullLines.push('[Page not found in build output]');
      llmsFullLines.push('');
      missing++;
    }
  }

  llmsTxtLines.push('');
}

if (missing > 0) {
  throw new Error(`[generate-llmstxt] ${missing} declared docs page(s) were missing from the build output`);
}

// Write output
fs.mkdirSync(OUT_DIR, { recursive: true });
fs.mkdirSync(PUBLIC_DIR, { recursive: true });

const llmsTxtPath = path.join(OUT_DIR, 'llms.txt');
const llmsFullPath = path.join(OUT_DIR, 'llms-full.txt');
const publicLlmsTxtPath = path.join(PUBLIC_DIR, 'llms.txt');
const publicLlmsFullPath = path.join(PUBLIC_DIR, 'llms-full.txt');
const noJekyllPath = path.join(OUT_DIR, '.nojekyll');

fs.writeFileSync(llmsTxtPath, llmsTxtLines.join('\n'), 'utf-8');
fs.writeFileSync(llmsFullPath, llmsFullLines.join('\n'), 'utf-8');
fs.writeFileSync(publicLlmsTxtPath, llmsTxtLines.join('\n'), 'utf-8');
fs.writeFileSync(publicLlmsFullPath, llmsFullLines.join('\n'), 'utf-8');
fs.writeFileSync(noJekyllPath, '', 'utf-8');

console.log(`[generate-llmstxt] ${llmsTxtPath} (${found} pages found, ${missing} missing)`);
console.log(`[generate-llmstxt] ${llmsFullPath}`);
console.log(`[generate-llmstxt] ${publicLlmsTxtPath}`);
console.log(`[generate-llmstxt] ${publicLlmsFullPath}`);
console.log(`[generate-llmstxt] ${noJekyllPath}`);
