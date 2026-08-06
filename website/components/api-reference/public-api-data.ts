import publicApiIndex from '../../public/api-data/public-api/index.json';

export type PublicApiNamespace = 'flopscope' | 'numpy' | 'stats' | 'accounting';

export interface PublicApiIndexEntry {
  href: string;
  import_path: string;
  display_name: string;
  summary: string;
  kind: string;
  namespace: PublicApiNamespace;
  module: string;
  callable: boolean;
  slug: string;
  is_operation_cost_leaf: boolean;
}

export interface NamespaceCard {
  namespace: PublicApiNamespace;
  title: string;
  importPath: string;
  href: string;
  summary: string;
  count: number;
  highlights: PublicApiIndexEntry[];
}

export interface NamespaceInventoryRow extends PublicApiIndexEntry {
  members?: Array<{ href: string; label: string }>;
}

export interface NamespaceInventorySection {
  title: string;
  description: string;
  entries: NamespaceInventoryRow[];
}

const payload = publicApiIndex as {
  entries: PublicApiIndexEntry[];
  operation_cost_index: PublicApiIndexEntry[];
};

const entries = [...payload.entries].sort((a, b) =>
  a.import_path.localeCompare(b.import_path),
);

const namespaceMeta: Record<PublicApiNamespace, Omit<NamespaceCard, 'count' | 'highlights'>> = {
  flopscope: {
    namespace: 'flopscope',
    title: 'Flopscope primitives',
    importPath: 'flopscope',
    href: '/docs/api/flopscope/',
    summary:
      'Budgets, symmetry helpers, public objects, and configuration primitives that sit outside the counted NumPy surface.',
  },
  numpy: {
    namespace: 'numpy',
    title: 'NumPy array routines',
    importPath: 'flopscope.numpy',
    href: '/docs/api/numpy/',
    summary:
      'The counted NumPy-shaped surface, including array construction, linear algebra, FFT, and random sampling.',
  },
  stats: {
    namespace: 'stats',
    title: 'Statistics',
    importPath: 'flopscope.stats',
    href: '/docs/api/stats/',
    summary:
      'Distribution objects and their methods for PDFs, CDFs, and inverse CDFs.',
  },
  accounting: {
    namespace: 'accounting',
    title: 'Accounting',
    importPath: 'flopscope.accounting',
    href: '/docs/api/accounting/',
    summary:
      'Analytical FLOP estimators and planning helpers for reasoning about cost before execution.',
  },
};

const featuredImportPaths = [
  'flopscope.numpy.einsum',
  'flopscope.BudgetContext',
  'flopscope.symmetrize',
  'flopscope.stats.norm',
  'flopscope.accounting.einsum_cost',
  'flopscope.numpy.random.symmetric',
];

function entryByImportPath(importPath: string): PublicApiIndexEntry | undefined {
  return entries.find((entry) => entry.import_path === importPath);
}

function membersFor(parentImportPath: string): Array<{ href: string; label: string }> {
  const prefix = `${parentImportPath}.`;
  return entries
    .filter((entry) => entry.import_path.startsWith(prefix))
    .map((entry) => ({
      href: entry.href,
      label: entry.import_path.slice(prefix.length),
    }))
    .sort((a, b) => a.label.localeCompare(b.label));
}

function topLevelFlopscopeEntries(names: string[]): NamespaceInventoryRow[] {
  return names
    .map((name) => entryByImportPath(`flopscope.${name}`))
    .filter((entry): entry is PublicApiIndexEntry => Boolean(entry))
    .map((entry) => ({...entry}));
}

function numpyEntriesForPrefix(prefix: string): NamespaceInventoryRow[] {
  return entries
    .filter((entry) => entry.import_path.startsWith(prefix))
    .map((entry) => ({...entry}));
}

function statsObjectEntries(): NamespaceInventoryRow[] {
  return entries
    .filter((entry) => {
      if (entry.namespace !== 'stats') return false;
      return entry.import_path.split('.').length === 3;
    })
    .map((entry) => ({
      ...entry,
      members: membersFor(entry.import_path),
    }));
}

export function getNamespaceCards(): NamespaceCard[] {
  return (Object.keys(namespaceMeta) as PublicApiNamespace[]).map((namespace) => {
    const namespaceEntries = entries.filter((entry) => entry.namespace === namespace);
    const highlights = namespaceEntries.slice(0, 3);
    return {
      ...namespaceMeta[namespace],
      count: namespaceEntries.length,
      highlights,
    };
  });
}

export function getFeaturedEntries(): PublicApiIndexEntry[] {
  return featuredImportPaths
    .map((importPath) => entryByImportPath(importPath))
    .filter((entry): entry is PublicApiIndexEntry => Boolean(entry));
}

export function getNamespaceInventory(namespace: PublicApiNamespace): NamespaceInventorySection[] {
  if (namespace === 'flopscope') {
    return [
      {
        title: 'Budgets & configuration',
        description:
          'Control runtime budgets, namespace accounting, and process-wide configuration.',
        entries: topLevelFlopscopeEntries([
          'BudgetContext',
          'budget',
          'budget_live',
          'budget_reset',
          'current_budget',
          'budget_summary',
          'budget_summary_dict',
          'configure',
          'namespace',
        ]),
      },
      {
        title: 'Symmetry & permutations',
        description:
          'Represent and preserve symmetry structure explicitly across counted workflows.',
        entries: topLevelFlopscopeEntries([
          'Permutation',
          'PermutationGroup',
          'Cycle',
          'as_symmetric',
          'is_symmetric',
          'symmetrize',
        ]),
      },
      {
        title: 'Objects & types',
        description:
          'Public result objects and tensor wrappers exposed by Flopscope itself.',
        entries: topLevelFlopscopeEntries([
          'FlopscopeArray',
          'BudgetSnapshot',
          'PathInfo',
          'StepInfo',
          'SymmetricTensor',
          'SymmetryInfo',
        ]),
      },
    ];
  }

  if (namespace === 'numpy') {
    return [
      {
        title: 'Core array routines',
        description:
          'Array construction, indexing, reductions, einsum, and other counted NumPy-style entry points.',
        entries: entries.filter((entry) => {
          if (entry.namespace !== 'numpy') return false;
          return !entry.import_path.startsWith('flopscope.numpy.linalg.') &&
            !entry.import_path.startsWith('flopscope.numpy.fft.') &&
            !entry.import_path.startsWith('flopscope.numpy.random.');
        }),
      },
      {
        title: 'Linear algebra',
        description: 'Counted linear algebra routines under `flopscope.numpy.linalg`.',
        entries: numpyEntriesForPrefix('flopscope.numpy.linalg.'),
      },
      {
        title: 'Fourier transforms',
        description: 'Counted FFT routines under `flopscope.numpy.fft`.',
        entries: numpyEntriesForPrefix('flopscope.numpy.fft.'),
      },
      {
        title: 'Random sampling',
        description: 'Counted random sampling routines, including symmetry-aware sampling helpers.',
        entries: numpyEntriesForPrefix('flopscope.numpy.random.'),
      },
    ];
  }

  if (namespace === 'stats') {
    return [
      {
        title: 'Distributions',
        description:
          'Distribution objects expose `pdf`, `cdf`, and `ppf` methods on their own reference pages.',
        entries: statsObjectEntries(),
      },
    ];
  }

  return [
    {
      title: 'Cost estimators',
      description:
        'Analytical helpers for estimating FLOPs without executing a counted operation.',
      entries: entries.filter((entry) => entry.namespace === 'accounting'),
    },
  ];
}

export function getOperationCostIndexEntries(): PublicApiIndexEntry[] {
  return [...payload.operation_cost_index].sort((a, b) =>
    a.import_path.localeCompare(b.import_path),
  );
}
