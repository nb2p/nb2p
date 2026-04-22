module.exports = {
  root: true,
  env: { browser: true, es2020: true },
  extends: [
    'standard',
    'standard-jsx',
  ],
  ignorePatterns: ['dist', '.eslintrc.cjs', 'src/routeTree.gen.ts'],
  parser: '@typescript-eslint/parser',
  plugins: ['react-refresh'],
  rules: {
    'react-refresh/only-export-components': [
      'warn',
      { allowConstantExport: true },
    ],
    'import/no-absolute-path': 'off'
  },
  parserOptions: {
    parser: '@typescript-eslint/parser',
    project: './tsconfig.json',
    tsconfigRootDir: __dirname,
  },
}
