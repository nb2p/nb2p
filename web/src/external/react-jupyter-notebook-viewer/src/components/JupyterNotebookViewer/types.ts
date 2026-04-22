export interface JupyterNotebookViewerType {
  filePath: string
  className?: string
  notebookInputLanguage?: string
  notebookOutputLanguage?: string
  inputCodeDarkTheme?: boolean
  outputDarkTheme?: boolean
  showInputLineNumbers?: boolean
  showOutputLineNumbers?: boolean
  inputMarkdownDarkTheme?: boolean
  outputTextClassName?: string
  inputTextClassName?: string
  outputBlockClassName?: string
  outputImageClassName?: string
  outputOuterClassName?: string
  inputOuterClassName?: string
  outputBorderClassName?: string
  inputBorderClassName?: string
  outputTableClassName?: string
  withOnClick?: boolean
  inputMarkdownBlockClassName?: string
  inputCodeBlockClassName?: string
  hideCodeBlocks?: boolean
  hideMarkdownBlocks?: boolean
  hideAllOutputs?: boolean
  hideAllInputs?: boolean
  remarkPlugins?: any
  rehypePlugins?: any
}

export interface NotebookTransformed {
  type: string
  executionCount: number
  source: string
  outputs?: NotebookOutputTransformed[]
}

export interface NotebookOutputTransformed {
  outputType: string
  data?: Data | string[]
}

export interface NotebookPartType {
  type: string
  executionCount: number | undefined
  source: string
  outputs?: NotebookOutputTransformed[]
}

export interface Notebook {
  cells: Cell[]
  metadata: NotebookMetadata
  nbformat: number
  nbformat_minor: number
}

export interface Cell {
  cell_type: CellType
  metadata: CellMetadata
  source: string[]
  execution_count?: number
  outputs?: Output[]
}

export enum CellType {
  Code = 'code',
  Markdown = 'markdown',
}

export interface CellMetadata {}

export interface Output {
  data?: Data
  execution_count?: number
  metadata?: OutputMetadata
  output_type: OutputType
  name?: string
  text?: string[]
  traceback?: string[]
}

export interface Data {
  'text/html'?: string[]
  'text/plain': string[]
  'image/png'?: string
}

export interface OutputMetadata {
  needs_background?: NeedsBackground
}

export enum NeedsBackground {
  Light = 'light',
}

export enum OutputType {
  DisplayData = 'display_data',
  ExecuteResult = 'execute_result',
  Stream = 'stream',
  Error = 'error',
}

export interface NotebookMetadata {
  interpreter: Interpreter
  kernelspec: Kernelspec
  language_info: LanguageInfo
  orig_nbformat: number
}

export interface Interpreter {
  hash: string
}

export interface Kernelspec {
  display_name: string
  name: string
}

export interface LanguageInfo {
  codemirror_mode: CodemirrorMode
  file_extension: string
  mimetype: string
  name: string
  nbconvert_exporter: string
  pygments_lexer: string
  version: string
}

export interface CodemirrorMode {
  name: string
  version: number
}
