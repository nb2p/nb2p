import { type NotebookPartType } from '@/external/react-jupyter-notebook-viewer/src/components/JupyterNotebookViewer/types'
import { type StateCreator } from 'zustand'

export type labelType = 'html' | 'svg' | 'string'
export interface d3Node {
  id: string
  label: string
  class?: string
  labelType?: labelType
  config?: object
}
export interface d3Link {
  source: string
  target: string
  class?: string
  label?: string
  config?: object
}

export interface PipelineGraph {
  nodes: d3Node[]
  links: d3Link[]
  err?: string
}

export interface NotebookSlice {
  notebookLoadingProgress: number
  notebookData: string
  pipelineGraph: PipelineGraph
  cellInputs: NotebookPartType[]
  segments: string[][]
  sampleNotebookPaths: Record<string, string>
  selectedNotebookPath: string

  setNotebookLoadingProgress: (progress: number) => void
  setPipelineGraph: (graph: PipelineGraph) => void
  setCellInputs: (cells: NotebookPartType[]) => void
  setSegments: (segments: string[][]) => void
  selectNotebook: (id: string) => void
}

export const SAMPLE_NOTEBOOK_PATHS = {
  'sample-1': 'nb_test.ipynb',
  'sample-2': 'nb_test2.ipynb',
  'sample-3': 'nb_test3.ipynb'
}

export const createNotebookSlice: StateCreator<NotebookSlice> = (set, get) => ({
  notebookLoadingProgress: 100,
  notebookData: '',
  pipelineGraph: { nodes: [], links: [] },
  cellInputs: [],
  segments: [],
  sampleNotebookPaths: SAMPLE_NOTEBOOK_PATHS,
  selectedNotebookPath: '',

  setNotebookLoadingProgress: (value: number) => {
    set({ notebookLoadingProgress: value })
  },
  setNotebookData: (value: string) => {
    set({ notebookData: value })
  },
  setPipelineGraph: (value: PipelineGraph) => {
    set({ pipelineGraph: value })
  },
  setCellInputs (value: NotebookPartType[]) {
    // cells.map((c) => {
    //   c.source = c.source
    //     .split('\n')
    //     .filter((l) => !(l.startsWith('#')) && !(l.trim().length === 0)).join('\n')
    // })
    set({ cellInputs: value })
  },
  setSegments (value: string[][]) {
    set({ segments: value })
  },
  selectNotebook (id: string) {
    set({ selectedNotebookPath: get().sampleNotebookPaths[id] })
  }
})
