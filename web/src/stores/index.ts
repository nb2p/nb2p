import { create } from 'zustand'
import { type BackendSlice, createBackendSlice } from './backend'
import { type NavSlice, createNavSlice } from './nav'
import { type NotebookSlice, createNotebookSlice } from './notebook'
import { type WorkflowSlice, createWorkflowSlice } from './workflow'

type BoundStore = BackendSlice & NavSlice & NotebookSlice & WorkflowSlice

export const useBoundStore = create<BoundStore>()((...a) => ({
  ...createBackendSlice(...a),
  ...createNavSlice(...a),
  ...createNotebookSlice(...a),
  ...createWorkflowSlice(...a)
}))
