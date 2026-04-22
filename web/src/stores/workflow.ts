import { type StateCreator } from 'zustand'

export interface WorkflowSlice {
  STModel: string
  setSTModel: (value: string) => void
}

export const createWorkflowSlice: StateCreator<WorkflowSlice> = (set) => ({
  STModel: 'NB2P-SS',

  setSTModel: (value: string) => { set({ STModel: value }) }
})
