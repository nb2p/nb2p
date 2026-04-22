import { type StateCreator } from 'zustand'

export interface NavSlice {
  currentNav: string
  setCurrentNav: (nav: string) => void
}

export const createNavSlice: StateCreator<NavSlice> = (set) => ({
  currentNav: '',

  setCurrentNav: (nav: string) => { set({ currentNav: nav }) }
})
