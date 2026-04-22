import { type StateCreator } from 'zustand'

export interface BackendSlice {
  backendURL: string
  backendStatus: string
  backendStatusColor: string

  setBackendURL: (url: string) => void
  connectToBackend: () => Promise<void>

  _updateBackendStatus: (status: string) => void
}

const STATUS_COLOR_MAP: Record<string, string> = {
  Connected: 'green',
  Disconnected: 'red',
  'Connecting...': 'orange'
}

export const createBackendSlice: StateCreator<BackendSlice> = (set, get) => ({
  backendURL: 'http://10.1.2.3:8000/nb2p',
  backendStatus: 'Disconnected',
  backendStatusColor: 'red',

  setBackendURL: (url) => { set({ backendURL: url }) },
  connectToBackend: async () => {
    try {
      get()._updateBackendStatus('Connecting...')
      const urlPath = new URL(get().backendURL + '/ping')
      const res = await (await fetch(urlPath)).json()
      if (res.msg === 'pong') {
        get()._updateBackendStatus('Connected')
      }
    } catch (e) {
      get()._updateBackendStatus('Disconnected')
    }
  },

  _updateBackendStatus: (status) => {
    set({ backendStatus: status, backendStatusColor: STATUS_COLOR_MAP[status] })
  }
})
