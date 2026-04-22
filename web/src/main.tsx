import React from 'react'
import ReactDOM from 'react-dom/client'
import { Router, RouterProvider } from '@tanstack/react-router'
import './index.css'

// Import the generated route tree
import { routeTree } from './routeTree.gen'
import { useBoundStore } from '@/stores'

// Create a new router instance
const router = new Router({ routeTree })

// Register the router instance for type safety
declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}

// Render the app
const rootElement = document.getElementById('root')
if (rootElement == null) {
  throw new Error('THIS SHOULD NOT HAPPEN')
}
ReactDOM.createRoot(rootElement).render(
  // <React.StrictMode>
  <RouterProvider router={router} />
  // </React.StrictMode>
)

useBoundStore.getState().connectToBackend()
