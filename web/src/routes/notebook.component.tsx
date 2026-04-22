import { JupyterNotebookViewer } from '@/external/react-jupyter-notebook-viewer/src/index'
import { useBoundStore } from '@/stores'
import { Navigate } from '@tanstack/react-router'
import { toast } from 'sonner'

export const component = function Index () {
  const path = useBoundStore(state => state.selectedNotebookPath)
  const selectedNotebookPath = useBoundStore(state => state.selectedNotebookPath)
  const setCurrentNav = useBoundStore((state) => state.setCurrentNav)

  if (!path) {
    toast.error('Please select a notebook.')

    setCurrentNav('')
    return (<Navigate to='/' />)
  }

  return (
    <div className='overflow-y-auto h-[calc(100vh-36px)]'>
      <JupyterNotebookViewer
        className='w-full'
        filePath={`/${selectedNotebookPath}`}
        notebookInputLanguage='python'
        inputMarkdownDarkTheme
        showInputLineNumbers
      />
    </div>
  )
}
