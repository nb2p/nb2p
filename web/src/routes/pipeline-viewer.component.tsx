import { Card, CardContent } from '@/components/ui/card'
import { useBoundStore } from '@/stores'
import { pythonLanguage } from '@codemirror/lang-python'
import { Navigate } from '@tanstack/react-router'
import { githubLight } from '@uiw/codemirror-theme-github'
import CodeMirror, { EditorView } from '@uiw/react-codemirror'
import DagreGraph from 'dagre-d3-react'
import { useCallback, useState } from 'react'
import '../d3.css'
import { toast } from 'sonner'

const themeDemo = EditorView.baseTheme({
  '&': { backgroundColor: '#fcfcfc !important' },
  '&dark .zebra-colored': { backgroundColor: '#aca2ff40' },
  '&light .zebra-colored': { backgroundColor: '#aca2ff33' },
  '::-webkit-scrollbar': { display: 'none' },
  '.cm-gutterElement': { width: '24px !important' }
})

declare type labelType = 'html' | 'svg' | 'string'
declare interface d3Node {
  id: string
  label: string
  class?: string
  labelType?: labelType
  config?: object
}
declare interface d3Link {
  source: string
  target: string
  class?: string
  label?: string
  config?: object
}

interface PipelineData {
  nodes: d3Node[]
  links: d3Link[]
}

export const component = function PipelineViewer () {
  const data = useBoundStore((state) => state.pipelineGraph)
  const segments = useBoundStore((state) => state.segments)
  const setCurrentNav = useBoundStore((state) => state.setCurrentNav)
  const [segmentCode, setSegmentCode] = useState<string[]>([])
  const [segmentID, setSegmentID] = useState<string>('')

  const selectSegment = useCallback((segmentID: string) => {
    setSegmentCode(segments[Number.parseInt(segmentID)])
    setSegmentID(segmentID)
  }, [])

  if (data === null || data === undefined || Object.keys(data).length === 0 || Object.keys(data.nodes).length === 0) {
    toast.error('Notebook not segmented. Please do segmentation first.')

    setCurrentNav('Segmentation')
    return (<Navigate to='/segmentation' />)
  }

  const ErrorElement = <div>ERROR: RESULT CANNOT BE PARSED</div>

  if (typeof data !== 'object' || data === null) {
    return ErrorElement
  }
  if (!('nodes' in data) || !('links' in data)) {
    return ErrorElement
  }

  const checkedData: PipelineData = data as PipelineData

  return (
    <div id='dag' className='grid grid-rows-2 size-full p-8'>
      <DagreGraph
        className='size-full'
        nodes={checkedData.nodes}
        links={checkedData.links}
        config={{
          rankdir: 'LR',
          align: 'UL',
          ranker: 'tight-tree'
        }}
        animate={500}
        shape='circle'
        zoomable
        onNodeClick={(e) => { selectSegment(e.original.label) }}
        onRelationshipClick={(e) => { console.log(e) }}
      />
      <div className='flex space-x-2 size-full overflow-x-hidden'>
        <div className='flex-none w-32'>
          <p className='text-bold'>Code of</p>
          <p className='text-bold'>Component {segmentID}</p>
        </div>
        <Card className='grow no-scrollbar rounded-none overflow-y-auto'>
          <CardContent className='p-0'>
            <CodeMirror
              value={segmentCode.join('\n')}
              theme={githubLight}
              basicSetup={{
                highlightActiveLine: false,
                highlightActiveLineGutter: false,
                foldGutter: false
                // lineNumbers: c.join('\n')[0] !== '#'
              }} extensions={[themeDemo, pythonLanguage]}
            />
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
