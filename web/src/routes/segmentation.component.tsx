import { useCallback, useEffect, useMemo, useState } from 'react'
import CodeMirror from '@uiw/react-codemirror'
import { pythonLanguage } from '@codemirror/lang-python'
import { githubLight } from '@uiw/codemirror-theme-github'
import { EditorView } from '@codemirror/view'
import { Card, CardContent } from '@/components/ui/card'
import { Button, ButtonLoading } from '@/components/ui/button'
import { useBoundStore } from '@/stores'
import { Separator } from '@/components/ui/separator'
import { type PipelineGraph } from '@/stores/notebook'
import { SelectMethod } from '@/components/select-method'
import { ScrollArea } from '@/components/ui/scroll-area'
import { toast } from 'sonner'
import { Navigate } from '@tanstack/react-router'

// const recommendedNotebooks = [
//   '/data0/mlcask/scs/kgtorrent/ipynb/aakarkale_pneumonia-detection-using-convolution-neural-nets.ipynb',
//   '/data0/mlcask/scs/kgtorrent/ipynb/araraonline_locating-schools-with-a-shortage-of-applicants.ipynb',
//   '/data0/mlcask/scs/kgtorrent/ipynb/agailloty_understand-the-wealth-of-nations.ipynb'
// ]

function segmentEndsToCodeLineNumbers (segmentEnds: number[], totalLines: number): number[][] {
  const result: number[][] = []

  for (const se of segmentEnds) {
    result.push([result.length > 0 ? result[result.length - 1][1] : 0, se + 1])
  }

  if (result.length > 0 && result[segmentEnds.length - 1][1] < (totalLines - 1)) {
    result.push([result[segmentEnds.length - 1][1], totalLines])
  }

  return result
}

export const component = function Segmentation () {
  const [isCodeModified, setCodeModified] = useState<boolean>(false)
  const [isSegmenting, setSegmenting] = useState<boolean>(false)
  const [isFetching, setFetching] = useState<boolean>(false)
  const [segmentEnds, setSegmentEnds] = useState<number[]>([])

  const cellInputs = useBoundStore(state => state.cellInputs)
  const setCellInputs = useBoundStore(state => state.setCellInputs)
  const segments = useBoundStore(state => state.segments)
  const setSegments = useBoundStore(state => state.setSegments)
  const setPipelineGraph = useBoundStore(state => state.setPipelineGraph)
  const backendURL = useBoundStore(state => state.backendURL)
  const setCurrentNav = useBoundStore(state => state.setCurrentNav)
  const stmodel = useBoundStore(state => state.STModel)
  const setSTModel = useBoundStore(state => state.setSTModel)

  const path = useBoundStore(state => state.selectedNotebookPath)

  if (!path) {
    toast.error('Please select a notebook.')

    setCurrentNav('')
    return (<Navigate to='/' />)
  }

  const cells = useMemo(() => {
    console.log(cellInputs.filter((c) => c.type === 'code'))
    return cellInputs.filter((c) => c.type === 'code').map((c) => c.source.split('\n'))
  }, [cellInputs])

  const fetchNotebookCells = useCallback(() => {
    (async () => {
      setFetching(true)

      const res = await fetch(backendURL + '/notebook/cells/', {
        method: 'POST',
        body: new URLSearchParams({
          path
        })
      })

      const body: { path: string, cells: string[][], err?: string } = await res.json()
      if (body.err != null) {
        return
      }

      const fetchedCells = body.cells
      console.log(`Cells for notebook ${path}: ${fetchedCells.toString()}`)
      setCodeModified(false)
      setFetching(false)

      const inputs = body.cells.map((c) => { return { type: 'code', executionCount: undefined, source: c.join('\n') } })
      console.log(inputs)

      setCellInputs([...inputs])
    })()
  }, [])

  const themeDemo = EditorView.baseTheme({
    // '&': { backgroundColor: '#fcfcfc !important' },
    // '&dark .zebra-colored': { backgroundColor: '#aca2ff40' },
    // '&light .zebra-colored': { backgroundColor: '#aca2ff33' },
    // '::-webkit-scrollbar': { display: 'none' },
    // '.cm-gutterElement': { width: '24px !important' }
  })

  const fetchSegmentEndsForCustomCode = () => {
    (async () => {
      setSegmenting(true)

      const res = await fetch(backendURL + '/notebook/segment-ends-v2/', {
        method: 'POST',
        body: new URLSearchParams({
          code: cells.map((c) => c.join('\n')).join('\n'),
          stmodel
        })
      })
      setSegmenting(false)

      const body: {
        path: string
        segmentEnds: number[]
        err?: string
        pipeline: PipelineGraph
      } = await res.json()

      if (body.err != null) {
        toast.error(body.err)
        return
      }

      const fetchedSegmentEnds = body.segmentEnds
      console.log(`Segment ends for notebook ${path}: ${fetchedSegmentEnds.toString()}`)
      setSegmentEnds(fetchedSegmentEnds)
      if (body.pipeline != null) {
        setPipelineGraph(body.pipeline)
      }
    })()

    return fetchSegmentEndsForCustomCode
  }

  useMemo(() => {
    if (segmentEnds.length === 0) {
      return
    }
    if (cells.length === 0) {
      return
    }

    const cellStrings = cells.flat()
    const newCodeSegments = []
    const zebraStrips = segmentEndsToCodeLineNumbers(segmentEnds, cellStrings.length)
    console.log('zebraStrips: ', zebraStrips)
    for (const zs of zebraStrips) {
      newCodeSegments.push(cellStrings.slice(zs[0], zs[1]))
    }
    setSegments(newCodeSegments)
  }, [cellInputs, segmentEnds])

  const onCodeChange = (value: string, idx: number) => {
    setCodeModified(true)
    console.log(idx, value)
    const newCells = structuredClone(cells)
    newCells[idx] = value.split('\n')
    console.log('Cells modified. New cells:')
    for (const c of newCells) {
      console.log(c)
    }
  }

  useEffect(() => {
    fetchNotebookCells()
  }, [fetchNotebookCells])

  return (
    <div className='overflow-y-auto h-[calc(100vh-36px)] p-4'>
      <div className='flex items-center space-x-2 mb-4'>
        {
          !isFetching && (
            <Button variant='secondary' onClick={fetchNotebookCells}>Refresh Notebook</Button>
          )
        }{
          isFetching && (
            <ButtonLoading>Refreshing...</ButtonLoading>
          )
        }
        <Separator orientation='vertical' />
        {
          !isSegmenting && (
            <Button onClick={fetchSegmentEndsForCustomCode}>Segment</Button>
          )
        }{
          isSegmenting && (
            <ButtonLoading>Segmenting...</ButtonLoading>
          )
        }
        <SelectMethod value={stmodel} onValueChange={setSTModel} />
      </div>

      <div className='grid grid-cols-2 gap-4'>
        <h2 className='text-large font-large'>Notebook Code{isCodeModified ? ' (Modified)' : ''}</h2>
        <h2 className='text-large font-large'>Notebook Segments</h2>
        <Card>
          <CardContent className='p-4'>
            <div className='w-full'>
              {cells.map((c, i) => (
                <div className='flex my-1 w-full' key={`c-${i}`}>
                  <div className='flex-none w-8 text-[#aaa]'>[{i + 1}]</div>
                  <Card className='flex-1 rounded-none overflow-x-hidden'>
                    <CardContent className='p-0'>
                      <ScrollArea>
                        <CodeMirror
                          value={c.join('\n')}
                          theme={githubLight}
                          basicSetup={{
                            highlightActiveLine: false,
                            highlightActiveLineGutter: false,
                            foldGutter: false
                            // lineNumbers: c.join('\n')[0] !== '#'
                          }} extensions={[
                            // themeDemo,
                            pythonLanguage]} onChange={(v) => { onCodeChange(v, i) }}
                        />
                      </ScrollArea>
                    </CardContent>
                  </Card>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className='p-4'>
            <div className='w-full'>
              {segments.map((c, i) => (
                <div className='flex my-1 w-full' key={`cs-${i}`}>
                  <div className='flex-none w-8 text-[#aaa]'>[{i + 1}]</div>
                  <Card className='w-[95%] no-scrollbar rounded-none'>
                    <CardContent className='p-0'>
                      <CodeMirror
                        value={c.join('\n')}
                        theme={githubLight}
                        extensions={[pythonLanguage, themeDemo]}
                        basicSetup={{
                          highlightActiveLine: false,
                          highlightActiveLineGutter: false,
                          foldGutter: false,
                          lineNumbers: true
                        }}
                      />
                    </CardContent>
                  </Card>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
