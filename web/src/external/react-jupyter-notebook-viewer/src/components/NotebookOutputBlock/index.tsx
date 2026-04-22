import React, { useEffect } from 'react'

import { vs2015, github } from 'react-syntax-highlighter/dist/esm/styles/hljs'
import { type NotebookOutputBlockType } from './types'

import SynaxHighlighter from 'react-syntax-highlighter'

const NotebookOutputBlock: React.FC<NotebookOutputBlockType> = (props) => {
  const {
    executionCount,
    data,
    outputType,
    notebookInputLanguage,
    notebookOutputLanguage,
    showOutputLineNumbers,
    outputDarkTheme,
    outputOuterClassName,
    outputTextClassName,
    outputBlockClassName,
    outputTableClassName,
    outputImageClassName,
    outputBorderClassName,
    activeExecutionCount,
    index
  } = props

  useEffect(() => {
    document.querySelectorAll('table').forEach((table) => {
      outputTableClassName
        ? outputTableClassName.split(' ').map((className) => { table.classList.add(className) })
        : table.classList.add(...['min-w-full', 'text-right', 'table-auto', 'mx-auto'])
    })

    document.querySelectorAll('table thead').forEach((table) => {
      table.classList.add(...['font-bold'])
      outputDarkTheme
        ? table.classList.add(...['bg-zinc-900', 'text-zinc-300'])
        : table.classList.add(...['bg-white', 'text-gray-700'])
    })

    document.querySelectorAll('table thead th').forEach((table) => {
      table.classList.add(...['p-3', 'border-b'])
      outputDarkTheme
        ? table.classList.add(...['border-gray-700'])
        : table.classList.add(...['border-gray-300'])
    })

    document.querySelectorAll('table tbody').forEach((table) => {
      outputDarkTheme
        ? table.classList.add(...['bg-zinc-900', 'text-zinc-300'])
        : table.classList.add(...['bg-white', 'text-gray-700'])
    })

    document.querySelectorAll('table tbody tr').forEach((table) => {
      table.classList.add(...['border-b'])
      outputDarkTheme
        ? table.classList.add(...['border-gray-700'])
        : table.classList.add(...['border-gray-300'])
    })

    document.querySelectorAll('table tbody tr td').forEach((table) => {
      table.classList.add(...['py-3', 'px-4'])
    })
  }, [data, outputDarkTheme, outputTableClassName])

  const renderPlainTextBlock = () => (
    <>
      {data && !Array.isArray(data) && (
        <SynaxHighlighter
          language={notebookOutputLanguage || notebookInputLanguage}
          style={outputDarkTheme ? vs2015 : github}
          showLineNumbers={showOutputLineNumbers}
        >
          {data['text/plain']}
        </SynaxHighlighter>
      )}
    </>
  )

  const renderHtmlTextBlock = () => (
    <>
      {data && !Array.isArray(data) && data['text/html'] && (
        <div
          className='overflow-x-auto'
          dangerouslySetInnerHTML={{
            __html: data['text/html']
          }}
        />
      )}
    </>
  )

  const renderImageBlock = () => (
    <>
      {data && !Array.isArray(data) && data['image/png'] && (
        <img
          className={outputImageClassName}
          src={`data:image/png;base64,${data['image/png']}`}
          alt=''
        />
      )}
    </>
  )

  const renderStreamBlock = () => (
    <SynaxHighlighter
      language={notebookOutputLanguage || notebookInputLanguage}
      style={outputDarkTheme ? vs2015 : github}
      showLineNumbers={showOutputLineNumbers}
    >
      {data && Array.isArray(data) ? data.join('') : data}
    </SynaxHighlighter>
  )

  const renderErrorBlock = () => (
    <SynaxHighlighter
      customStyle={{ backgroundColor: 'rgb(239, 68, 68)' }}
      language={notebookOutputLanguage || notebookInputLanguage}
      style={outputDarkTheme ? vs2015 : github}
      showLineNumbers={showOutputLineNumbers}
    >
      {data && Array.isArray(data) ? data.join('') : data}
    </SynaxHighlighter>
  )

  return (
    <div
      className={`output-block output-${executionCount} flex w-full py-2 text-sm ${
                outputOuterClassName || ''
            } ${
                activeExecutionCount === executionCount
                    ? `border-l-8 ${
                          outputBorderClassName || 'border-blue-400'
                      } my-2 pl-2 md:pl-0`
                    : 'border-l-8 border-transparent my-2 pl-2 md:pl-0'
            }`}
    >
      <p
        className={`output-block-text hidden md:flex font-semibold justify-end md:pr-12 xl:pr-6 ${
                    outputTextClassName || ''
                } ${
                    activeExecutionCount === executionCount
                        ? 'text-red-500'
                        : outputDarkTheme
                        ? 'text-white'
                        : 'text-black'
                }
                }`}
      >
        {index === 0 && <>Out [{executionCount}]:</>}
      </p>
      <div className={`w-full ${outputBlockClassName || ''}`}>
        {data &&
                    !Array.isArray(data) &&
                    !data['text/html'] &&
                    data['text/plain'] &&
                    renderPlainTextBlock()}
        {data &&
                    !Array.isArray(data) &&
                    data['text/html'] &&
                    data['text/plain'] &&
                    renderHtmlTextBlock()}
        {data && !Array.isArray(data) && data['image/png'] && renderImageBlock()}
        {outputType === 'stream' && renderStreamBlock()}
        {outputType === 'error' && renderErrorBlock()}
      </div>
    </div>
  )
}

export { NotebookOutputBlock }
