import { Outlet, RootRoute } from '@tanstack/react-router'

import { Menu } from '@/components/menu'
import { useState } from 'react'
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable'
import { TooltipProvider } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'
import { Nav } from '@/components/nav'
import { NotebookTabs, File, Send, ArchiveX } from 'lucide-react'
import { Toaster } from '@/components/ui/sonner'

const Root = ({
  defaultLayout = [265, 440, 655],
  defaultCollapsed = false,
  navCollapsedSize = 4
}) => {
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed)
  return (
    <>
      <div className='h-[calc(100vh-37px)] '>
        <div className='border-b'>
          <Menu />
        </div>
        <TooltipProvider delayDuration={0}>
          <ResizablePanelGroup
            direction='horizontal'
            onLayout={(sizes: number[]) => {
              document.cookie = `react-resizable-panels:layout=${JSON.stringify(
                sizes
              )}`
            }}
            className='min-h-full items-stretch'
          >
            <ResizablePanel
              defaultSize={defaultLayout[0]}
              collapsedSize={navCollapsedSize}
              collapsible
              minSize={15}
              maxSize={20}
              onCollapse={() => {
                setIsCollapsed(true)
                document.cookie = `react-resizable-panels:collapsed=${JSON.stringify(
                  true
                )}`
              }}
              onExpand={() => {
                setIsCollapsed(false)
                document.cookie = `react-resizable-panels:collapsed=${JSON.stringify(
                  false
                )}`
              }}
              className={cn(isCollapsed && 'min-w-[50px] transition-all duration-300 ease-in-out')}
            >
              <Nav
                isCollapsed={isCollapsed}
                links={[
                  {
                    title: 'Notebook Viewer',
                    icon: NotebookTabs,
                    to: '/notebook'
                  },
                  {
                    title: 'Restored Session',
                    icon: File,
                    to: '/session'
                  },
                  {
                    title: 'Segmentation',
                    icon: Send,
                    to: '/segmentation'
                  },
                  {
                    title: 'Pipeline Viewer',
                    icon: ArchiveX,
                    to: '/pipeline-viewer'
                  }
                ]}
              />
            </ResizablePanel>
            <ResizableHandle withHandle />
            <ResizablePanel defaultSize={defaultLayout[1]} minSize={30}>
              <Outlet />
            </ResizablePanel>
          </ResizablePanelGroup>
        </TooltipProvider>
      </div>
      <Toaster />
    </>
  )
}

export const Route = new RootRoute({
  component: () => (<Root />)
})
