'use client'

import { type LucideIcon } from 'lucide-react'

import { cn } from '@/lib/utils'
import { buttonVariants } from '@/components/ui/button'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'
import { Link } from '@tanstack/react-router'
import { useBoundStore } from '@/stores'

interface NavProps {
  isCollapsed: boolean
  links: Array<{
    title: string
    icon: LucideIcon
    to: string
  }>
}

export function Nav ({ links, isCollapsed }: NavProps): JSX.Element {
  const currentSelected = useBoundStore((state) => state.currentNav)
  const setCurrentSelected = useBoundStore((state) => state.setCurrentNav)
  const variants = links.map((l) => l.title === currentSelected ? 'default' : 'ghost')

  return (
    <div
      data-collapsed={isCollapsed}
      className='group flex flex-col h-full justify-center gap-4 py-2 data-[collapsed=true]:py-2'
    >
      <nav className='grid gap-1 px-2 group-[[data-collapsed=true]]:justify-center group-[[data-collapsed=true]]:px-2'>
        {links.map((link, index) =>
          isCollapsed
            ? (
              <Tooltip key={index} delayDuration={0}>
                <TooltipTrigger asChild>
                  <Link
                    to={link.to}
                    className={cn(
                      buttonVariants({ variant: variants[index], size: 'icon' }),
                      'h-10 w-10',
                      link.title === currentSelected &&
                      'dark:bg-muted dark:text-muted-foreground dark:hover:bg-muted dark:hover:text-white'
                    )}
                    onClick={() => { setCurrentSelected(link.title) }}
                  >
                    <link.icon className='h-4 w-4' />
                  </Link>
                </TooltipTrigger>
                <TooltipContent side='right' className='flex items-center gap-4'>
                  {link.title}
                </TooltipContent>
              </Tooltip>
              )
            : (
              <Link
                key={index}
                to={link.to}
                className={cn(
                  buttonVariants({ variant: variants[index], size: 'lg' }),
                  'px-4',
                  link.title === currentSelected &&
                  'dark:bg-muted dark:text-white dark:hover:bg-muted dark:hover:text-white',
                  'justify-start'
                )}
                onClick={() => { setCurrentSelected(link.title) }}
              >
                <link.icon className='mr-2 h-4 w-4' />
                {link.title}
              </Link>
              )
        )}
      </nav>
    </div>
  )
}
