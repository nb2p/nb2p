import { Typography } from '@/components/ui/typography'

export const component = function About (): JSX.Element {
  return (
    <div className='flex flex-col justify-center items-center h-full'>
      <Typography variant='h2'>Welcome to NB2P!</Typography>
      <Typography variant='p'>Please start by
        <code className='relative rounded bg-muted mx-[0.5rem] font-mono text-base font-semibold'>
          File - Open Notebook
        </code>
      </Typography>
    </div>
  )
}
