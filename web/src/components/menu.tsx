import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Menubar,
  MenubarContent,
  MenubarItem,
  MenubarLabel,
  MenubarMenu,
  MenubarRadioGroup,
  MenubarRadioItem,
  MenubarSeparator,
  MenubarShortcut,
  MenubarTrigger
} from '@/components/ui/menubar'
import { Separator } from '@/components/ui/separator'
import { useBoundStore } from '@/stores'
import { Link, useRouter } from '@tanstack/react-router'
import { Typography } from './ui/typography'

import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { FormEvent, useState } from 'react'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'

export function Menu () {
  const url = useBoundStore((state) => state.backendURL)
  const setURL = useBoundStore((state) => state.setBackendURL)
  const connStatus = useBoundStore((state) => state.backendStatus)
  const connStatusColor = useBoundStore((state) => state.backendStatusColor)
  const connect = useBoundStore((state) => state.connectToBackend)
  const [isOpeningNotebook, setOpeningNotebook] = useState<boolean>(false)
  const router = useRouter()
  const setCurrentNav = useBoundStore((state) => state.setCurrentNav)
  const selectNotebook = useBoundStore((state) => state.selectNotebook)

  const onOpenNotebookSubmit = (e: FormEvent) => {
    e.preventDefault()
    setOpeningNotebook(false)
    setTimeout(() => {
      setCurrentNav('Notebook Viewer')
      router.navigate({ to: '/notebook' })
    }, 200)
  }

  const warningMessage = 'Currently, the uploading functionality is not implemented. ' +
    'If you want to test your own notebook, ' +
    'please put them into the `public` folder and overwrite existing samples.'

  return (
    <>
      <Menubar className='rounded-none border-b border-none px-2 lg:px-4'>
        <Link to='/'>
          <MenubarMenu>
            <MenubarTrigger className='font-bold'>NB2P</MenubarTrigger>
          </MenubarMenu>
        </Link>
        <MenubarMenu>
          <MenubarTrigger className='relative'>File</MenubarTrigger>
          <MenubarContent>
            <MenubarItem onClick={() => { setOpeningNotebook(true) }}>
              Open Notebook <MenubarShortcut>⌘O</MenubarShortcut>
            </MenubarItem>
          </MenubarContent>
        </MenubarMenu>
        <MenubarMenu>
          <MenubarTrigger className='hidden md:block'>Account</MenubarTrigger>
          <MenubarContent forceMount>
            <MenubarLabel inset>Switch Account</MenubarLabel>
            <MenubarSeparator />
            <MenubarRadioGroup value='benoit'>
              <MenubarRadioItem value='andy'>Andy</MenubarRadioItem>
              <MenubarRadioItem value='benoit'>Benoit</MenubarRadioItem>
              <MenubarRadioItem value='Luis'>Luis</MenubarRadioItem>
            </MenubarRadioGroup>
            <MenubarSeparator />
            <MenubarItem inset>Manage Accounts</MenubarItem>
          </MenubarContent>
        </MenubarMenu>
        <MenubarMenu>
          <MenubarTrigger>About</MenubarTrigger>
          <MenubarContent>
            <Link to='/about'>
              <MenubarItem>About NB2P</MenubarItem>
            </Link>
            <MenubarSeparator />
            <MenubarItem>
              Preferences... <MenubarShortcut>⌘,</MenubarShortcut>
            </MenubarItem>
          </MenubarContent>
        </MenubarMenu>

        <div className='flex w-full h-full space-x-2 sm:justify-end items-center'>
          <Label>Backend: </Label>
          <Input className='w-56 h-6' defaultValue={url} onChange={(e) => setURL(e.target.value)} />
          <Button variant='secondary' size='sm' className='h-6' onClick={connect}>Connect</Button>
          <Separator orientation='vertical' />
          <div className='w-36 flex space-x-2'>
            <Label>Status: </Label>
            <Typography affects='small' style={{ color: connStatusColor }}>{connStatus}</Typography>
          </div>
        </div>
      </Menubar>

      <Dialog open={isOpeningNotebook} onOpenChange={() => setOpeningNotebook(!isOpeningNotebook)}>
        <DialogContent className='sm:max-w-md'>
          <DialogHeader>
            <DialogTitle>Open Notebook</DialogTitle>
          </DialogHeader>
          {/* <p className='text-sm'>Please input the URL to fetch the notebook file.</p>
          <div className='flex items-center space-x-2'>
            <div className='grid flex-1 gap-2'>
              <Label htmlFor='link' className='sr-only'>
                URL
              </Label>
              <Input
                id='link'
                defaultValue='https://demo.nb2p.org/example-notebook.ipynb'
                readOnly
              />
            </div>
          </div>
          <Separator />
          <p className='text-sm'>Or, select sample notebooks for test purposes.</p> */}
          <p className='text-sm'>Please select sample notebooks for test purposes.</p>
          <p className='text-xs'>{warningMessage}</p>

          <Select onValueChange={(value) => selectNotebook(value)}>
            <SelectTrigger>
              <SelectValue placeholder='Select sample notebook' />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value='sample-1'>Sample Notebook 1</SelectItem>
              <SelectItem value='sample-2'>Sample Notebook 2</SelectItem>
              <SelectItem value='sample-3'>Sample Notebook 3</SelectItem>
            </SelectContent>
          </Select>
          <DialogFooter className='sm:justify-start'>
            <DialogClose asChild>
              <Button type='button' onClick={onOpenNotebookSubmit}>
                Open
              </Button>
            </DialogClose>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
