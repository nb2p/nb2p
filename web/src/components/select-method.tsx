import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'

export function SelectMethod ({ value, onValueChange }: {
  value?: string
  onValueChange?(value: string): void
}) {
  return (
    <Select value={value} onValueChange={onValueChange}>
      <SelectTrigger>
        <SelectValue placeholder='Select segmentation method' />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value='Decision Tree'>Decision Tree</SelectItem>
        <SelectItem value='Random Forest'>Random Forest</SelectItem>
        <SelectItem value='XGBoost'>XGBoost</SelectItem>
        <SelectItem value='Transformer'>Transformer</SelectItem>
        <SelectItem value='NB2P-SS'>NB2P-SS</SelectItem>
      </SelectContent>
    </Select>
  )
}
