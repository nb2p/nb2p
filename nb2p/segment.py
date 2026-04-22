"""Segment-related classes."""

from abc import ABCMeta, abstractmethod
from typing import List, Optional

from .notebook import Cell, Notebook


class SegmentsBuilder(metaclass=ABCMeta):
    def __init__(self, notebook: Notebook) -> None:
        self._notebook = notebook

    @abstractmethod
    def build(self, pred_ends: List[int]) -> Optional[List[Cell]]:
        raise NotImplementedError()


class CellEndSegmentsBuilder(SegmentsBuilder):
    def __init__(self, notebook: Notebook) -> None:
        super().__init__(notebook)

    def build(self, pred_ends: List[int]) -> Optional[List[List[str]]]:
        result = []
        curr_segment = []

        for i, cell in enumerate(self._notebook.code_cells):
            curr_segment.extend(cell.content)

            if i >= len(pred_ends) or pred_ends[i] > 0.5:
                ### For the first expression:
                ### Some has problematic ending. Consider it as 1
                result.append(curr_segment)
                curr_segment = []

        return result


class LineEndSegmentsBuilder(SegmentsBuilder):
    def __init__(self, notebook: Notebook) -> None:
        super().__init__(notebook)

    def build(self, pred_ends: List[int]) -> Optional[List[List[str]]]:
        result = []
        curr_segment = []

        for i, line in enumerate(self._notebook.code_lines):
            curr_segment.append(line)

            if i >= len(pred_ends) or pred_ends[i] > 0.5:
                result.append("\n".join(curr_segment))
                curr_segment = []

        return result
