from rich.console import Console, ConsoleOptions, RenderResult
from rich.control import Control
from rich.live import Live
from rich.live_render import LiveRender
from rich.segment import Segment


class RowCountingLiveRender(LiveRender):
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        self.console = console
        segments = list(super().__rich_console__(console, options))
        self.lengths = [Segment.get_line_length(line) for line in Segment.split_lines(segments)]
        yield from segments

    def count_rows(self) -> None:
        if self._shape is not None:
            width = self.console.width
            self._shape = (width, sum(max(1, -(-length // width)) for length in self.lengths))

    def position_cursor(self) -> Control:
        self.count_rows()
        return super().position_cursor()

    def restore_cursor(self) -> Control:
        self.count_rows()
        return super().restore_cursor()


def count_rows_after_resize(live: Live) -> Live:
    # Rich erases as many rows as it printed lines, but a terminal that became narrower has wrapped the lines
    # that no longer fit onto more rows, and the topmost of them stay on the screen.
    live._live_render = RowCountingLiveRender(live.get_renderable(), vertical_overflow=live.vertical_overflow)
    return live
