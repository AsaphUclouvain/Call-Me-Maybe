import sys


class JsonStreamPrettyPrinter:
    OPPOSITE = {"{": "}", "[": "]"}
    WHITESPACE = " \n\r\t"

    def __init__(self) -> None:
        self.quote = 0
        self.escaped = False
        self.opened = 0
        # open bracket awaiting its "empty?" verdict
        self.pending_bracket: str | None = None
        self.chunks: list[str] = []

    def _update_quote(self, c: str) -> None:
        if self.escaped:
            self.escaped = False
        elif c == '\\':
            self.escaped = True
        elif c == '"':
            self.quote += 1

    def _emit(self, s: str) -> None:
        self.chunks.append(s)

    def _handle(self, c: str) -> None:
        self._update_quote(c)
        outside_string = self.quote % 2 == 0

        if self.pending_bracket is not None:
            if outside_string and c in self.WHITESPACE:
                return  # keep swallowing, still undecided
            opener = self.pending_bracket
            if c == self.OPPOSITE[opener]:
                self._emit(opener + c)          # empty pair -> "{}" / "[]"
                self.pending_bracket = None
                return
            self.opened += 1
            self._emit(opener + '\n' + '\t' * self.opened)  # not empty
            self.pending_bracket = None
            # fall through, c still needs normal handling

        if outside_string and c in "[{":
            self.pending_bracket = c
            return
        if outside_string and c in "}]":
            self.opened -= 1
            self._emit('\n' + '\t' * self.opened + c)
            return
        if outside_string and c == ',':
            self._emit(c + '\n' + '\t' * self.opened)
            return
        if outside_string and c == ':':
            self._emit(c + ' ')
            return
        if outside_string and c in self.WHITESPACE:
            return
        self._emit(c)

    def feed(self, token: str) -> str:
        start = len(self.chunks)
        for c in token:
            self._handle(c)
        return "".join(self.chunks[start:])

    def finish(self) -> str:
        start = len(self.chunks)
        if self.pending_bracket is not None:
            self.opened += 1
            self._emit(self.pending_bracket + '\n' + '\t' * self.opened)
            self.pending_bracket = None
        return "".join(self.chunks[start:])


def pretty_print_json(
    finish: bool,
    json_str: str,
    printer: JsonStreamPrettyPrinter
) -> None:
    s = printer.feed(json_str)
    if s:
        print(s, end="")
    if finish:
        print(printer.finish(), end="")
    sys.stdout.flush()
