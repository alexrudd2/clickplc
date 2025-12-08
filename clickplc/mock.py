"""
Python mock driver for AutomationDirect (formerly Koyo) ClickPLCs.

Uses local storage instead of remote communications.

Distributed under the GNU General Public License v2
"""
from collections import defaultdict
from dataclasses import dataclass
from unittest.mock import MagicMock

try:
    from pymodbus.pdu.bit_message import (  # type: ignore[import-not-found]
        ReadCoilsResponse,  # type: ignore[reportAssignmentType]
        WriteMultipleCoilsResponse,  # type: ignore[reportAssignmentType]
    )
    from pymodbus.pdu.register_message import (  # type: ignore[import-not-found]
        ReadHoldingRegistersResponse,  # type: ignore[reportAssignmentType]
        WriteMultipleRegistersResponse,  # type: ignore[reportAssignmentType]
    )
except ImportError:

    @dataclass
    class ReadCoilsResponse:  # type: ignore[no-redef] # noqa: D101
        bits: list[bool]

    class WriteMultipleCoilsResponse(MagicMock): ...  # type: ignore[no-redef] # noqa: D101
    @dataclass
    class ReadHoldingRegistersResponse:  # type: ignore[no-redef] # noqa: D101
        registers: list[int]

    class WriteMultipleRegistersResponse(MagicMock): ...  # type: ignore[no-redef] # noqa: D101


from clickplc.driver import ClickPLC as realClickPLC


class AsyncClientMock(MagicMock):
    """Magic mock that works with async methods."""

    async def __call__(self, *args, **kwargs):
        """Convert regular mocks into into an async coroutine."""
        return super().__call__(*args, **kwargs)


class ClickPLC(realClickPLC):
    """A version of the driver replacing remote communication with local storage for testing."""

    def __init__(self, address, tag_filepath='', timeout=1):
        self.tags = self._load_tags(tag_filepath)
        self.active_addresses = self._get_address_ranges(self.tags)
        self.client = AsyncClientMock()
        self._coils: dict[int, bool] = defaultdict(bool)
        self._discrete_inputs: dict[int, bool] = defaultdict(bool)
        self._registers: dict[int, bytes] = defaultdict(bytes)
        self._detect_pymodbus_version()
        if self.pymodbus33plus:
            self.client.close = lambda: None

    async def _request(self, method, address, count=0, values=()):  # type: ignore
        if method == 'read_coils':
            bits = [self._coils[address + i] for i in range(count)]
            return ReadCoilsResponse(bits=bits)
        elif method == 'read_holding_registers':
            registers = [int.from_bytes(self._registers[address + i], byteorder='big')
                         for i in range(count)]
            return ReadHoldingRegistersResponse(registers=registers)
        elif method == 'write_coils':
            for i, d in enumerate(values):
                self._coils[address + i] = d
            return WriteMultipleCoilsResponse(address, values)
        elif method == 'write_registers':
            for i, d in enumerate(values):
                self._registers[address + i] = d.to_bytes(length=2, byteorder='big')
            return WriteMultipleRegistersResponse(address, values)
        raise NotImplementedError(f'Unrecognised method: {method}')
