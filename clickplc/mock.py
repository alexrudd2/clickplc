"""
Python mock driver for AutomationDirect (formerly Koyo) ClickPLCs.

Uses local storage instead of remote communications.

Distributed under the GNU General Public License v2
"""
from collections import defaultdict
from unittest.mock import MagicMock

try:
    from pymodbus.pdu.bit_message import (
        ReadCoilsResponse,
        ReadDiscreteInputsResponse,
        WriteMultipleCoilsResponse,
    )
    from pymodbus.pdu.register_message import ReadHoldingRegistersResponse, WriteMultipleRegistersResponse
    pymodbus38plus = True
except ImportError:
    pymodbus38plus = False
    try:  # pymodbus 3.7.x
        from pymodbus.pdu.bit_read_message import ReadCoilsResponse, ReadDiscreteInputsResponse  # type: ignore
        from pymodbus.pdu.bit_write_message import WriteMultipleCoilsResponse  # type: ignore
        from pymodbus.pdu.register_read_message import ReadHoldingRegistersResponse  # type: ignore
        from pymodbus.pdu.register_write_message import WriteMultipleRegistersResponse  # type: ignore
    except ImportError:
        from pymodbus.bit_read_message import ReadCoilsResponse, ReadDiscreteInputsResponse  # type: ignore
        from pymodbus.bit_write_message import WriteMultipleCoilsResponse  # type: ignore
        from pymodbus.register_read_message import ReadHoldingRegistersResponse  # type: ignore
        from pymodbus.register_write_message import WriteMultipleRegistersResponse  # type: ignore

from clickplc.driver import ClickPLC as realClickPLC


class AsyncClientMock(MagicMock):
    """Magic mock that works with async methods."""

    async def __call__(self, *args, **kwargs):
        """Convert regular mocks into into an async coroutine."""
        return super().__call__(*args, **kwargs)

    def stop(self) -> None:
        """Close the connection (2.5.3)."""
        ...


class ClickPLC(realClickPLC):
    """A version of the driver replacing remote communication with local storage for testing."""

    def __init__(self, address, tag_filepath='', timeout=1):
        self.tags = self._load_tags(tag_filepath)
        self.active_addresses = self._get_address_ranges(self.tags)
        self.client = AsyncClientMock()
        self._coils = defaultdict(bool)
        self._discrete_inputs = defaultdict(bool)
        self._registers = defaultdict(bytes)
        self._detect_pymodbus_version()
        if self.pymodbus33plus:
            self.client.close = lambda: None

    async def _request(self, method, address, count=0, values=(), **kwargs):  # noqa: C901
        if method == 'read_coils':
            bits = [self._coils[address + i] for i in range(count)]
            if pymodbus38plus:
                return ReadCoilsResponse(bits=bits)
            return ReadCoilsResponse(bits)  # type: ignore[arg-type]
        if method == 'read_discrete_inputs':
            bits = bits = [self._discrete_inputs[address + i]
                           for i in range(count)]
            if pymodbus38plus:
                return ReadDiscreteInputsResponse(bits=bits)
            return ReadDiscreteInputsResponse(bits)  # type: ignore[arg-type]
        elif method == 'read_holding_registers':
            registers = [int.from_bytes(self._registers[address + i], byteorder='big')
                         for i in range(count)]
            if pymodbus38plus:
                return ReadHoldingRegistersResponse(registers=registers)
            return ReadHoldingRegistersResponse(registers)  # type: ignore[arg-type]
        elif method == 'write_coils':
            for i, d in enumerate(values):
                self._coils[address + i] = d
            return WriteMultipleCoilsResponse(address, values)
        elif method == 'write_registers':
            for i, d in enumerate(values):
                self._registers[address + i] = d.to_bytes(length=2, byteorder='big')
            return WriteMultipleRegistersResponse(address, values)
        return NotImplementedError(f'Unrecognised method: {method}')
