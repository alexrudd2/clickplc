"""Base functionality for modbus communication.

Distributed under the GNU General Public License v2
Copyright (C) 2022 NuMat Technologies
Copyright (C) 2024 Alex Ruddick
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal, overload

import pymodbus.exceptions
from pymodbus.client import AsyncModbusTcpClient  # 3.x

if TYPE_CHECKING:
    try:  # pymodbus >= 3.8.x
        from pymodbus.pdu.bit_message import ReadCoilsResponse, WriteMultipleCoilsResponse
        from pymodbus.pdu.register_message import ReadHoldingRegistersResponse, WriteMultipleRegistersResponse
    except ImportError:
        try:
            from pymodbus.pdu.bit_read_message import ReadCoilsResponse  # type: ignore
            from pymodbus.pdu.bit_write_message import WriteMultipleCoilsResponse  # type: ignore
            from pymodbus.pdu.register_read_message import ReadHoldingRegistersResponse  # type: ignore
            from pymodbus.pdu.register_write_message import WriteMultipleRegistersResponse  # type: ignore
        except ImportError:  # pymodbus < 3.7.0
            from pymodbus.bit_read_message import ReadCoilsResponse  # type: ignore
            from pymodbus.bit_write_message import WriteMultipleCoilsResponse  # type: ignore
            from pymodbus.register_read_message import ReadHoldingRegistersResponse  # type: ignore
            from pymodbus.register_write_message import WriteMultipleRegistersResponse  # type: ignore

class AsyncioModbusClient:
    """A generic asyncio client.

    This expands upon the pymodbus AsyncModbusTcpClient by
    including standard timeouts, async context manager, and queued requests.
    """

    def __init__(self, address, timeout=1):
        """Set up communication parameters."""
        self.ip = address
        self.port = 5020 if address == '127.0.0.1' else 502  # pymodbus simulator is 127.0.0.1:5020
        self.timeout = timeout
        self._detect_pymodbus_version()
        self.client = AsyncModbusTcpClient(address, timeout=timeout, port=self.port)  # pyright: ignore [reportPossiblyUnboundVariable]
        self.lock = asyncio.Lock()
        self.connectTask = asyncio.create_task(self._connect())

    async def __aenter__(self):
        """Asynchronously connect with the context manager."""
        return self

    async def __aexit__(self, *args) -> None:
        """Provide exit to the context manager."""
        await self._close()

    def _detect_pymodbus_version(self) -> None:
        """Detect various pymodbus versions."""
        major, minor, _patch = map(int, pymodbus.__version__.split('.')[:3])
        self.pymodbus30plus = major == 3
        self.pymodbus32plus = major == 3 and minor >= 2
        self.pymodbus33plus = major == 3 and minor >= 3
        self.pymodbus35plus = major == 3 and minor >= 5

    async def _connect(self) -> None:
        """Start asynchronous reconnect loop."""
        try:
            await asyncio.wait_for(self.client.connect(), timeout=self.timeout)  # 3.x
        except Exception as e:
            raise OSError(f"Could not connect to '{self.ip}'.") from e

    async def read_coils(self, address: int, count):
        """Read modbus output coils (0 address prefix)."""
        return await self._request('read_coils', address=address, count=count)

    async def read_registers(self, address: int, count):
        """Read modbus registers.

        The Modbus protocol doesn't allow responses longer than 250 bytes
        (ie. 125 registers, 62 DF addresses), which this function manages by
        chunking larger requests.
        """
        registers = []
        while count > 124:
            r = await self._request('read_holding_registers', address=address, count=124)
            registers += r.registers
            address, count = address + 124, count - 124
        r = await self._request('read_holding_registers', address=address, count=count)
        registers += r.registers
        return registers

    async def write_coils(self, address: int, values) -> None:
        """Write modbus coils."""
        await self._request('write_coils', address=address, values=values)

    async def write_registers(self, address: int, values) -> None:
        """Write modbus registers.

        The Modbus protocol doesn't allow requests longer than 250 bytes
        (ie. 125 registers, 62 DF addresses), which this function manages by
        chunking larger requests.
        """
        while len(values) > 62:
            await self._request('write_registers', address=address, values=values)
            address, values = address + 124, values[62:]
        await self._request('write_registers', address=address, values=values)

    @overload
    async def _request(
        self, method: Literal["read_coils"], address: int, count: int,
    ) -> ReadCoilsResponse: ...

    @overload
    async def _request(
        self, method: Literal["write_coils"], address: int, values: Any,
    ) -> WriteMultipleCoilsResponse: ...

    @overload
    async def _request(
        self, method: Literal["read_holding_registers"], address: int, count: int,
    ) -> ReadHoldingRegistersResponse: ...

    @overload
    async def _request(
        self, method: Literal["write_registers"], address:int, values: Any,
    ) -> WriteMultipleRegistersResponse: ...
    async def _request(self, method, *args, **kwargs):
        """Send a request to the device and awaits a response.

        This mainly ensures that requests are sent serially, as the Modbus
        protocol does not allow simultaneous requests (it'll ignore any
        request sent while it's processing something). The driver handles this
        by assuming there is only one client instance. If other clients
        exist, other logic will have to be added to either prevent or manage
        race conditions.
        """
        await self.connectTask
        async with self.lock:
            try:
                if self.pymodbus32plus:  # noqa: SIM108
                    future = getattr(self.client, method)
                else:
                    future = getattr(self.client.protocol, method)  # type: ignore[attr-defined]
                return await future(*args, **kwargs)
            except (asyncio.TimeoutError, pymodbus.exceptions.ConnectionException) as e:
                raise TimeoutError("Not connected to PLC.") from e

    async def _close(self) -> None:
        """Close the TCP connection."""
        if self.pymodbus33plus:
            self.client.close()  # 3.3.x
        else:
            await self.client.close()  # type: ignore  # 3.0.x - 3.2.x
