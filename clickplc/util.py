"""Base functionality for modbus communication.

Distributed under the GNU General Public License v2
Copyright (C) 2022 NuMat Technologies
"""
from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any, Literal

import pymodbus.exceptions

try:
    from pymodbus.client import AsyncModbusSerialClient, AsyncModbusTcpClient  # 3.x
except ImportError:  # 2.4.x - 2.5.x
    from pymodbus.client.asynchronous.async_io import (  # type: ignore
        ReconnectingAsyncioModbusSerialClient,
        ReconnectingAsyncioModbusTcpClient,
    )



class AsyncioModbusClient:
    """A generic asyncio client.

    This expands upon the pymodbus AsyncModbusTcpClient by
    including standard timeouts, async context manager, and queued requests.
    """

    def __init__(self,
                 address,
                 timeout=1,
                 interfacetype: Literal["TCP", "Serial"] = "TCP",
                 *,
                 baudrate=38400,
                 parity='O',
                 stopbits=1,
                 bytesize=8):
        """Set up communication parameters."""
        self.ip = address
        self.port = 5020 if address == '127.0.0.1' else 502  # pymodbus simulator is 127.0.0.1:5020
        self.timeout = timeout
        self._detect_pymodbus_version()
        if self.pymodbus30plus and interfacetype == "TCP":
            self.client = AsyncModbusTcpClient(address, timeout=timeout, port=self.port)  # pyright: ignore [reportPossiblyUnboundVariable]
        elif self.pymodbus30plus and interfacetype == "Serial":
            self.client = AsyncModbusSerialClient(
                port=address,
                timeout=timeout,
                baudrate=baudrate,
                bytesize=bytesize,
                parity=parity,
                stopbits=stopbits
            )
        elif interfacetype == "TCP":  # 2.x
            self.client = ReconnectingAsyncioModbusTcpClient()  # pyright: ignore [reportPossiblyUnboundVariable]
        elif interfacetype == "Serial":
            self.client = ReconnectingAsyncioModbusSerialClient() # pyright: ignore [reportPossiblyUnboundVariable]
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
        self.pymodbus30plus = int(pymodbus.__version__[0]) == 3
        self.pymodbus32plus = self.pymodbus30plus and int(pymodbus.__version__[2]) >= 2
        self.pymodbus33plus = self.pymodbus30plus and int(pymodbus.__version__[2]) >= 3
        self.pymodbus35plus = self.pymodbus30plus and int(pymodbus.__version__[2]) >= 5

    async def _connect(self) -> None:
        """Start asynchronous reconnect loop."""
        try:
            if self.pymodbus30plus:
                await asyncio.wait_for(self.client.connect(), timeout=self.timeout)  # 3.x
            else:  # 2.4.x - 2.5.x
                await self.client.start(host=self.ip, port=self.port)  # type: ignore[attr-defined]
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
    

    def _convert_from_registers(
            self,
            registers: list[int],
            data_type: Any,
            word_order: Literal['big', 'little'] = 'big',
            string_encoding: str = 'utf-8'
            ) -> Any: #int | float | str | list[bool] | list[int] | list[float]:
        return self.client.convert_from_registers(
            registers,
            data_type
        )

    async def write_coils(self, address: int, values):
        """Write modbus coils."""
        await self._request('write_coils', address=address, values=values)

    async def write_registers(self, address: int, values):
        """Write modbus registers.

        The Modbus protocol doesn't allow requests longer than 250 bytes
        (ie. 125 registers, 62 DF addresses), which this function manages by
        chunking larger requests.
        """
        while len(values) > 62:
            await self._request('write_registers', address=address, values=values)
            address, values = address + 124, values[62:]
        await self._request('write_registers', address=address, values=values)

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

    async def _close(self):
        """Close the TCP connection."""
        if self.pymodbus33plus:
            self.client.close()  # 3.3.x
        elif self.pymodbus30plus:
            await self.client.close()  # type: ignore  # 3.0.x - 3.2.x
        else:  # 2.4.x - 2.5.x
            self.client.stop()  # type: ignore[attr-defined]
