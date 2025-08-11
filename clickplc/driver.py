"""
A Python driver for Koyo ClickPLC ethernet units.

Distributed under the GNU General Public License v2
Copyright (C) 2024 Alex Ruddick
Copyright (C) 2020 NuMat Technologies
"""
from __future__ import annotations

import copy
import csv
import pydoc
import struct
from collections import defaultdict
from string import digits
from typing import Any, ClassVar, overload

from clickplc.util import AsyncioModbusClient


class ClickPLC(AsyncioModbusClient):
    """Ethernet driver for the Koyo ClickPLC.

    This interface handles the quirks of both Modbus TCP/IP and the ClickPLC,
    abstracting corner cases and providing a simple asynchronous interface.
    """

    data_types: ClassVar[dict] = {
        'x': 'bool',     # Input point
        'y': 'bool',     # Output point
        'c': 'bool',     # (C)ontrol relay
        't': 'bool',     # (T)imer
        'ct': 'bool',    # (C)oun(t)er
        'sc': 'bool',    # (S)ystem (c)ontrol relay
        'ds': 'int16',   # (D)ata register (s)ingle
        'dd': 'int32',   # (D)ata register, (d)ouble
        'dh': 'int16',   # (D)ata register, (h)ex
        'df': 'float',   # (D)ata register (f)loating point
        'td': 'int16',   # (T)imer register
        'ctd': 'int32',  # (C)oun(t)er Current values, (d)ouble
        'sd': 'int16',   # (S)ystem (D)ata register, single
        'txt': 'str',    # ASCII Text
    }

    def __init__(self, address, tag_filepath='', timeout=1):
        """Initialize PLC connection and data structure.

        Args:
            address: The PLC IP address or DNS name
            tag_filepath: Path to the PLC tags file
            timeout (optional): Timeout when communicating with PLC. Default 1s.

        """
        super().__init__(address, timeout)
        self.tags = self._load_tags(tag_filepath)
        self.active_addresses = self._get_address_ranges(self.tags)

    def get_tags(self) -> dict:
        """Return all tags and associated configuration information.

        Use this data for debugging or to provide more detailed
        information on user interfaces.

        Returns:
            A dictionary containing information associated with each tag name.

        """
        return copy.deepcopy(self.tags)

    async def get(self, address: str | None = None) -> dict:
        """Get variables from the ClickPLC.

        Args:
            address: ClickPLC address(es) to get. Specify a range with a
                hyphen, e.g. 'DF1-DF40'

        If driver is loaded with a tags file this can be called without an
        address to return all nicknamed addresses in the tags file
        >>> plc.get()
        {'P-101': 0.0, 'P-102': 0.0 ..., T-101:0.0}

        Otherwise one or more internal variable can be requested
        >>> plc.get('df1')
        0.0
        >>> plc.get('df1-df20')
        {'df1': 0.0, 'df2': 0.0, ..., 'df20': 0.0}
        >>> plc.get('y101-y316')
        {'y101': False, 'y102': False, ..., 'y316': False}

        This uses the ClickPLC's internal variable notation, which can be
        found in the Address Picker of the ClickPLC software.
        """
        if address is None:
            if not self.tags:
                raise ValueError('An address must be supplied to get if tags were not '
                                 'provided when driver initialized')
            results = {}
            for category, _address in self.active_addresses.items():
                results.update(await getattr(self, '_get_' + category)
                                            (_address['min'], _address['max']))
            return {tag_name: results[tag_info['id'].lower()]
                    for tag_name, tag_info in self.tags.items()}

        if '-' in address:
            start, end = address.split('-')
        else:
            start, end = address, None
        i = next(i for i, s in enumerate(start) if s.isdigit())
        category, start_index = start[:i].lower(), int(start[i:])
        end_index = None if end is None else int(end[i:])

        if end_index is not None and end_index <= start_index:
            raise ValueError("End address must be greater than start address.")
        if category not in self.data_types:
            raise ValueError(f"{category} currently unsupported.")
        if end is not None and end[:i].lower() != category:
            raise ValueError("Inter-category ranges are unsupported.")
        return await getattr(self, '_get_' + category)(start_index, end_index)

    async def set(self, address: str, data):
        """Set values on the ClickPLC.

        Args:
            address: ClickPLC address to set. If `data` is a list, it will set
                this and subsequent addresses.
            data: A value or list of values to set.

        >>> plc.set('df1', 0.0)  # Sets DF1 to 0.0
        >>> plc.set('df1', [0.0, 0.0, 0.0])  # Sets DF1-DF3 to 0.0.
        >>> plc.set('myTagNickname', True)  # Sets address named myTagNickname to true

        This uses the ClickPLC's internal variable notation, which can be
        found in the Address Picker of the ClickPLC software. If a tags file
        was loaded at driver initialization, nicknames can be used instead.
        """
        if address in self.tags:
            address = self.tags[address]['id']
        if not isinstance(data, list):
            data = [data]

        i = next(i for i, s in enumerate(address) if s.isdigit())
        category, index = address[:i].lower(), int(address[i:])
        if category not in self.data_types:
            raise ValueError(f"{category} currently unsupported.")
        data_type = self.data_types[category].rstrip(digits)
        for datum in data:
            if type(datum) == int and data_type == 'float':  # noqa: E721
                datum = float(datum)
            if type(datum) != pydoc.locate(data_type):  # noqa: E721
                raise ValueError(f"Expected {address} as a {data_type}.")
        return await getattr(self, '_set_' + category)(index, data)

    async def _get_x(self, start: int, end: int | None) -> dict:
        """Read X addresses. Called by `get`.

        X entries start at 0 (1 in the Click software's 1-indexed
        notation). This function also handles some of the quirks of the unit.

        First, the modbus addresses aren't sequential. Instead, the pattern is:
            X001 0
            [...]
            X016 15
            X101 32
            [...]
        The X addressing only goes up to *16, then jumps 16 coils to get to
        the next hundred. Rather than the overhead of multiple requests, this
        is handled by reading all the data and throwing away unowned addresses.

        Second, the response always returns a full byte of data. If you request
        a number of addresses not divisible by 8, it will have extra data. The
        extra data here is discarded before returning.
        """
        if start % 100 == 0 or start % 100 > 16:
            raise ValueError('X start address must be *01-*16.')
        if start < 1 or start > 816:
            raise ValueError('X start address must be in [001, 816].')

        start_coil = 32 * (start // 100) + start % 100 - 1
        if end is None:
            coils = await self.read_coils(start_coil, 1)
            return coils.bits[0]

        if end % 100 == 0 or end % 100 > 16:
            raise ValueError('X end address must be *01-*16.')
        if end < 1 or end > 816:
            raise ValueError('X end address must be in [001, 816].')
        end_coil = 32 * (end // 100) + end % 100 - 1
        count = end_coil - start_coil + 1
        coils = await self.read_coils(start_coil, count)
        output = {}
        current = start
        for bit in coils.bits:
            if current > end:
                break
            elif current % 100 <= 16:
                output[f'x{current:03}'] = bit
            elif current % 100 == 32:
                current += 100 - 32
            current += 1
        return output

    async def _get_y(self, start: int, end: int | None) -> dict:
        """Read Y addresses. Called by `get`.

        Y entries start at 8192 (8193 in the Click software's 1-indexed
        notation). This function also handles some of the quirks of the unit.

        First, the modbus addresses aren't sequential. Instead, the pattern is:
            Y001 8192
            [...]
            Y016 8208
            Y101 8224
            [...]
        The Y addressing only goes up to *16, then jumps 16 coils to get to
        the next hundred. Rather than the overhead of multiple requests, this
        is handled by reading all the data and throwing away unowned addresses.

        Second, the response always returns a full byte of data. If you request
        a number of addresses not divisible by 8, it will have extra data. The
        extra data here is discarded before returning.
        """
        if start % 100 == 0 or start % 100 > 16:
            raise ValueError('Y start address must be *01-*16.')
        if start < 1 or start > 816:
            raise ValueError('Y start address must be in [001, 816].')

        start_coil = 8192 + 32 * (start // 100) + start % 100 - 1
        if end is None:
            coils = await self.read_coils(start_coil, 1)
            return coils.bits[0]

        if end % 100 == 0 or end % 100 > 16:
            raise ValueError('Y end address must be *01-*16.')
        if end < 1 or end > 816:
            raise ValueError('Y end address must be in [001, 816].')
        end_coil = 8192 + 32 * (end // 100) + end % 100 - 1
        count = end_coil - start_coil + 1
        coils = await self.read_coils(start_coil, count)
        output = {}
        current = start
        for bit in coils.bits:
            if current > end:
                break
            elif current % 100 <= 16:
                output[f'y{current:03}'] = bit
            elif current % 100 == 32:
                current += 100 - 32
            current += 1
        return output

    async def _get_c(self, start: int, end: int | None) -> dict | bool:
        """Read C addresses. Called by `get`.

        C entries start at 16384 (16385 in the Click software's 1-indexed
        notation) and span a total of 2000 bits, ending at 18383.

        The response always returns a full byte of data. If you request
        a number of addresses not divisible by 8, it will have extra data. The
        extra data here is discarded before returning.
        """
        if start < 1 or start > 2000:
            raise ValueError('C start address must be 1-2000.')

        start_coil = 16384 + start - 1
        if end is None:
            return (await self.read_coils(start_coil, 1)).bits[0]

        if end <= start or end > 2000:
            raise ValueError('C end address must be >start and <=2000.')
        end_coil = 16384 + end - 1
        count = end_coil - start_coil + 1
        coils = await self.read_coils(start_coil, count)
        return {f'c{(start + i)}': bit for i, bit in enumerate(coils.bits) if i < count}

    async def _get_t(self, start: int, end: int | None) -> dict | bool:
        """Read T addresses.

        T entries start at 45056 (45057 in the Click software's 1-indexed
        notation) and span a total of 500 bits, ending at 45555.

        The response always returns a full byte of data. If you request
        a number of addresses not divisible by 8, it will have extra data. The
        extra data here is discarded before returning.
        """
        if start < 1 or start > 500:
            raise ValueError('T start address must be 1-500.')

        start_coil = 45057 + start - 1
        if end is None:
            coils = await self.read_coils(start_coil, 1)
            return coils.bits[0]

        if end <= start or end > 500:
            raise ValueError('T end address must be >start and <=500.')
        end_coil = 14555 + end - 1
        count = end_coil - start_coil + 1
        coils = await self.read_coils(start_coil, count)
        return {f't{(start + i)}': bit for i, bit in enumerate(coils.bits) if i < count}

    async def _get_ct(self, start: int, end: int | None) -> dict | bool:
        """Read CT addresses.

        CT entries start at 49152 (49153 in the Click software's 1-indexed
        notation) and span a total of 250 bits, ending at 49401.

        The response always returns a full byte of data. If you request
        a number of addresses not divisible by 8, it will have extra data. The
        extra data here is discarded before returning.
        """
        if start < 1 or start > 250:
            raise ValueError('CT start address must be 1-250.')

        start_coil = 49152 + start - 1
        if end is None:
            coils = await self.read_coils(start_coil, 1)
            return coils.bits[0]
        else:
            if end <= start or end > 250:
                raise ValueError('CT end address must be >start and <=250.')
            end_coil = 49401 + end - 1
            count = end_coil - start_coil + 1

        coils = await self.read_coils(start_coil, count)
        return {f'ct{(start + i)}': bit for i, bit in enumerate(coils.bits) if i < count}

    async def _get_sc(self, start: int, end: int | None) -> dict | bool:
        """Read SC addresses. Called by `get`.

        SC entries start at 61440 (61441 in the Click software's 1-indexed
        notation) and span a total of 1000 bits, ending at 62439.

        Args:
            start: Starting SC address (1-indexed as per ClickPLC).
            end: Optional ending SC address (inclusive, 1-indexed).

        Returns:
            A dictionary of SC values if `end` is provided, or a single bool
            value if `end` is None.

        Raises:
            ValueError: If the start or end address is out of range or invalid.
        """
        if start < 1 or start > 1000:
            raise ValueError('SC start address must be in [1, 1000]')
        if end is not None and (end <= start or end > 1000):
            raise ValueError("SC end address must be >= start and <= 1000.")

        start_coil = 61440 + (start - 1)  # Modbus coil address for SC
        if end is None:
            # Read a single coil
            return (await self.read_coils(start_coil, 1)).bits[0]

        end_coil = 61440 + (end - 1)
        count = end_coil - start_coil + 1
        coils = await self.read_coils(start_coil, count)
        return {f'sc{start + i}': bit for i, bit in enumerate(coils.bits) if i < count}

    async def _get_ds(self, start: int, end: int | None) -> dict | int:
        """Read DS registers. Called by `get`.

        DS entries start at Modbus address 0 (1 in the Click software's
        1-indexed notation). Each DS entry takes 16 bits.
        """
        if start < 1 or start > 4500:
            raise ValueError('DS must be in [1, 4500]')
        if end is not None and (end < 1 or end > 4500):
            raise ValueError('DS end must be in [1, 4500]')

        address = 0 + start - 1
        count = 1 if end is None else (end - start + 1)
        registers = await self.read_registers(address, count)
        # pack all as unsigned 16-bit little-endian and then unpack as signed 16-bit ints
        packed = struct.pack(f'<{count}H', *registers)
        values = struct.unpack(f'<{count}h', packed)
        if count == 1:
            return values[0]
        return {f'ds{start + i}': v for i, v in enumerate(values)}

    async def _get_dd(self, start: int, end: int | None) -> dict | int:
        if start < 1 or start > 1000:
            raise ValueError('DD must be in [1, 1000]')
        if end is not None and (end < 1 or end > 1000):
            raise ValueError('DD end must be in [1, 1000]')

        address = 16384 + 2 * (start - 1)
        count = 2 if end is None else 2 * (end - start + 1)
        registers = await self.read_registers(address, count)

        # Pack registers as 16-bit unsigned shorts, little-endian ('<'), then unpack as signed 32-bit ints
        packed = struct.pack(f'<{count}H', *registers)
        values = struct.unpack(f'<{count // 2}i', packed)  # 'i' = signed 32-bit int

        if count == 2:
            return values[0]
        return {f'dd{start + i}': v for i, v in enumerate(values)}


    async def _get_dh(self, start: int, end: int | None) -> dict | int:
        if start < 1 or start > 500:
            raise ValueError('DH must be in [1, 500]')
        if end is not None and (end < 1 or end > 500):
            raise ValueError('DH end must be in [1, 500]')

        address = 24576 + start - 1
        count = 1 if end is None else (end - start + 1)
        registers = await self.read_registers(address, count)

        if count == 1:
            return int(registers[0])  # unsigned 16-bit int can just cast with int()
        return {f'dh{start + n}': int(v) for n, v in enumerate(registers)}


    async def _get_df(self, start: int, end: int | None) -> dict | float:
        """Read DF registers. Called by `get`.

        DF entries start at Modbus address 28672 (28673 in the Click software's
        1-indexed notation). Each DF entry takes 32 bits, or 2 16-bit
        registers.
        """
        if start < 1 or start > 500:
            raise ValueError('DF must be in [1, 500]')
        if end is not None and (end < 1 or end > 500):
            raise ValueError('DF end must be in [1, 500]')

        address = 28672 + 2 * (start - 1)
        count = 2 * (1 if end is None else (end - start + 1))
        registers = await self.read_registers(address, count)
        # pack the pairs of 16-bit registers (little-endian) and then unpack as 32-byte floats
        packed = struct.pack(f'<{count}H', *registers)
        values = struct.unpack(f'<{count // 2}f', packed)
        if count == 2:
            return values[0]
        return {f'df{start + n}': v
                for n, v in enumerate(values)}

    async def _get_td(self, start: int, end: int | None) -> dict | int:
        """Read TD registers. Called by `get`.

        TD entries start at Modbus address 45056 (45057 in the Click software's
        1-indexed notation). Each TD entry takes 16 bits.
        """
        if start < 1 or start > 500:
            raise ValueError('TD must be in [1, 500]')
        if end is not None and (end < 1 or end > 500):
            raise ValueError('TD end must be in [1, 500]')

        address = 45056 + (start - 1)
        count = 1 if end is None else (end - start + 1)
        registers = await self.read_registers(address, count)

        # pack all as unsigned 16-bit little-endian and then unpack as signed 16-bit ints
        packed = struct.pack(f'<{count}H', *registers)
        values = struct.unpack(f'<{count}h', packed)
        if count == 1:
            return values[0]
        return {f'td{start + i}': v for i, v in enumerate(values)}


    async def _get_ctd(self, start: int, end: int | None) -> dict:
        """Read CTD registers. Called by `get`.

        CTD entries start at Modbus address 449152 (449153 in the Click software's
        1-indexed notation). Each CTD entry takes 32 bits, which is 2 16bit registers.
        """
        if start < 1 or start > 250:
            raise ValueError('CTD must be in [1, 250]')
        if end is not None and (end < 1 or end > 250):
            raise ValueError('CTD end must be in [1, 250]')

        address = 49152 + 2 * (start - 1)  # 32-bit
        count = 2 if end is None else 2 * (end - start + 1)
        registers = await self.read_registers(address, count)

        # pack the pairs of 16-bit registers (little-endian) and then unpack as 32-byte signed ints
        print(registers, count)
        packed = struct.pack(f'<{count}H', *registers)
        values = struct.unpack(f'<{count // 2}i', packed)
        return {f'ctd{start + n}': v for n, v in enumerate(values)}

    async def _get_sd(self, start: int, end: int | None) -> dict | int:
        """Read SD registers. Called by `get`.

        SD entries start at Modbus address 361440 (361441 in the Click software's
        1-indexed notation). Each SD entry takes 16 bits.
        """
        if start < 1 or start > 1000:
            raise ValueError('SD must be in [1, 1000]')
        if end is not None and (end < 1 or end > 1000):
            raise ValueError('SD end must be in [1, 1000]')

        address = 61440 + start - 1
        count = 1 if end is None else (end - start + 1)
        registers = await self.read_registers(address, count)

        if count == 1 and end is None:
            return int(registers[0])  # unsigned 16-bit int can just cast with int()
        return {f'sd{start + n}': int(v) for n, v in enumerate(registers)}

    @overload
    async def _get_txt(self, start: int, end: None) -> str: ...
    @overload
    async def _get_txt(self, start: int, end: int) -> dict[str, str]: ...
    async def _get_txt(self, start, end):
        """Read txt registers. Called by `get`.

        TXT entries start at Modbus address 36864 (36865 in the Click software's
        1-indexed notation). Each TXT entry takes 8 bits - which means a pair of
        adjacent TXT entries are packed into a single 16-bit register.  For some
        strange reason they are packed little-endian, so the registers must be
        manually decoded.
        """
        if start < 1 or start > 1000:
            raise ValueError('TXT must be in [1, 1000]')
        if end is not None and (end < 1 or end > 1000):
            raise ValueError('TXT end must be in [1, 1000]')

        address = 36864 + (start - 1) // 2
        if end is None:
            registers = await self.read_registers(address, 1)
            r = registers[0]
            assert isinstance(r, int)
            if start % 2:
                return chr(r & 0x00FF)  # if starting on the second byte of a 16-bit register, discard the MSB
            return chr((r >> 8) & 0x00FF)  # otherwise discard LSB

        count = 1 + (end - start) // 2 + (start - 1) % 2
        registers = await self.read_registers(address, count)

        # Swap the two bytes within each 16-bit register (i.e., 0x4231 -> 0x3142)
        swapped = [((reg & 0xFF) << 8) | (reg >> 8) for reg in registers]
        byte_data = b''.join(reg.to_bytes(2, 'big') for reg in swapped)
        r = byte_data.decode('ascii')
        if end % 2:  # if ending on the first byte of a 16-bit register, discard the final LSB
            r = r[:-1]
        if not start % 2:
            r = r[1:]  # if starting on the last byte of a 16-bit register, discard the first MSB
        return {f'txt{start}-txt{end}': r}

    async def _set_y(self, start: int, data: list[bool]):
        """Set Y addresses. Called by `set`.

        For more information on the quirks of Y coils, read the `_get_y`
        docstring.
        """
        if start % 100 == 0 or start % 100 > 16:
            raise ValueError('Y start address must be *01-*16.')
        if start < 1 or start > 816:
            raise ValueError('Y start address must be in [001, 816].')
        coil = 8192 + 32 * (start // 100) + start % 100 - 1

        if len(data) > 16 * (9 - start // 100) - start % 100 + 1:
            raise ValueError('Data list longer than available addresses.')
        values = []
        if (start % 100) + len(data) > 16:
            i = 17 - (start % 100)
            values += data[:i] + [False] * 16
            data = data[i:]
        while len(data) > 16:
            values += data[:16] + [False] * 16
            data = data[16:]
        values += data
        await self.write_coils(coil, values)

    async def _set_c(self, start: int, data: list[bool]):
        """Set C addresses. Called by `set`.

        For more information on the quirks of C coils, read the `_get_c`
        docstring.
        """
        if start < 1 or start > 2000:
            raise ValueError('C start address must be 1-2000.')
        coil = 16384 + start - 1

        if len(data) > (2000 - start + 1):
            raise ValueError('Data list longer than available addresses.')
        await self.write_coils(coil, data)

    async def _set_sc(self, start: int, data: list[bool]):
        """Set SC addresses. Called by `set`.

        SC entries start at 61440 (61441 in the Click software's 1-indexed
        notation). This continues for 1000 bits.

        Args:
            start: Starting SC address (1-indexed as per ClickPLC).
            data: List of values to set.

        Raises:
            ValueError: If the start address is out of range or is not writable,
                or if the data list exceeds the allowed writable range.

        Notes:
            Only the following SC addresses are writable:
            SC50, SC51, SC53, SC55, SC60, SC61, SC65, SC66, SC67, SC75, SC76, SC120, SC121.
            (SC50 and SC51 may actually be read-only!)
        """
        writable_sc_addresses = (
            50,   # _PLC_Mode_Change_to_STOP - FIXME: may not be writeable
            51,   # _Watchdog_Timer_Reset - FIXME: may not be writeable
            53,   # _RTC_Date_Change
            55,   # _RTC_Time_Change
            60,   # _BT_Disable_Pairing  (Plus only?)
            61,   # _BT_Activate_Pairing (Plus only?)
            65,   # _SD_Eject
            66,   # _SD_Delete_All
            67,   # _SD_Copy_System
            75,   # _WLAN_Reset (Plus only?)
            76,   # _Sub_CPU_Reset,
            120,  # _Network_Time_Request
            121,  # _Network_Time_DST
        )

        if start < 1 or start > 1000:
            raise ValueError('SC start address must be in [1, 1000]')
        if len(data) > 1000 - start + 1:
            raise ValueError('Data list longer than available SC addresses.')
        for i in range(len(data)):
            if (start + i) not in writable_sc_addresses:
                raise ValueError(f"SC{start + i} is not writable.")

        coil = 61440 + (start - 1)

        await self.write_coils(coil, data)

    async def _set_df(self, start: int, data: list[float]):
        """Set DF registers. Called by `set`.

        The ClickPLC is little endian, but on registers ("words") instead
        of bytes. As an example, take a random floating point number:
            Input: 0.1
            Hex: 3dcc cccd (IEEE-754 float32)
            Click: -1.076056E8
            Hex: cccd 3dcc
        To fix, we need to convert the floats into pairs of unsigned ints, which have the same
        bytes as the float.
        """
        if start < 1 or start > 500:
            raise ValueError('DF must be in [1, 500]')
        address = 28672 + 2 * (start - 1)

        if len(data) > 500 - start + 1:
            raise ValueError('Data list longer than available addresses.')

        values: list[bytes] = []
        for datum in data:
            packed_4_bytes = struct.pack('<f', datum)  # Little-endian single-precision float
            values.extend(struct.unpack('<HH', packed_4_bytes))  # unpack 2x uint_16

        await self.write_registers(address, values)

    async def _set_ds(self, start: int, data: list[int]):
        """Set DS registers. Called by `set`.

        See _get_ds for more information.
        """
        if start < 1 or start > 4500:
            raise ValueError('DS must be in [1, 4500]')
        address = (start - 1)
        if len(data) > 4500 - start + 1:
            raise ValueError('Data list longer than available addresses.')

        # since pymodbus is expecting list[uint_16], cast from int_16
        values = [d & 0xffff for d in data]  # two's complement

        await self.write_registers(address, values)

    async def _set_dd(self, start: int, data: list[int]):
        """Set DD registers. Called by `set`.

        See _get_dd for more information.
        """
        if start < 1 or start > 1000:
            raise ValueError('DD must be in [1, 1000]')
        address = 16384 + 2 * (start - 1)

        if len(data) > 1000 - start + 1:
            raise ValueError('Data list longer than available addresses.')

        # pymodbus is expecting list[uint_16]
        # convert each 32-bit signed int into a uint_16 pair (little-endian) with the same byte value
        values: list[bytes] = []
        for datum in data:
            packed_4_bytes = struct.pack('<i', datum)  # pack int_32
            values.extend(struct.unpack('<HH', packed_4_bytes))  # unpack 2x uint_16

        await self.write_registers(address, values)

    async def _set_dh(self, start: int, data: list[int]):
        """Set DH registers. Called by `set`.

        See _get_dh for more information.
        """
        if start < 1 or start > 500:
            raise ValueError('DH must be in [1, 500]')
        address = 24576 + (start - 1)

        if len(data) > 500 - start + 1:
            raise ValueError('Data list longer than available addresses.')
        await self.write_registers(address, values=data)

    async def _set_td(self, start: int, data: list[int]):
        """Set TD registers. Called by `set`.

        See _get_td for more information.
        """
        if start < 1 or start > 500:
            raise ValueError('TD must be in [1, 500]')
        address = 45056 + (start - 1)

        if len(data) > 500 - start + 1:
            raise ValueError('Data list longer than available addresses.')

        # since pymodbus is expecting list[uint_16], cast from int_16
        values = [d & 0xffff for d in data]  # two's complement

        await self.write_registers(address, values)

    async def _set_sd(self, start: int, data: list[int]):
        """Set writable SD registers. Called by `set`.

        SD entries start at Modbus address 61440 (61441 in the Click software's
        1-indexed notation). Each SD entry takes 16 bits.

        Args:
            start: Starting SD address (1-indexed as per ClickPLC).
            data: Single value or list of values to set.

        Raises:
            ValueError: If an address is not writable or if data list exceeds the
                allowed writable range.

        Notes:
            Only the following SD addresses are writable:
            SD29, SD31, SD32, SD34, SD35, SD36, SD40, SD41, SD42, SD50,
            SD51, SD60, SD61, SD106, SD107, SD108, SD112, SD113, SD114,
            SD140, SD141, SD142, SD143, SD144, SD145, SD146, SD147,
            SD214, SD215
        """
        writable_sd_addresses = (
            29, 31, 32, 34, 35, 36, 40, 41, 42, 50, 51, 60, 61, 106, 107, 108,
            112, 113, 114, 140, 141, 142, 143, 144, 145, 146, 147, 214, 215
        )

        def validate_address(address: int):
            if address not in writable_sd_addresses:
                raise ValueError(f"SD{address} is not writable. Only specific SD registers are writable.")
        for idx, _ in enumerate(data):
            validate_address(start + idx)

        address = 61440 + (start - 1)

        if len(data) > len(writable_sd_addresses):
            raise ValueError('Data list contains more elements than writable SD registers.')

        # since pymodbus is expecting list[uint_16], cast from int_16
        values = [d & 0xffff for d in data]  # two's complement

        await self.write_registers(address, values)

    async def _set_txt(self, start: int, data: list[str]):
        """Set TXT registers. Called by `set`.

        See _get_txt for more information.
        """
        if start < 1 or start > 1000:
            raise ValueError('TXT must be in [1, 1000]')
        address = 36864 + (start - 1) // 2

        if len(data) > 1000 - start + 1:
            raise ValueError('Data list longer than available addresses.')
        string = data[0]

        # two 8-bit text addresses are packed into one 16-bit modbus register
        # and we can't mask a modbus write (i.e. all 16 bits must be written)
        # thus, if changing a single address retrieve and write its 'twin' address back
        if len(string) % 2:
            if start % 2:
                string += await self._get_txt(start=start + 1, end=None)
            else:
                string = await self._get_txt(start=start - 1, end=None) + string

        # every 2 8-bit characters become one 16-bit modbus register
        values = [value for value,                               # note the comma to index the tuple
                  in struct.iter_unpack('<H', string.encode())]  # '<H' is little-endian uint_16

        await self.write_registers(address, values)

    def _load_tags(self, tag_filepath: str) -> dict:
        """Load tags from file path.

        This tag file is optional but is needed to identify the appropriate variable names,
        and modbus addresses for tags in use on the PLC.

        """
        if not tag_filepath:
            return {}
        with open(tag_filepath) as csv_file:
            csv_data = csv_file.read().splitlines()
        csv_data[0] = csv_data[0].lstrip('## ')
        parsed: dict[str, dict[str, Any]] = {
            row['Nickname']: {
                'address': {
                    'start': int(row['Modbus Address']),
                },
                'id': row['Address'],
                'comment': row['Address Comment'],
                'type': self.data_types.get(
                    row['Address'].rstrip(digits).lower()
                ),
            }
            for row in csv.DictReader(csv_data)
            if row['Nickname'] and not row['Nickname'].startswith("_")
        }
        for data in parsed.values():
            if not data['comment']:
                del data['comment']
            if not data['type']:
                raise TypeError(
                    f"{data['id']} is an unsupported data type. Open a "
                    "github issue at numat/clickplc to get it added."
                )
        sorted_tags = {k: parsed[k] for k in
                       sorted(parsed, key=lambda k: parsed[k]['address']['start'])}
        return sorted_tags

    @staticmethod
    def _get_address_ranges(tags: dict) -> dict[str, dict]:
        """Determine range of addresses required.

        Parse the loaded tags to determine the range of addresses that must be
        queried to return all values
        """
        address_dict: dict = defaultdict(lambda: {'min': 1, 'max': 1})
        for tag_info in tags.values():
            i = next(i for i, s in enumerate(tag_info['id']) if s.isdigit())
            category, index = tag_info['id'][:i].lower(), int(tag_info['id'][i:])
            address_dict[category]['min'] = min(address_dict[category]['min'], index)
            address_dict[category]['max'] = max(address_dict[category]['max'], index)
        return address_dict
