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
from collections import defaultdict
from string import digits
from typing import Any, ClassVar, overload

from pymodbus.constants import Endian
from pymodbus.payload import BinaryPayloadBuilder, BinaryPayloadDecoder

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
        self.bigendian = Endian.BIG if self.pymodbus35plus else Endian.Big  # type:ignore[attr-defined]
        self.lilendian = Endian.LITTLE if self.pymodbus35plus else Endian.Little  # type:ignore[attr-defined]
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

        if end_index is not None and end_index < start_index:
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
        notation). This continues for 2000 bits, ending at 18383.

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
        notation). This continues for 500 bits, ending at 45555.

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
        notation). This continues for 250 bits, ending at 49402.

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
        decoder = BinaryPayloadDecoder.fromRegisters(registers,
                                                     byteorder=self.bigendian,
                                                     wordorder=self.lilendian)
        if end is None:
            return decoder.decode_16bit_int()
        return {f'ds{n}': decoder.decode_16bit_int() for n in range(start, end + 1)}

    async def _get_dd(self, start: int, end: int | None) -> dict | int:
        """Read DD registers.

        DD entries start at Modbus address 16384 (16385 in the Click software's
        1-indexed notation). Each DS entry takes 32 bits.
        """
        if start < 1 or start > 1000:
            raise ValueError('DD must be in [1, 1000]')
        if end is not None and (end < 1 or end > 1000):
            raise ValueError('DD end must be in [1, 1000]')

        address = 16384 + 2 * (start - 1)  # 32-bit
        count = 2 if end is None else 2 * (end - start + 1)
        registers = await self.read_registers(address, count)
        decoder = BinaryPayloadDecoder.fromRegisters(registers,
                                                     byteorder=self.bigendian,
                                                     wordorder=self.lilendian)
        if end is None:
            return decoder.decode_32bit_int()
        return {f'dd{n}': decoder.decode_32bit_int() for n in range(start, end + 1)}

    async def _get_dh(self, start: int, end: int | None) -> dict | int:
        """Read DH registers.

        DH entries start at Modbus address 24576 (24577 in the Click software's
        1-indexed notation). Each DH entry takes 16 bits.
        """
        if start < 1 or start > 500:
            raise ValueError('DH must be in [1, 500]')
        if end is not None and (end < 1 or end > 500):
            raise ValueError('DH end must be in [1, 500]')

        address = 24576 + start - 1
        count = 1 if end is None else (end - start + 1)
        registers = await self.read_registers(address, count)
        decoder = BinaryPayloadDecoder.fromRegisters(registers,
                                                     byteorder=self.bigendian,
                                                     wordorder=self.lilendian)
        if end is None:
            return decoder.decode_16bit_uint()
        return {f'dh{n}': decoder.decode_16bit_uint() for n in range(start, end + 1)}

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
        decoder = BinaryPayloadDecoder.fromRegisters(registers,
                                                     byteorder=self.bigendian,
                                                     wordorder=self.lilendian)
        if end is None:
            return decoder.decode_32bit_float()
        return {f'df{n}': decoder.decode_32bit_float() for n in range(start, end + 1)}

    async def _get_td(self, start: int, end: int | None) -> dict:
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
        decoder = BinaryPayloadDecoder.fromRegisters(registers,
                                                     byteorder=self.bigendian,
                                                     wordorder=self.lilendian)
        if end is None:
            return decoder.decode_16bit_int()
        return {f'td{n}': decoder.decode_16bit_int() for n in range(start, end + 1)}

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
        count = 1 if end is None else (end - start + 1)
        registers = await self.read_registers(address, count * 2)
        decoder = BinaryPayloadDecoder.fromRegisters(registers,
                                                     byteorder=self.bigendian,
                                                     wordorder=self.lilendian)
        if end is None:
            return decoder.decode_32bit_int()
        return {f'ctd{n}': decoder.decode_32bit_int() for n in range(start, end + 1)}

    async def _get_sd(self, start: int, end: int | None) -> dict | int:
        """Read SD registers. Called by `get`.

        SD entries start at Modbus address 361440 (361441 in the Click software's
        1-indexed notation). Each SD entry takes 16 bits.
        """
        if start < 1 or start > 4500:
            raise ValueError('SD must be in [1, 4500]')
        if end is not None and (end < 1 or end > 4500):
            raise ValueError('SD end must be in [1, 4500]')

        address = 61440 + start - 1
        count = 1 if end is None else (end - start + 1)
        registers = await self.read_registers(address, count)
        decoder = BinaryPayloadDecoder.fromRegisters(registers,
                                                     byteorder=self.bigendian,
                                                     wordorder=self.lilendian)
        if end is None:
            return decoder.decode_16bit_int()
        return {f'sd{n}': decoder.decode_16bit_int() for n in range(start, end + 1)}

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
            decoder = BinaryPayloadDecoder.fromRegisters(registers)
            if start % 2:  # if starting on the second byte of a 16-bit register, discard the MSB
                decoder.decode_string()
            return decoder.decode_string().decode()

        count = 1 + (end - start) // 2 + (start - 1) % 2
        registers = await self.read_registers(address, count)
        decoder = BinaryPayloadDecoder.fromRegisters(registers)  # endian irrelevant; manual decode
        r = ''
        for _ in range(count):
            msb = chr(decoder.decode_8bit_int())
            lsb = chr(decoder.decode_8bit_int())
            r += lsb + msb
        if end % 2:  # if ending on the first byte of a 16-bit register, discard the final LSB
            r = r[:-1]
        if not start % 2:
            r = r[1:]  # if starting on the last byte of a 16-bit register, discard the first MSB
        return {f'txt{start}-txt{end}': r}

    async def _set_y(self, start: int, data: list[bool] | bool):
        """Set Y addresses. Called by `set`.

        For more information on the quirks of Y coils, read the `_get_y`
        docstring.
        """
        if start % 100 == 0 or start % 100 > 16:
            raise ValueError('Y start address must be *01-*16.')
        if start < 1 or start > 816:
            raise ValueError('Y start address must be in [001, 816].')
        coil = 8192 + 32 * (start // 100) + start % 100 - 1

        if isinstance(data, list):
            if len(data) > 16 * (9 - start // 100) - start % 100 + 1:
                raise ValueError('Data list longer than available addresses.')
            payload = []
            if (start % 100) + len(data) > 16:
                i = 17 - (start % 100)
                payload += data[:i] + [False] * 16
                data = data[i:]
            while len(data) > 16:
                payload += data[:16] + [False] * 16
                data = data[16:]
            payload += data
            await self.write_coils(coil, payload)
        else:
            await self.write_coil(coil, data)

    async def _set_c(self, start: int, data: list[bool] | bool):
        """Set C addresses. Called by `set`.

        For more information on the quirks of C coils, read the `_get_c`
        docstring.
        """
        if start < 1 or start > 2000:
            raise ValueError('C start address must be 1-2000.')
        coil = 16384 + start - 1

        if isinstance(data, list):
            if len(data) > (2000 - start + 1):
                raise ValueError('Data list longer than available addresses.')
            await self.write_coils(coil, data)
        else:
            await self.write_coil(coil, data)

    async def _set_df(self, start: int, data: list[float] | float):
        """Set DF registers. Called by `set`.

        The ClickPLC is little endian, but on registers ("words") instead
        of bytes. As an example, take a random floating point number:
            Input: 0.1
            Hex: 3dcc cccd (IEEE-754 float32)
            Click: -1.076056E8
            Hex: cccd 3dcc
        To fix, we need to flip the registers. Implemented below in `pack`.
        """
        if start < 1 or start > 500:
            raise ValueError('DF must be in [1, 500]')
        address = 28672 + 2 * (start - 1)

        def _pack(values: list[float]):
            builder = BinaryPayloadBuilder(byteorder=self.bigendian,
                                           wordorder=self.lilendian)
            for value in values:
                builder.add_32bit_float(float(value))
            return builder.build()

        if isinstance(data, list):
            if len(data) > 500 - start + 1:
                raise ValueError('Data list longer than available addresses.')
            payload = _pack(data)
            await self.write_registers(address, payload, skip_encode=True)
        else:
            await self.write_register(address, _pack([data]), skip_encode=True)

    async def _set_ds(self, start: int, data: list[int] | int):
        """Set DS registers. Called by `set`.

        See _get_ds for more information.
        """
        if start < 1 or start > 4500:
            raise ValueError('DS must be in [1, 4500]')
        address = (start - 1)

        def _pack(values: list[int]):
            builder = BinaryPayloadBuilder(byteorder=self.bigendian,
                                           wordorder=self.lilendian)
            for value in values:
                builder.add_16bit_int(int(value))
            return builder.build()

        if isinstance(data, list):
            if len(data) > 4500 - start + 1:
                raise ValueError('Data list longer than available addresses.')
            payload = _pack(data)
            await self.write_registers(address, payload, skip_encode=True)
        else:
            await self.write_register(address, _pack([data]), skip_encode=True)

    async def _set_dd(self, start: int, data: list[int] | int):
        """Set DD registers. Called by `set`.

        See _get_dd for more information.
        """
        if start < 1 or start > 1000:
            raise ValueError('DD must be in [1, 1000]')
        address = 16384 + 2 * (start - 1)

        def _pack(values: list[int]):
            builder = BinaryPayloadBuilder(byteorder=self.bigendian,
                                           wordorder=self.lilendian)
            for value in values:
                builder.add_32bit_int(int(value))
            return builder.build()

        if isinstance(data, list):
            if len(data) > 1000 - start + 1:
                raise ValueError('Data list longer than available addresses.')
            payload = _pack(data)
            await self.write_registers(address, payload, skip_encode=True)
        else:
            await self.write_register(address, _pack([data]), skip_encode=True)

    async def _set_dh(self, start: int, data: list[int] | int):
        """Set DH registers. Called by `set`.

        See _get_dh for more information.
        """
        if start < 1 or start > 500:
            raise ValueError('DH must be in [1, 500]')
        address = 24576 + (start - 1)

        def _pack(values: list[int]):
            builder = BinaryPayloadBuilder(byteorder=self.bigendian,
                                           wordorder=self.lilendian)
            for value in values:
                builder.add_16bit_uint(int(value))
            return builder.build()

        if isinstance(data, list):
            if len(data) > 500 - start + 1:
                raise ValueError('Data list longer than available addresses.')
            payload = _pack(data)
            await self.write_registers(address, payload, skip_encode=True)
        else:
            await self.write_register(address, _pack([data]), skip_encode=True)


    async def _set_td(self, start: int, data: list[int] | int):
        """Set TD registers. Called by `set`.

        See _get_td for more information.
        """
        if start < 1 or start > 500:
            raise ValueError('TD must be in [1, 500]')
        address = 45056 + (start - 1)

        def _pack(values: list[int]):
            builder = BinaryPayloadBuilder(byteorder=self.bigendian,
                                           wordorder=self.lilendian)
            for value in values:
                builder.add_16bit_int(int(value))
            return builder.build()

        if isinstance(data, list):
            if len(data) > 500 - start + 1:
                raise ValueError('Data list longer than available addresses.')
            payload = _pack(data)
            await self.write_registers(address, payload, skip_encode=True)
        else:
            await self.write_register(address, _pack([data]), skip_encode=True)

    async def _set_txt(self, start: int, data: str | list[str]):
        """Set TXT registers. Called by `set`.

        See _get_txt for more information.
        """
        if start < 1 or start > 1000:
            raise ValueError('TXT must be in [1, 1000]')
        address = 36864 + (start - 1) // 2

        def _pack(values: str):
            assert len(values) % 2 == 0
            builder = BinaryPayloadBuilder()  # endianness irrelevant; manual packing
            for i in range(0, len(values), 2):
                builder.add_8bit_uint(ord(values[i + 1]))
                builder.add_8bit_uint(ord(values[i + 0]))
            return builder.build()

        assert isinstance(data, list)
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
        payload = _pack(string)
        await self.write_registers(address, payload, skip_encode=True)

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
