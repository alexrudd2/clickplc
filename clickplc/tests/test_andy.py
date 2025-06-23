"""This module will just test by connecting directly to the Click Plus PLC."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from driver import ClickPLC
import asyncio
import random

ADDRESS_IP = "192.168.1.2"

async def all_tests():
    async with ClickPLC(ADDRESS_IP, interfacetype="TCP") as plc:
        # y tests
        for _hundred in range(0, 9):
            random_16_bools = []
            for _ in range(16):
                random_16_bools.append(random.randint(0, 1) == 0)

            await plc.set(f'y{_hundred}01', random_16_bools)

            plc_bools = await plc.get(f'y{_hundred}01-y{_hundred}16')
            plc_bools_real = [plc_bools[x] for x in plc_bools]

            assert random_16_bools == plc_bools_real

asyncio.run(all_tests())