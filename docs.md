### PyModbus
This library is built on top of the python library `pymodbus`. The default connection of this library uses the internet protocol [TCP](https://en.wikipedia.org/wiki/Transmission_Control_Protocol). Connecting an Ethernet cable to the Click Plus PLC from whatever machine is running this library, along with inputting the IP address into the ClickPLC argument will initialize the connection with the [three way handshake](https://www.geeksforgeeks.org/computer-networks/tcp-3-way-handshake-process/).

However, you can also connect over [Serial connection](https://en.wikipedia.org/wiki/Serial_communication). The Click Plus PLC has two possible options for this: the RS-232 (Port 2) port and the RS-485 (Port 3) Port.
<p align="center">
    <img src="https://github.com/user-attachments/assets/fb474433-eff4-4fd0-9ae3-7486e3adc62e" />
</p>

Both of these ports will work with this library. Connect the ClickPLC to the computer using either RS-232 or RS-485 - just make sure you set the `interfacetype` during initialization to "Serial". See the below examples for further explanation.

### 1-Addressing
In the Click Programming Software, all of the addresses are 1-addressed. X1 is listed as address 1-00001. This library accounts for that by subtracting 1 from an address before sending it.

## Examples
Here's some examples on how to use this library efficiently:

Example 1: Set a Y address using TCP.
```python
import asyncio
from clickplc import ClickPLC

async def setY14():
    # in this example, we say that the ClickPLC's IP Address is 192.168.1.2
    async with ClickPLC("192.168.1.2", interfacetype="TCP") as plc:
        await plc.set("y14", True)

asyncio.run(setY14())
```

Example 2: Set a DD address using Serial. *(Note: You can still set a DD address using TCP. This example is just meant to show how to use Serial, as well as pass through integers as values. Nothing in this library is exclusive to TCP or Serial)*
```python
import asyncio
from clickplc import ClickPLC

async def setDD3(value: int):
    # in this example, we say that the ClickPLC is connected to the computer on the computer's COM4 port
    async with ClickPLC("COM4", interfacetype="Serial") as plc:
        await plc.set("dd3", value)

asyncio.run(setDD3(193))
```

Example 3: Check an X address. If it's `True`, set its corresponding Y address to `False`. Otherwise, don't adjust the Y address.
```python
import asyncio
from clickplc import ClickPLC

async def checkXsetY(numerical_address: int):
    async with ClickPLC("COM4", interfacetype="Serial") as plc:
        x_value = await plc.get(f"x{numerical_address}")
        if x_value is True :
            await plc.set(f"y{numerical_address}", False)

asyncio.run(checkXsetY(45))
```

Example 4: Check all DS addresses. If any of them are above the value of 231, set DS14 to -14.
```python
import asyncio
from clickplc import ClickPLC

async def funnyDS():
    async with ClickPLC("192.168.1.2", interfacetype="TCP") as plc:
        all_ds_values = await plc.get("ds1-ds4500")
        for key, value in all_ds_values.items():
            if value > 231:
                print(f"Culprit found! {key}")
                await plc.set("ds14", -14)
                break

asyncio.run(funnyDS())
```
