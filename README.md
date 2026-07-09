clickplc
========

Python driver and command-line tool for [Koyo Ethernet ClickPLCs](https://www.automationdirect.com/clickplcs).

<p align="center">
  <img src="https://www.automationdirect.com/microsites/clickplcs/images/expandedclick.jpg" />
</p>

Installation
============

```
uv pip install clickplc
```

Usage
=====

### Command Line

```
$ clickplc the-plc-ip-address
```

This will print all the X, Y, DS, and DF registers to stdout as JSON. You can pipe
this as needed. However, you'll likely want the python functionality below.

### Python

This uses Python ≥3.5's async/await syntax to asynchronously communicate with
a ClickPLC via TCP (ethernet) or Serial connection. For example:

```python
import asyncio
from clickplc import ClickPLC

async def foo():
    async with ClickPLC('the-plc-ip-address') as plc:
        print(await plc.get('df1-df500'))

asyncio.run(foo())
```

Additionally, you can use the same functionality over Serial connection:
```python
import asyncio
from clickplc import ClickPLC

async def bar():
    async with ClickPLC('the-com-port') as plc:
        print(await plc.get('df1-df500'))

asyncio.run(bar())
```

The entire API is `get` and `set`, and takes a range of inputs:

```python
>>> await plc.get('df1')
0.0
>>> await plc.get('df1-df20')
{'df1': 0.0, 'df2': 0.0, ..., 'df20': 0.0}
>>> await plc.get('y101-y316')
{'y101': False, 'y102': False, ..., 'y316': False}

>>> await plc.set('df1', 0.0)  # Sets DF1 to 0.0
>>> await plc.set('df1', [0.0, 0.0, 0.0])  # Sets DF1-DF3 to 0.0.
>>> await plc.set('y101', True)  # Sets Y101 to true
```

All of the following datatypes are supported:

|     |        |                                          |    |
|-----|--------|------------------------------------------|----|
| x   | bool   | Input point                              | R  |
| y   | bool   | Output point                             | RW |
| c   | bool   | (C)ontrol relay                          | RW |
| t   | bool   | (T)imer                                  | R  |
| ct  | bool   | (C)oun(t)er                              | R  |
| ds  | int16  | (D)ata register, (s)ingle signed int     | RW |
| dd  | int32  | (D)ata register, (d)double signed int    | RW |
| dh  | uint16 | (D) register, (h)ex                      | RW |
| df  | float  | (D)ata register, (f)loating point        | RW |
| td  | int16  | (T)ime (d)elay register                  | RW |
| ctd | int32  | (C)oun(t)er Current Values, (d)ouble int | RW |
| sd  | int16  | (S)ystem (D)ata register                 | R* |
| txt | char   | (T)e(xt)                                 | RW |

`*` Note: Only certain System Data registers are writeable.

### Tags / Nicknames

Recent ClickPLC software provides the ability to export a "tags file", which
contains all variables with user-assigned nicknames. The tags file can be used
with this driver to improve code readability. (Who really wants to think about
modbus addresses and register/coil types?)

To export a tags file, open the ClickPLC software, go to the Address Picker,
select "Display MODBUS address", and export the file.

Once you have this file, simply pass the file path to the driver. You can now
`set` variables by name and `get` all named variables by default.

```python
async with ClickPLC('the-plc-ip-address', 'path-to-tags.csv') as plc:
    await plc.set('my-nickname', True)  # Set variable by nickname
    print(await plc.get())  # Get all named variables in tags file
```

Additionally, the tags file can be used with the commandline tool to provide more informative output:
```
$ clickplc the-plc-ip-address tags-filepath
```

### Warning with Serial Connection
If you're using Serial connection (RS-232, maybe RS-485?) to communicate with the PLC, you **cannot** have something else running on your machine accessing that port. Maybe this is obvious to someone with further experience with Serial, but if you are running the Click Programming Software on that Serial port, you cannot also access the PLC with this library on that same Serial port - which is not how a TCP connection works. You can monitor values with the Click Programming Software on a TCP connection while connecting with this library on the same TCP connection.

### Further Documentation
If you want further documentation about how this library works, see [this guide](docs.md).