"""Test the driver correctly parses a tags file and responds with correct data."""
import asyncio
import contextlib
from unittest import mock

import pytest

try:
    from pymodbus.server import ModbusTcpServer
except ImportError:
    from pymodbus.server.async_io import ModbusTcpServer  # type: ignore[no-redef]

from clickplc import ClickPLC, command_line
from clickplc.mock import ClickPLC as MockClickPLC

# Test against pymodbus simulator
ADDRESS = '127.0.0.1'
autouse = True
# Uncomment below to use a real PLC
# ADDRESS = '172.16.0.168'
# autouse = False

@pytest.fixture(scope='session', autouse=autouse)
async def _sim():
    """Start a modbus server and datastore."""
    from pymodbus.datastore import (
        ModbusSequentialDataBlock,
        ModbusServerContext,
    )
    try:
        from pymodbus.datastore import ModbusDeviceContext  # 3.10
    except ImportError:
        from pymodbus.datastore import ModbusSlaveContext as ModbusDeviceContext  # type: ignore
    store = ModbusDeviceContext(
        di=ModbusSequentialDataBlock(0, [0] * 65536),  # Discrete Inputs
        co=ModbusSequentialDataBlock(0, [0] * 65536),  # Coils
        hr=ModbusSequentialDataBlock(0, [0] * 65536),  # Holding Registers
        ir=ModbusSequentialDataBlock(0, [0] * 65536)   # Input Registers
    )
    context = ModbusServerContext(store, single=True)
    server = ModbusTcpServer(context=context, address=("127.0.0.1", 5020))
    asyncio.ensure_future(server.serve_forever())  # noqa: RUF006
    await(asyncio.sleep(0))
    yield
    with contextlib.suppress(AttributeError):  # 2.x
        await server.shutdown()  # type: ignore

@pytest.fixture(scope='session')
async def plc_driver():
    """Confirm the driver correctly initializes without a tags file."""
    async with ClickPLC(ADDRESS) as c:
        yield c

@pytest.fixture
def expected_tags():
    """Return the tags defined in the tags file."""
    return {
        'IO2_24V_OK': {'address': {'start': 16397}, 'id': 'C13', 'type': 'bool'},
        'IO2_Module_OK': {'address': {'start': 16396}, 'id': 'C12', 'type': 'bool'},
        'LI_101': {'address': {'start': 428683}, 'id': 'DF6', 'type': 'float'},
        'LI_102': {'address': {'start': 428681}, 'id': 'DF5', 'type': 'float'},
        'P_101': {'address': {'start': 8289}, 'id': 'Y301', 'type': 'bool'},
        'P_101_auto': {'address': {'start': 16385}, 'id': 'C1', 'type': 'bool'},
        'P_102_auto': {'address': {'start': 16386}, 'id': 'C2', 'type': 'bool'},
        'P_103': {'address': {'start': 8290}, 'id': 'Y302', 'type': 'bool'},
        'TIC101_PID_ErrorCode': {'address': {'start': 400100},
                                 'comment': 'PID Error Code',
                                 'id': 'DS100',
                                 'type': 'int16'},
        'TI_101': {'address': {'start': 428673}, 'id': 'DF1', 'type': 'float'},
        'VAHH_101_OK': {'address': {'start': 16395}, 'id': 'C11', 'type': 'bool'},
        'VAH_101_OK': {'address': {'start': 16394}, 'id': 'C10', 'type': 'bool'},
        'VI_101': {'address': {'start': 428685}, 'id': 'DF7', 'type': 'float'},
        'PLC_Error_Code': {'address': {'start': 361441}, 'id': 'SD1', 'type': 'int16'},
        'timer': {'address': {'start': 449153}, 'id': 'CTD1', 'type': 'int32'},
    }


def test_driver_cli(capsys):
    """Confirm the commandline interface works without a tags file."""
    command_line([ADDRESS])
    captured = capsys.readouterr()
    assert 'x816' in captured.out
    assert 'c100' in captured.out
    assert 'df100' in captured.out


@mock.patch('clickplc.ClickPLC', MockClickPLC)
def test_driver_cli_tags_mock(capsys):
    """Confirm the (mocked) commandline interface works with a tags file."""
    command_line([ADDRESS, 'clickplc/tests/plc_tags.csv'])
    captured = capsys.readouterr()
    assert 'P_101' in captured.out
    assert 'VAHH_101_OK' in captured.out
    assert 'TI_101' in captured.out
    with pytest.raises(SystemExit):
        command_line([ADDRESS, 'tags', 'bogus'])


def test_driver_cli_tags(capsys):
    """Confirm the commandline interface works with a tags file."""
    command_line([ADDRESS, 'clickplc/tests/plc_tags.csv'])
    captured = capsys.readouterr()
    assert 'P_101' in captured.out
    assert 'VAHH_101_OK' in captured.out
    assert 'TI_101' in captured.out
    with pytest.raises(SystemExit):
        command_line([ADDRESS, 'tags', 'bogus'])

@mock.patch('clickplc.util.AsyncioModbusClient.__init__')
def test_unsupported_tags(mock_init):
    """Confirm the driver detects an improper tags file."""
    with pytest.raises(TypeError, match='unsupported data type'):
        ClickPLC(ADDRESS, 'clickplc/tests/bad_tags.csv')

@pytest.mark.asyncio(loop_scope='session')
async def test_tagged_driver(expected_tags):
    """Test a roundtrip with the driver using a tags file."""
    async with ClickPLC(ADDRESS, 'clickplc/tests/plc_tags.csv') as tagged_driver:
        await tagged_driver.set('VAH_101_OK', True)
        state = await tagged_driver.get()
        assert state.get('VAH_101_OK')
        assert expected_tags == tagged_driver.get_tags()

@pytest.mark.asyncio(loop_scope='session')
async def test_y_roundtrip(plc_driver):
    """Confirm y (output bools) are read back correctly after being set."""
    await plc_driver.set('y1', [False, True, False, True])
    expected = {'y001': False, 'y002': True, 'y003': False, 'y004': True}
    assert expected == await plc_driver.get('y1-y4')
    await plc_driver.set('y816', True)
    assert await plc_driver.get('y816') is True

@pytest.mark.asyncio(loop_scope='session')
async def test_c_roundtrip(plc_driver):
    """Confirm c bools are read back correctly after being set."""
    await plc_driver.set('c2', True)
    await plc_driver.set('c3', [False, True])
    expected = {'c1': False, 'c2': True, 'c3': False, 'c4': True, 'c5': False}
    assert expected == await plc_driver.get('c1-c5')
    await plc_driver.set('c2000', True)
    assert await plc_driver.get('c2000') is True

@pytest.mark.asyncio(loop_scope='session')
async def test_sc_roundtrip(plc_driver):
    """Confirm writable SC bools are read back correctly after being set."""
    # FIXME docs say this is writable, but firmware 3.60 says read-only
    # Test writing to SC50 (_PLC_Mode_Change_to_STOP) to stop PLC mode
    # await plc_driver.set('sc50', True)
    # assert await plc_driver.get('sc50') is True

    # Test writing to SC60 and SC61 (_BT_Disable_Pairing, _BT_Activate_Pairing) to
    # manage Bluetooth pairing
    await plc_driver.set('sc60', [True, False])
    expected = {'sc60': True, 'sc61': False}
    assert expected == await plc_driver.get('sc60-sc61')

    # Test writing to SC120 (_Network_Time_Request) to start an NTP request
    await plc_driver.set('sc120', True)
    assert await plc_driver.get('sc120') is True

    # Test error handling for non-writable SC62 (_BT_Paired_Devices)
    with pytest.raises(ValueError, match="SC62 is not writable"):
        await plc_driver.set('sc62', True)

@pytest.mark.asyncio(loop_scope='session')
async def test_ds_roundtrip(plc_driver):
    """Confirm ds ints are read back correctly after being set."""
    await plc_driver.set('ds1', 1)
    await plc_driver.set('ds3', [-32768, 32767])
    expected = {'ds1': 1, 'ds2': 0, 'ds3': -32768, 'ds4': 32767, 'ds5': 0}
    assert expected == await plc_driver.get('ds1-ds5')
    await plc_driver.set('ds4500', 4500)
    assert await plc_driver.get('ds4500') == 4500

@pytest.mark.asyncio(loop_scope='session')
async def test_df_roundtrip(plc_driver):
    """Confirm df floats are read back correctly after being set."""
    await plc_driver.set('df1', 0.0)
    await plc_driver.set('df2', [2.2, 3.3, 4.0, 0.0])
    expected = {'df1': 0.0, 'df2': 2.2, 'df3': 3.3, 'df4': 4.0, 'df5': 0.0}
    # python floats are 64 bits and the PLC are 32
    assert expected == pytest.approx(await plc_driver.get('df1-df5'), rel=1e-6)
    await plc_driver.set('df500', 1.0)
    assert await plc_driver.get('df500') == 1.0

@pytest.mark.asyncio(loop_scope='session')
async def test_td_roundtrip(plc_driver):
    """Confirm td ints are read back correctly after being set."""
    await plc_driver.set('td1', 1)
    await plc_driver.set('td2', [2, -32768, 32767, 0])
    expected = {'td1': 1, 'td2': 2, 'td3': -32768, 'td4': 32767, 'td5': 0}
    assert expected == await plc_driver.get('td1-td5')
    await plc_driver.set('td500', 500)
    assert await plc_driver.get('td500') == 500

@pytest.mark.asyncio(loop_scope='session')
async def test_dd_roundtrip(plc_driver):
    """Confirm dd double ints are read back correctly after being set."""
    await plc_driver.set('dd1', 1)
    await plc_driver.set('dd3', [-2**31, 2**31 - 1])
    expected = {'dd1': 1, 'dd2': 0, 'dd3': -2**31, 'dd4': 2**31 - 1, 'dd5': 0}
    assert expected == await plc_driver.get('dd1-dd5')
    assert await plc_driver.get('dd3') == -2**31
    assert await plc_driver.get('dd4') == 2**31 - 1
    await plc_driver.set('dd1000', 1000)
    assert await plc_driver.get('dd1000') == 1000

@pytest.mark.asyncio(loop_scope='session')
async def test_dh_roundtrip(plc_driver):
    """Confirm dh single ints are read back correctly after being set."""
    await plc_driver.set('dh1', 1)
    await plc_driver.set('dh3', [3, 2**16 - 1])
    expected = {'dh1': 1, 'dh2': 0, 'dh3': 3, 'dh4': 2**16 - 1, 'dh5': 0}
    assert expected == await plc_driver.get('dh1-dh5')
    await plc_driver.set('dh500', 500)
    assert await plc_driver.get('dh500') == 500

@pytest.mark.asyncio(loop_scope='session')
async def test_sd_roundtrip(plc_driver):
    """Confirm writable SD ints are read back correctly after being set."""
    # Test writing to SD112 (_EIP_Con2_LostCount) to reset lost packets counter for Ethernet/IP Connection 2
    await plc_driver.set('sd112', 0)
    assert await plc_driver.get('sd112') == 0

    # Test error handling for non-writable SD62 (_BT_Paired_Device_Count)
    with pytest.raises(ValueError, match="SD62 is not writable"):
        await plc_driver.set('sd62', 5)

@pytest.mark.asyncio(loop_scope='session')
async def test_set_date(plc_driver):
    """Test setting the date components (SD29, SD31, SD32) and triggering SC53 to update the RTC date."""
    # Set date values
    await plc_driver.set('sd29', 2024)  # Year
    await plc_driver.set('sd31', 12)   # Month
    await plc_driver.set('sd32', 25)   # Day

    # Trigger the update
    await plc_driver.set('sc53', True)
    await asyncio.sleep(1)  # Wait for the update to process

    # Confirm no errors
    assert await plc_driver.get('sc54') == 0, "SC54 indicates a date update error."

    # Turn SC53 OFF
    await plc_driver.set('sc53', False)

    # Verify date components
    assert await plc_driver.get('sd29') == 2024
    assert await plc_driver.get('sd31') == 12
    assert await plc_driver.get('sd32') == 25

@pytest.mark.asyncio(loop_scope='session')
async def test_set_time(plc_driver):
    """Test setting the time components (SD34, SD35, SD36) and triggering SC55 to update the RTC time."""
    # Set time values
    await plc_driver.set('sd34', 12)  # Hour
    await plc_driver.set('sd35', 30)  # Minute
    await plc_driver.set('sd36', 45)  # Second

    # Trigger the update
    await plc_driver.set('sc55', True)
    await asyncio.sleep(1)  # Wait for the update to process

    # Confirm no errors
    assert await plc_driver.get('sc56') == 0, "SC56 indicates a time update error."

    # Turn SC55 OFF
    await plc_driver.set('sc55', False)

    # Verify time components
    assert await plc_driver.get('sd34') == 12
    assert await plc_driver.get('sd35') == 30
    assert await plc_driver.get('sd36') == 45

@pytest.mark.asyncio(loop_scope='session')
async def test_txt_roundtrip(plc_driver):
    """Confirm texts are read back correctly after being set."""
    await plc_driver.set('txt1', 'AB')
    await plc_driver.set('txt3', 'CDEF')
    await plc_driver.set('txt7', 'G')
    expected = {'txt1-txt7': 'ABCDEFG'}
    assert expected == await plc_driver.get('txt1-txt7')
    expected = {'txt2-txt7': 'BCDEFG'}
    assert expected == await plc_driver.get('txt2-txt7')

    await plc_driver.set('txt1000', '0')
    assert await plc_driver.get('txt1000') == '0'
    await plc_driver.set('txt999', '9')
    assert await plc_driver.get('txt999') == '9'
    assert await plc_driver.get('txt1000') == '0'  # ensure txt999 did not clobber it

@pytest.mark.asyncio(loop_scope='session')
async def test_get_error_handling(plc_driver):
    """Confirm the driver gives an error on invalid get() calls."""
    with pytest.raises(ValueError, match='An address must be supplied'):
        await plc_driver.get()
    with pytest.raises(ValueError, match='End address must be greater than start address'):
        await plc_driver.get('c3-c1')
    with pytest.raises(ValueError, match='foo currently unsupported'):
        await plc_driver.get('foo1')
    with pytest.raises(ValueError, match='Inter-category ranges are unsupported'):
        await plc_driver.get('c1-x3')

@pytest.mark.asyncio(loop_scope='session')
async def test_set_error_handling(plc_driver):
    """Confirm the driver gives an error on invalid set() calls."""
    with pytest.raises(ValueError, match='foo currently unsupported'):
        await plc_driver.set('foo1', 1)

@pytest.mark.asyncio(loop_scope='session')
@pytest.mark.parametrize('prefix', ['x', 'y'])
async def test_get_xy_error_handling(plc_driver, prefix):
    """Ensure errors are handled for invalid get requests of x and y registers."""
    with pytest.raises(ValueError, match=r'address must be \*01-\*16.'):
        await plc_driver.get(f'{prefix}17')
    with pytest.raises(ValueError, match=r'address must be in \[001, 816\].'):
        await plc_driver.get(f'{prefix}1001')
    with pytest.raises(ValueError, match=r'address must be \*01-\*16.'):
        await plc_driver.get(f'{prefix}1-{prefix}17')
    with pytest.raises(ValueError, match=r'address must be in \[001, 816\].'):
        await plc_driver.get(f'{prefix}1-{prefix}1001')

@pytest.mark.asyncio(loop_scope='session')
async def test_set_y_error_handling(plc_driver):
    """Ensure errors are handled for invalid set requests of y registers."""
    with pytest.raises(ValueError, match=r'address must be \*01-\*16.'):
        await plc_driver.set('y17', True)
    with pytest.raises(ValueError, match=r'address must be in \[001, 816\].'):
        await plc_driver.set('y1001', True)
    with pytest.raises(ValueError, match=r'Data list longer than available addresses.'):
        await plc_driver.set('y816', [True, True])

@pytest.mark.asyncio(loop_scope='session')
async def test_c_error_handling(plc_driver):
    """Ensure errors are handled for invalid requests of c registers."""
    with pytest.raises(ValueError, match=r'C start address must be 1-2000.'):
        await plc_driver.get('c2001')
    with pytest.raises(ValueError, match=r'C end address must be >start and <=2000.'):
        await plc_driver.get('c1-c2001')
    with pytest.raises(ValueError, match=r'C start address must be 1-2000.'):
        await plc_driver.set('c2001', True)
    with pytest.raises(ValueError, match=r'Data list longer than available addresses.'):
        await plc_driver.set('c2000', [True, True])

@pytest.mark.asyncio(loop_scope='session')
async def test_sc_error_handling(plc_driver):
    """Ensure errors are handled for invalid requests of SC registers."""
    # Test invalid boundary (below range)
    with pytest.raises(ValueError, match=r'SC start address must be in \[1, 1000\]'):
        await plc_driver.set('sc0', True)  # Below valid range

    # Test invalid boundary (above range)
    with pytest.raises(ValueError, match=r'SC start address must be in \[1, 1000\]'):
        await plc_driver.set('sc1001', True)  # Above valid range

    # Test valid read-only SC
    with pytest.raises(ValueError, match=r"SC62 is not writable."):
        await plc_driver.set('sc62', True)  # Read-only SC

    # Test end address below start address
    with pytest.raises(ValueError, match=r'End address must be greater than start address.'):
        await plc_driver.get('sc100-sc50')  # End address less than start address

    # Test invalid range crossing writable boundaries
    with pytest.raises(ValueError, match=r'SC52 is not writable.'):
        # Range includes non-writable SC
        await plc_driver.set('sc52', [True, True])

    # Test data type mismatch
    with pytest.raises(ValueError, match=r"Expected sc50 as a bool."):
        await plc_driver.set('sc50', 123)  # SC expects a bool value

@pytest.mark.asyncio(loop_scope='session')
async def test_t_error_handling(plc_driver):
    """Ensure errors are handled for invalid requests of t registers."""
    with pytest.raises(ValueError, match=r'T start address must be 1-500.'):
        await plc_driver.get('t501')
    with pytest.raises(ValueError, match=r'T end address must be >start and <=500.'):
        await plc_driver.get('t1-t501')

@pytest.mark.asyncio(loop_scope='session')
async def test_ct_error_handling(plc_driver):
    """Ensure errors are handled for invalid requests of ct registers."""
    with pytest.raises(ValueError, match=r'CT start address must be 1-250.'):
        await plc_driver.get('ct251')
    with pytest.raises(ValueError, match=r'CT end address must be >start and <=250.'):
        await plc_driver.get('ct1-ct251')

@pytest.mark.asyncio(loop_scope='session')
async def test_dh_error_handling(plc_driver):
    """Ensure errors are handled for invalid requests of df registers."""
    with pytest.raises(ValueError, match=r'DH must be in \[1, 500\]'):
        await plc_driver.get('dh501')
    with pytest.raises(ValueError, match=r'DH end must be in \[1, 500\]'):
        await plc_driver.get('dh1-dh501')
    with pytest.raises(ValueError, match=r'DH must be in \[1, 500\]'):
        await plc_driver.set('dh501', 1)
    with pytest.raises(ValueError, match=r'Data list longer than available addresses.'):
        await plc_driver.set('dh500', [1, 2])

@pytest.mark.asyncio(loop_scope='session')
async def test_df_error_handling(plc_driver):
    """Ensure errors are handled for invalid requests of df registers."""
    with pytest.raises(ValueError, match=r'DF must be in \[1, 500\]'):
        await plc_driver.get('df501')
    with pytest.raises(ValueError, match=r'DF end must be in \[1, 500\]'):
        await plc_driver.get('df1-df501')
    with pytest.raises(ValueError, match=r'DF must be in \[1, 500\]'):
        await plc_driver.set('df501', 1.0)
    with pytest.raises(ValueError, match=r'Data list longer than available addresses.'):
        await plc_driver.set('df500', [1.0, 2.0])

@pytest.mark.asyncio(loop_scope='session')
async def test_ds_error_handling(plc_driver):
    """Ensure errors are handled for invalid requests of ds registers."""
    with pytest.raises(ValueError, match=r'DS must be in \[1, 4500\]'):
        await plc_driver.get('ds4501')
    with pytest.raises(ValueError, match=r'DS end must be in \[1, 4500\]'):
        await plc_driver.get('ds1-ds4501')
    with pytest.raises(ValueError, match=r'DS must be in \[1, 4500\]'):
        await plc_driver.set('ds4501', 1)
    with pytest.raises(ValueError, match=r'Data list longer than available addresses.'):
        await plc_driver.set('ds4500', [1, 2])

@pytest.mark.asyncio(loop_scope='session')
async def test_dd_error_handling(plc_driver):
    """Ensure errors are handled for invalid requests of dd registers."""
    with pytest.raises(ValueError, match=r'DD must be in \[1, 1000\]'):
        await plc_driver.get('dd1001')
    with pytest.raises(ValueError, match=r'DD end must be in \[1, 1000\]'):
        await plc_driver.get('dd1-dd1001')
    with pytest.raises(ValueError, match=r'DD must be in \[1, 1000\]'):
        await plc_driver.set('dd1001', 1)
    with pytest.raises(ValueError, match=r'Data list longer than available addresses.'):
        await plc_driver.set('dd1000', [1, 2])

@pytest.mark.asyncio(loop_scope='session')
async def test_td_error_handling(plc_driver):
    """Ensure errors are handled for invalid requests of td registers."""
    with pytest.raises(ValueError, match=r'TD must be in \[1, 500\]'):
        await plc_driver.get('td501')
    with pytest.raises(ValueError, match=r'TD end must be in \[1, 500\]'):
        await plc_driver.get('td1-td501')

@pytest.mark.asyncio(loop_scope='session')
async def test_ctd_error_handling(plc_driver):
    """Ensure errors are handled for invalid requests of ctd registers."""
    with pytest.raises(ValueError, match=r'CTD must be in \[1, 250\]'):
        await plc_driver.get('ctd251')
    with pytest.raises(ValueError, match=r'CTD end must be in \[1, 250\]'):
        await plc_driver.get('ctd1-ctd251')

@pytest.mark.asyncio(loop_scope='session')
async def test_sd_error_handling(plc_driver):
    """Ensure errors are handled for invalid requests of SD registers."""
    # Test out-of-range addresses
    with pytest.raises(ValueError, match=r'SD must be in \[1, 1000\]'):
        await plc_driver.get('sd1001')  # Above valid range
    with pytest.raises(ValueError, match=r'SD end must be in \[1, 1000\]'):
        await plc_driver.get('sd1-sd1001')  # Range includes invalid end address
    with pytest.raises(ValueError, match=r'SD1001 is not writable. Only specific SD registers are writable.'):
        await plc_driver.set('sd1001', 1)  # Above valid range

    # Test read-only boundaries
    with pytest.raises(ValueError, match=r'SD62 is not writable'):
        await plc_driver.set('sd62', 1)  # Read-only SD register
    with pytest.raises(ValueError, match=r'SD63 is not writable'):
        await plc_driver.set('sd63', 1)  # Read-only SD register

    # Test type mismatch
    with pytest.raises(ValueError, match=r'Expected sd29 as a int'):
        await plc_driver.set('sd29', 'string')  # SD expects an integer value
    with pytest.raises(ValueError, match=r'Expected sd29 as a int'):
        await plc_driver.set('sd29', [1, 'string'])  # SD expects all integers

    # Test valid writable SD
    await plc_driver.set('sd29', 2024)  # Valid writable address
    assert await plc_driver.get('sd29') == 2024

@pytest.mark.asyncio(loop_scope='session')
@pytest.mark.parametrize('prefix', ['y', 'c'])
async def test_bool_typechecking(plc_driver, prefix):
    """Ensure errors are handled for set() requests that should be bools."""
    with pytest.raises(ValueError, match=r'Expected .+ as a bool'):
        await plc_driver.set(f'{prefix}1', 1)
    with pytest.raises(ValueError, match=r'Expected .+ as a bool'):
        await plc_driver.set(f'{prefix}1', [1.0, 1])

@pytest.mark.asyncio(loop_scope='session')
async def test_df_typechecking(plc_driver):
    """Ensure errors are handled for set() requests that should be floats."""
    await plc_driver.set('df1', 1)
    with pytest.raises(ValueError, match=r'Expected .+ as a float'):
        await plc_driver.set('df1', True)
    with pytest.raises(ValueError, match=r'Expected .+ as a float'):
        await plc_driver.set('df1', [True, True])

@pytest.mark.asyncio(loop_scope='session')
@pytest.mark.parametrize('prefix', ['ds', 'dd'])
async def test_ds_dd_typechecking(plc_driver, prefix):
    """Ensure errors are handled for set() requests that should be ints."""
    with pytest.raises(ValueError, match=r'Expected .+ as a int'):
        await plc_driver.set(f'{prefix}1', 1.0)
    with pytest.raises(ValueError, match=r'Expected .+ as a int'):
        await plc_driver.set(f'{prefix}1', True)
    with pytest.raises(ValueError, match=r'Expected .+ as a int'):
        await plc_driver.set(f'{prefix}1', [True, True])
