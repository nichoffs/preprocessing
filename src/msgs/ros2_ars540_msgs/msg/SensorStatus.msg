# sync status
uint8 TIME_SYNC = 1
uint8 TIME_NEVER_SYNC = 2
uint8 TIME_SYNC_LOST = 3

# plug orientation
uint8 PLUG_LEFT = 0
uint8 PLUG_RIGHT = 1

# frequency slot
uint8 LOW = 0 # 76.23 GHz
uint8 MID = 1 # 76.48 GHz
uint8 HIGH = 2 # 76.73 GHz

# hcc
uint8 WORLDWIDE = 1
uint8 JAPAN = 2

# powersave standstill
uint8 OFF = 0
uint8 ON = 1

# status enum for:
#     status_longitudinal_velocity
#     status_longitudinal_acceleration
#     status_lateral_acceleration
#     status_yaw_rate
#     status_steering_rate
#     status_driving_direction
#     status_characteristic_speed
uint8 VDY_OK = 0
uint8 VDY_NOTOK = 1

# radar status
uint8 STATE_INIT = 0
uint8 STATE_OK = 1
uint8 STATE_INVALID = 2

# status voltage
uint8 CURRENT_UNDERVOLTAGE = 1
uint8 PAST_UNDERVOLTAGE  = 2
uint8 CURRENT_OVERVOLTAGE = 4
uint8 PAST_OVERVOLTAGE = 8

# status temperature
uint8 CURRENT_UNDERTEMPERATURE = 1
uint8 PAST_UNDERTEMPERATURE = 2
uint8 CURRENT_OVERTEMPERATURE = 4
uint8 PAST_OVERTEMPERATURE = 8

# status blockage
uint8 VALUE_BLIND = 0
uint8 VALUE_HIGH = 1
uint8 VALUE_MID = 2
uint8 VALUE_LOW = 3
uint8 VALUE_NONE = 4
uint8 VALUE_INVALID = 255

std_msgs/Header header

# timestamp
uint32 fractional_seconds
uint32 seconds

# software version
uint8 sw_version_major
uint8 sw_version_minor
uint8 sw_version_patch

# time syncronization status
uint8 sync_status

# Longitudinal sensor position (AUTOSAR)
float32 longitudinal
# Lateral sensor position (AUTOSAR)
float32 lateral
# Vertical sensor position (AUTOSAR)
float32 vertical
# Sensor yaw angle (AUTOSAR)
float32 yaw
# Sensor pitch angle (AUTOSAR)
float32 pitch
# Orientation of plug
uint8 plug_orientation

# Vehicle length
float32 length
# Vehicle width
float32 width
# Vehicle height
float32 height
# Vehicle wheelbase
float32 wheelbase

# Maximum detection distance
uint16 maximum_distance
# Center frequency
uint8 frequency_slot
# Cycle time
uint8 cycle_time
# Cycle offset
uint8 time_slot
# Country code
uint8 hcc
# Power saving in standstill
uint8 powersave_standstill

# Sensor IP address
uint32 sensor_ip_address_0
# Reserved
uint32 sensor_ip_address_1

# Counter that counts up if new configuration has been received and accepted
uint8 configuration_counter

# Signals if current VDY is OK or timed out
uint8 status_longitudinal_velocity
# Signals if current VDY is OK or timed out
uint8 status_longitudinal_acceleration
# Signals if current VDY is OK or timed out
uint8 status_lateral_acceleration
# Signals if current VDY is OK or timed out
uint8 status_yaw_rate
# Signals if current VDY is OK or timed out
uint8 status_steering_rate
# Signals if current VDY is OK or timed out
uint8 status_driving_direction
# Signals if current VDY is OK or timed out
uint8 status_characteristic_speed

# Signals if Radar Status is OK
uint8 status_radar
# Signals if Voltage is OK
uint8 status_voltage
# Signals if Temperature  is OK
uint8 status_temperature
# Signals if Blockage is OK
uint8 status_blockage
