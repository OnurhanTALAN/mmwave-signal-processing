if (ar1.ChanNAdcConfig(1, 0, 0, 1, 1, 1, 1, 2, 1, 0) == 0) then
    WriteToLog("ChanNAdcConfig Success\n", "green")
else
    WriteToLog("ChanNAdcConfig failure\n", "red")
end

if (ar1.LPModConfig(0, 0) == 0) then
    WriteToLog("Regualar mode Cfg Success\n", "green")
else
    WriteToLog("Regualar mode Cfg failure\n", "red")
end

if (ar1.RfInit() == 0) then
    WriteToLog("RfInit Success\n", "green")
else
    WriteToLog("RfInit failure\n", "red")
end

RSTD.Sleep(1000)

if (ar1.DataPathConfig(513, 1216644097, 0) == 0) then
    WriteToLog("DataPathConfig Success\n", "green")
else
    WriteToLog("DataPathConfig failure\n", "red")
end

if (ar1.LvdsClkConfig(1, 1) == 0) then
    WriteToLog("LvdsClkConfig Success\n", "green")
else
    WriteToLog("LvdsClkConfig failure\n", "red")
end

if (ar1.LVDSLaneConfig(0, 1, 1, 0, 0, 1, 0, 0) == 0) then
    WriteToLog("LVDSLaneConfig Success\n", "green")
else
    WriteToLog("LVDSLaneConfig failure\n", "red")
end

if(ar1.ProfileConfig(0, 77, 100, 6, 130, 0, 0, 0, 0, 0, 0, 29.982, 0, 256, 10000, 0, 0, 30) == 0) then
    WriteToLog("ProfileConfig Success\n", "green")
else
    WriteToLog("ProfileConfig failure\n", "red")
end

if (ar1.ChirpConfig(0, 0, 0, 0, 0, 0, 0, 1, 0, 0) == 0) then
    WriteToLog("ChirpConfig Success\n", "green")
else
    WriteToLog("ChirpConfig failure\n", "red")
end

frame_num = 100
chirp_num = 64
period = 20

if (ar1.FrameConfig(0, 0, frame_num, chirp_num, period, 0, 0, 1) == 0) then
    WriteToLog("FrameConfig Success\n", "green")
else
    WriteToLog("FrameConfig failure\n", "red")
end

-- select Device type
if (ar1.SelectCaptureDevice("DCA1000") == 0) then
    WriteToLog("SelectCaptureDevice Success\n", "green")
else
    WriteToLog("SelectCaptureDevice failure\n", "red")
end

--DATA CAPTURE CARD API
if (ar1.CaptureCardConfig_EthInit("192.168.33.30", "192.168.33.180", "12:34:56:78:90:12", 4096, 4098) == 0) then
    WriteToLog("CaptureCardConfig_EthInit Success\n", "green")
else
    WriteToLog("CaptureCardConfig_EthInit failure\n", "red")
end

--AWR12xx or xWR14xx-1, xWR16xx or xWR18xx or xWR68xx- 2 (second parameter indicates the device type)
if (ar1.CaptureCardConfig_Mode(1, 2, 1, 2, 3, 30) == 0) then
    WriteToLog("CaptureCardConfig_Mode Success\n", "green")
else
    WriteToLog("CaptureCardConfig_Mode failure\n", "red")
end

if (ar1.CaptureCardConfig_PacketDelay(25) == 0) then
    WriteToLog("CaptureCardConfig_PacketDelay Success\n", "green")
else
    WriteToLog("CaptureCardConfig_PacketDelay failure\n", "red")
end

--Start Record ADC data
-- ar1.CaptureCardConfig_StartRecord(adc_data_path, 1)
-- RSTD.Sleep(1000)

-- --Trigger frame
-- ar1.StartFrame()
-- RSTD.Sleep(5000)

--Post process the Capture RAW ADC data
-- ar1.StartMatlabPostProc(adc_data_path)
-- WriteToLog("Please wait for a few seconds for matlab post processing .....!!!! \n", "green")
-- RSTD.Sleep(10000)


-- ============================================================
-- RECORDING CONFIGURATION
-- ============================================================
-- Total recording time: 5 minutes = 300,000 ms
-- Each file duration: 0.5 seconds = 500 ms
-- Frames per file: 500ms / 20ms = 25 frames
-- Total files: 300,000ms / 500ms = 600 files
-- ============================================================

-- ============================================================
-- RECORDING CONFIGURATION
-- ============================================================
-- Total recording time: 20 minutes = 1200000 ms
-- Each file duration: 0.5 seconds = 500 ms
-- Frames per file: 500ms / 20ms = 25 frames
-- Total files: 1200000ms / 500ms = 2400 files
-- ============================================================

-- Recording parameters (must match FrameConfig above)
ms_per_record = frame_num * period
number_of_records = 2400

-- Output directories
output_dir_inc = "C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\PostProc\\pipe-vibration\\increasing-vibration\\"
output_dir_dec = "C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\PostProc\\pipe-vibration\\decreasing-vibration\\"

motor_signal_file = "C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\PostProc\\motor_signal.txt"

function setMotor(state)
    local file = io.open(motor_signal_file, "w")
    if file then
        file:write(state)
        file:close()
        WriteToLog("Motor signal: " .. state .. "\n", "blue")
    else
        WriteToLog("WARNING: Could not write motor signal file!\n", "red")
    end
end

function motorInc()
    setMotor("INC")
end

function motorDec()
    setMotor("DEC")
end

function motorOff()
    setMotor("OFF")
end

function motorExit()
    setMotor("EXIT")
end

print("============================================================")
print("Starting " .. number_of_records .. " paired recordings")
print("Each recording: " .. ms_per_record .. " ms")
print("Total files: " .. (number_of_records * 2))
print("Ensure Python helper is running: python motor_serial.py COM3")
print("============================================================")

motorOff()
RSTD.Sleep(500)

for i = 1, number_of_records do
    print("=== Pair Recording " .. i .. "/" .. number_of_records .. " ===")

    ------------------------------------------------------------------
    -- INCREASING VIBRATION RECORD
    ------------------------------------------------------------------
    adc_data_path = output_dir_inc .. "adc_data_" .. i .. ".bin"

    motorOff()
    ar1.CaptureCardConfig_StartRecord(adc_data_path, 1)
    RSTD.Sleep(500)

    motorInc()
    RSTD.Sleep(100)

    ar1.StartFrame()
    RSTD.Sleep(ms_per_record + 500)

    motorOff()
    ar1.CaptureCardConfig_StopRecord()
    RSTD.Sleep(1500)

    ------------------------------------------------------------------
    -- DECREASING VIBRATION RECORD
    ------------------------------------------------------------------
    adc_data_path = output_dir_dec .. "adc_data_" .. i .. ".bin"

    motorOff()
    ar1.CaptureCardConfig_StartRecord(adc_data_path, 1)
    RSTD.Sleep(500)

    motorDec()
    RSTD.Sleep(100)

    ar1.StartFrame()
    RSTD.Sleep(ms_per_record + 500)

    motorOff()
    ar1.CaptureCardConfig_StopRecord()
    RSTD.Sleep(1500)
end

motorOff()
RSTD.Sleep(500)
motorExit()

print("============================================================")
print("Recording complete! " .. (number_of_records * 2) .. " files saved.")
print("============================================================")