-- ============================================================
-- SINGLE RECORD LUA SCRIPT FOR GUI APPLICATION
-- ============================================================
-- This script is designed to work with the Python GUI application.
-- It monitors a signal file and takes a single radar recording
-- when commanded by the GUI.
-- 
-- Configuration is identical to radar-script.lua
-- ============================================================

-- ============================================================
-- RADAR CONFIGURATION (same as radar-script.lua)
-- ============================================================

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

-- Frame configuration
frame_num = 110
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

-- DATA CAPTURE CARD API
if (ar1.CaptureCardConfig_EthInit("192.168.33.30", "192.168.33.180", "12:34:56:78:90:12", 4096, 4098) == 0) then
    WriteToLog("CaptureCardConfig_EthInit Success\n", "green")
else
    WriteToLog("CaptureCardConfig_EthInit failure\n", "red")
end

-- AWR12xx or xWR14xx-1, xWR16xx or xWR18xx or xWR68xx- 2 (second parameter indicates the device type)
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

-- ============================================================
-- GUI COMMUNICATION CONFIGURATION
-- ============================================================

-- Signal file location (Python GUI writes commands here)
signal_file = "C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\PostProc\\gui_signal.txt"

-- Default output directory
default_output_dir = "C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\PostProc\\gui-output\\"

-- Recording duration in ms
ms_per_record = frame_num * period

-- ============================================================
-- HELPER FUNCTIONS
-- ============================================================

function readSignalFile()
    local file = io.open(signal_file, "r")
    if file then
        local command = file:read("*line")
        local output_path = file:read("*line")
        file:close()
        return command, output_path
    end
    return nil, nil
end

function clearSignalFile()
    local file = io.open(signal_file, "w")
    if file then
        file:write("")
        file:close()
    end
end

function writeStatus(status)
    local status_file = signal_file:gsub("gui_signal.txt", "gui_status.txt")
    local file = io.open(status_file, "w")
    if file then
        file:write(status)
        file:close()
    end
end

function normalizeWindowsPath(path)
    -- Convert forward slashes to backslashes for Windows
    return path:gsub("/", "\\")
end

function takeRecording(output_path)
    WriteToLog("Starting recording: " .. output_path .. "\n", "blue")
    writeStatus("RECORDING")
    
    -- Normalize path for Windows (convert / to \)
    local normalized_path = normalizeWindowsPath(output_path)
    WriteToLog("Normalized path: " .. normalized_path .. "\n", "blue")
    
    -- Pass full path with .bin extension to CaptureCardConfig_StartRecord
    -- This matches the working radar-script.lua format
    ar1.CaptureCardConfig_StartRecord(normalized_path, 1)
    RSTD.Sleep(500)
    
    -- Trigger frame
    ar1.StartFrame()
    RSTD.Sleep(ms_per_record + 500)
    
    -- Stop recording
    ar1.CaptureCardConfig_StopRecord()
    RSTD.Sleep(500)
    
    WriteToLog("Recording complete: " .. normalized_path .. "\n", "green")
    writeStatus("COMPLETE")
end

-- ============================================================
-- MAIN LOOP - MONITOR FOR GUI COMMANDS
-- ============================================================

print("============================================================")
print("GUI RADAR CONTROL SCRIPT")
print("============================================================")
print("Monitoring signal file: " .. signal_file)
print("Frame duration: " .. ms_per_record .. " ms")
print("Waiting for commands from GUI...")
print("============================================================")

-- Clear any existing signal
clearSignalFile()
writeStatus("READY")

-- Main monitoring loop
recording_count = 0
running = true

while running do
    -- Check for command from GUI
    local command, output_path = readSignalFile()
    
    if command == "RECORD" and output_path then
        recording_count = recording_count + 1
        print("=== Recording #" .. recording_count .. " ===")
        
        -- Clear signal file to acknowledge command
        clearSignalFile()
        
        -- Take the recording
        takeRecording(output_path)
        
        print("Recording saved to: " .. output_path)
        
    elseif command == "EXIT" then
        print("Exit command received. Shutting down...")
        running = false
        
    end
    
    -- Small delay before checking again
    RSTD.Sleep(500)
end

writeStatus("STOPPED")
print("============================================================")
print("GUI Control Script ended. Total recordings: " .. recording_count)
print("============================================================")
