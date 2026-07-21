-- Open a Gaussian cube and create its signed isosurface in GaussView 6.
-- GaussView's Qt surface controls do not expose named macOS accessibility
-- elements, so the three isovalue fields are addressed relative to the dialog.

on gaussViewPidForPath(targetPath)
    tell application "System Events"
        repeat with candidateProcess in (every process whose name is "gview")
            set candidatePid to unix id of candidateProcess
            set windowTitles to name of every window of candidateProcess
            repeat with windowTitle in windowTitles
                if (windowTitle as text) contains targetPath then return candidatePid
            end repeat
        end repeat
    end tell
    return 0
end gaussViewPidForPath

on run argv
    if (count of argv) is less than 2 then error "Cube path and isovalue are required."
    set cubePath to (item 1 of argv) as text
    set isoValue to (item 2 of argv) as text

    set gaussViewPid to 0
    repeat 120 times
        set gaussViewPid to my gaussViewPidForPath(cubePath)
        if gaussViewPid is not 0 then exit repeat
        delay 0.25
    end repeat
    if gaussViewPid is 0 then error "GaussView did not open the requested cube."

    tell application "System Events"
        tell (first process whose unix id is gaussViewPid)
            set frontmost to true
            set cubeWindowTitle to ""
            set windowTitles to name of every window
            repeat with windowTitle in windowTitles
                if (windowTitle as text) contains (my cubePath) then
                    set cubeWindowTitle to windowTitle as text
                    exit repeat
                end if
            end repeat
            if cubeWindowTitle is "" then error "The requested GaussView cube window disappeared."

            -- GaussView keeps its own active-view state. Selecting the cube
            -- from Windows makes Results act on this cube even when several
            -- molecule groups are open.
            click menu item cubeWindowTitle of menu 1 of menu bar item "Windows" of menu bar 1
            delay 0.5
            click menu item "Surfaces/Contours..." of menu 1 of menu bar item "Results" of menu bar 1

            set dialogFound to false
            repeat 80 times
                try
                    if (name of window 1) ends with "Surfaces and Contours" then
                        set dialogFound to true
                        exit repeat
                    end if
                end try
                delay 0.25
            end repeat
            if not dialogFound then error "GaussView did not open Surfaces and Contours."

            set {windowX, windowY} to position of window 1
            set {windowWidth, windowHeight} to size of window 1
            if windowWidth < 590 or windowWidth > 660 or windowHeight < 590 or windowHeight > 660 then
                error "Unexpected GaussView 6.0.16 surface-dialog size; no coordinates were clicked."
            end if

            -- Set MO, density, and Laplacian defaults alike. This makes the
            -- launcher independent of how GaussView classifies a custom cube.
            set fieldY to (windowY + (windowHeight * 0.58)) as integer
            repeat with fieldFraction in {0.42, 0.67, 0.91}
                set fieldX to (windowX + (windowWidth * fieldFraction)) as integer
                click at {fieldX, fieldY}
                keystroke "a" using command down
                keystroke (my isoValue)
            end repeat

            -- Open Surface Actions and accept its first item, New Surface.
            set actionsX to (windowX + (windowWidth * 0.89)) as integer
            set actionsY to (windowY + (windowHeight * 0.35)) as integer
            click at {actionsX, actionsY}
            delay 0.2
            key code 115
            key code 36
        end tell
    end tell
end run
