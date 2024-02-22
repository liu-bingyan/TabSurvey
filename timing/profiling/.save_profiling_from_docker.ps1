$containerId = docker ps -q
$localPath = Resolve-Path -Path "TabSurvey/timing/profiling"

# Get the list of files in the container directory
$files = docker exec ${containerId} ls /opt/notebooks | Where-Object { $_ -like "*.lprof" }

# Loop through each file and copy it to the local path
foreach ($file in $files) {
    docker cp ${containerId}:/opt/notebooks/$file $localPath
    python -m line_profiler $localPath/$file > $localPath/$file.txt
    Remove-Item -Path $localPath/$file -Force
}


# Get the list of files in the container directory
$files = docker exec ${containerId} ls /opt/notebooks | Where-Object { $_ -like "*.txt" }

# Loop through each file and copy it to the local path
foreach ($file in $files) {
    docker cp ${containerId}:/opt/notebooks/$file $localPath
}

