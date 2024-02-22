$containerId = docker ps -q
$localPath = Resolve-Path -Path "TabSurvey/timing/models"

# Get the list of files in the container directory
$files = docker exec ${containerId} ls /opt/notebooks/logs

# Loop through each file and copy it to the local path
foreach ($file in $files) {
    docker cp ${containerId}:/opt/notebooks/logs/$file $localPath
}

