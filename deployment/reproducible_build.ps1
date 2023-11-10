Set-PSDebug -Trace 1
$Env:PYTHONHASHSEED = 42

$release_dir = "neon_$(git describe --tags --long)_windows_x64"

Write-Output "Creating bundle at $release_dir"
pyinstaller pupil_core.spec --noconfirm --log-level INFO --distpath $release_dir
