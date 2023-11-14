#! /bin/bash -xe
export PYTHONHASHSEED=42
platform="$(uname -s)"
case "${platform}" in
    Linux*)     machine=linux;;
    Darwin*)    machine=macos;;
    *)          machine="UNKNOWN:${platform}"
esac
release_dir="neon_player_$(git describe --tags --long)_${machine}_x64"
echo "+ Creating bundle at $release_dir"
pyinstaller neon_player.spec --noconfirm --log-level DEBUG --distpath $release_dir
