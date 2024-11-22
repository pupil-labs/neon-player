# Pupil
<a
href="https://pupil-labs.com"
rel="noopener"
target="_blank">
	<p align="center">
		<img
		src="https://raw.githubusercontent.com/wiki/pupil-labs/neon-player/media/images/pupil_labs_neon_player_banner.jpg"
		alt="Pupil Labs - Neon Player"/>
	</p>
</a>

*Neon Player* is an offline desktop application for [Neon](https://pupil-labs.com/products/neon) users to examine, visualize, and export eyetracking data.

To connect with us and our community, chat with us on [Discord](https://pupil-labs.com/chat "Pupil Server on Discord").

## Users
To get started
* Download the [latest release](https://github.com/pupil-labs/neon-player/releases) and launch the Neon Player application
* [Transfer your recordings from your Neon Companion Device to your computer](https://docs.pupil-labs.com/neon/how-tos/data-collection/transfer-recordings-via-usb.html)
* Drag-and-drop a recording folder onto the Neon Player window

## Developers

### Installing Dependencies and Code

To run the source code, you will need Python 3.7 or newer! We target Python 3.11 in our newer bundles and we recommend you to do the same.

Note: It is recommended to install the requirements into a
[virtual environment](https://docs.python.org/3/tutorial/venv.html).

If you are on MacOS, then you will want to use Homebrew to install the following packages first (and then configure your environment appropriately):

```sh
brew install glew libusb-compat eigen opencv ffmpeg@5
export LDFLAGS="-L/opt/homebrew/opt/ffmpeg@5/lib:$LDFLAGS"
export CPPFLAGS="-I/opt/homebrew/opt/ffmpeg@5/include:$CPPFLAGS"
export PKG_CONFIG_PATH="/opt/homebrew/opt/ffmpeg@5/lib/pkgconfig:$PKG_CONFIG_PATH"
```

Then, you can proceed with the following commands on all systems:

```sh
git clone https://github.com/pupil-labs/neon-player.git
cd neon-player
python -m pip install -r requirements.txt
```

If you have trouble installing any of the dependencies, please see the corresponding
code repository for manual installation steps and troubleshooting.

#### Linux

##### Audio Playback

The [`sounddevice`](https://python-sounddevice.readthedocs.io/en/0.4.5/installation.html#installation) package depends on the `libportaudio2` library:

```sh
sudo apt install libportaudio2
```

### Run Neon Player

```sh
cd pupil_src
python main.py
```

#### Command Line Arguments

The following arguments are supported:

| Flag                   | Description                              |
| ---------------------- | ---------------------------------------- |
| `-h, --help`           | Show help message and exit.              |
| `--version`            | Show version and exit.                   |
| `--debug`              | Display debug log messages.              |
| `--profile`            | Profile the app's CPU time.              |
| `<recording>`          | (Player) Path to recording.              |


## License
All source code written by Pupil Labs is open for use in compliance with the [GNU Lesser General Public License (LGPL v3.0)](http://www.gnu.org/licenses/lgpl-3.0.en.html). We want you to change and improve the code -- make a fork! Make sure to share your work with the community!
