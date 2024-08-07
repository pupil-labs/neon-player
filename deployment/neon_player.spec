# -*- mode: python -*-

import logging
import os
import pathlib
import platform
import sys
from importlib.resources import files

import PyInstaller
from PyInstaller.utils.hooks import collect_all
from rich.logging import RichHandler
from rich.traceback import install

install(suppress=[PyInstaller])
logging.getLogger().handlers = [RichHandler()]

sys.path.append(os.path.join("..", "pupil_src", "shared_modules"))
sys.path.append(".")

from _packaging import (
    ICON_EXT,
    LIB_EXT,
    SupportedPlatform,
    linux,
    macos,
    pupil_version,
    windows,
    write_version_file,
)

SPECPATH: str
DISTPATH: str


def main():
    cwd = SPECPATH
    current_platform = SupportedPlatform(platform.system())
    deployment_root = pathlib.Path(cwd)

    logging.debug(f"Writing version file to {DISTPATH}")
    version_file_path: pathlib.Path = write_version_file(DISTPATH)

    all_datas = []
    all_binaries = []
    all_hidden_imports = []
    for name in (
        "cysignals",
        "pupil_labs.rec_export",
        "xgboost",
        "zmq",
        "pyre",
        "pyglui",
        "ndsi",
        "pupil_apriltags",
        "pye3d",
        "OpenGL.GL",
        "OpenGL.platform.egl",  # wayland support
        "OpenGL.platform.glx",  # x11 support
        "pylsl",
        "sklearn",
        "glfw",
        "scipy._lib.array_api_compat.numpy.fft",
        "scipy.special._special_ufuncs",
    ):
        datas, binaries, hiddenimports = collect_all(
            name, exclude_datas=["**/__pycache__"]
        )
        all_datas.extend(datas)
        all_binaries.extend(binaries)
        all_hidden_imports.extend(hiddenimports)

    if current_platform == SupportedPlatform.linux:
        runtime_hooks = ["runtime_hook_sounddevice.py"]
    else:
        runtime_hooks = []

    a = Analysis(
        ["../pupil_src/main.py"],
        pathex=["../pupil_src/shared_modules/"],
        datas=all_datas,
        binaries=all_binaries,
        hiddenimports=all_hidden_imports,
        runtime_hooks=runtime_hooks,
    )
    pyz = PYZ(a.pure)

    for name in ("player", ):
        icon_name = "neon-" + name + ICON_EXT[current_platform]
        icon_path = (deployment_root / "icons" / icon_name).resolve()

        exe_kwargs = {
            "exclude_binaries": True,
            "name": f"neon_{name}",
            "debug": False,
            "strip": False,
            "upx": False,
            "console": True,
            "icon": str(icon_path),
            "resources": [f"{icon_path},ICON,icon"],
            "target_arch": "x86_64",
            "entitlements_file": "entitlements.plist",
            "codesign_identity": os.environ.get("MACOS_CODESIGN_IDENTITY", None),
        }

        try:
            exe = EXE(pyz, a.scripts, **exe_kwargs)
        except SystemError as exc:
            if "codesign_identity" in exe_kwargs:
                print("Failed to build EXE. Codesign problem? Trying again without codesign...", file=sys.stderr)
                del exe_kwargs["codesign_identity"]
                exe = EXE(pyz, a.scripts, **exe_kwargs)

            else:
                raise exc

        extras: list[tuple[str, str, str]] = []
        extras.append((version_file_path.name, str(version_file_path), "DATA"))

        # Add binaries that are not being collected automatically

        apriltags: pathlib.Path = files("pupil_apriltags")
        apriltags = apriltags.with_name(apriltags.name + ".libs")
        extras.extend(
            (bin_path.name, str(bin_path), "BINARY")
            for bin_path in apriltags.rglob("*" + LIB_EXT[current_platform])
        )

        binaries = a.binaries
        if current_platform == SupportedPlatform.linux:
            # libc is also not meant to travel with the bundle. Otherwise pyre.helpers with segfault.
            binaries = (b for b in binaries if not "libc.so" in b[0])
            # libstdc++ is also not meant to travel with the bundle. Otherwise nvideo opengl drivers will fail to load.
            binaries = (b for b in binaries if not "libstdc++.so" in b[0])
            # required for 14.04 16.04 interoperability.
            binaries = (b for b in binaries if not "libgomp.so.1" in b[0])
            # required for 17.10 interoperability.
            binaries = (b for b in binaries if not "libdrm.so.2" in b[0])
            binaries = list(binaries)

        whitelist = {"cv2"}
        blacklist_ext = {
            ".c",
            ".py",
            ".txt",
            ".cpp",
            ".pxi",
            ".typed",
            ".csv",
            ".md",
            ".rst",
            ".pxd",
            ".h",
            ".pyi",
            ".pyx",
            ".pyc",
        }

        data = [
            d
            for d in a.datas
            if (
                any(pat in d[0] for pat in whitelist)
                or os.path.splitext(d[0])[1] not in blacklist_ext
            )
        ]

        app_name = f"Neon {name.capitalize()}"
        collection = COLLECT(
            exe,
            binaries,
            a.zipfiles,
            data,
            extras,
            strip=False,
            upx=False,
            name=app_name,
        )

        BUNDLE(
            collection,
            name=f"{app_name}.app",
            icon=icon_path,
            version=str(pupil_version()),
            bundle_identifier=(
                f"com.pupil-labs.{app_name.lower().replace(' ','_')}"
            ),
            info_plist={
                'NSHighResolutionCapable': 'True'
            },
        )

    bundle_postprocessing = {
        SupportedPlatform.windows: windows.create_compressed_msi,
        SupportedPlatform.macos: macos.package_bundles_as_dmg,
        SupportedPlatform.linux: linux.create_zipped_deb_packages,
    }
    bundle_postprocessing[current_platform](pathlib.Path(DISTPATH), pupil_version())


def apriltag_relative_path(absolute_path: pathlib.Path):
    """Returns pupil_apriltags/lib/*"""
    return os.path.join(*absolute_path.parts[-3:])


main()
