import pathlib
import platform
import subprocess
import sys
import os
import shutil

def setup(args) -> None:
    """
    Interface of the cross-platform scikit-build script
    :param args: A list of arguments passed to `setup.py`
    """
    subprocess.run([sys.executable, "setup.py"] + args).check_returncode()


def delocate_wheel(args) -> None:
    """
    Interface of `delocate-wheel` that repairs wheel files on macOS
    :param args: A list of arguments passed to `delocate-wheel`
    """
    subprocess.run(["delocate-wheel"] + args).check_returncode()


def auditwheel(args) -> None:
    """
    Interface of `auditwheel` that repairs wheel files on Ubuntu
    :param args: A list of arguments passed to "auditwheel"
    """
    subprocess.run(["auditwheel"] + args).check_returncode()


def delvewheel(args) -> None:
    """
    Interface of `delvewheel` that repairs wheel files on Windows
    :param args: A list of arguments passed to `delvewheel`
    """
    subprocess.run([sys.executable, "-m", "delvewheel"] + args).check_returncode()


def repair_wheel(wheel) -> None:
    """
    Repair the generated wheel file by copying all required dynamic libraries to it
    :param wheel: Path to the generated wheel file
    """
    # Fetch the current operating system
    system = platform.system()
    # Fetch the name of the generated wheel file
    wheels = os.listdir("dist")
    assert len(wheels) == 1, "The number of generated wheels is not 1. All wheels: {}.".format(wheels)
    wheel_path = "dist/{}".format(wheels[0])
    # Repair the wheel
    if system == "Darwin":
        delocate_wheel(["-w", "dist", "-v", wheel_path])
    elif system == "Linux":
        auditwheel(["repair", "-w", "dist", "--plat", "linux_x86_64", wheel_path])
    elif system == "Windows":
        search_path = pathlib.Path(os.getenv("VCPKG_INSTALLATION_ROOT")) / "installed\\x64-windows\\bin"
        delvewheel(["repair", "--no-mangle-all", "--add-path", search_path, wheel_path, "-w", "dist"])
    else:
        print("{} is not supported.".format(system))
        raise EnvironmentError


def remove_dir_if_exists(str) -> None:
    if os.path.exists(str):
        shutil.rmtree(str)


if __name__ == '__main__':
    try:
        print(">> Cleaning the garbage...")
        remove_dir_if_exists("dist")
        remove_dir_if_exists("gosdt.egg-info")
        setup(["clean"])

        print(">> Rebuilding the project from scratch...")
        setup(["bdist_wheel", "--build-type=Release", "-G", "Ninja", "--", "--", "-j{}".format(os.cpu_count())])

        print(">> Adding required dynamic libraries to the wheel file...")
        wheels = os.listdir("dist")
        assert len(wheels) == 1, "The number of generated wheels is not 1. All wheels: {}.".format(wheels)
        wheel = "dist/{}".format(wheels[0])
        print("Wheel file to be fixed: {}.".format(wheel))
        repair_wheel(wheel)

        print("All done.")
        exit(0)
    except (EnvironmentError, subprocess.CalledProcessError):
        exit(1)
