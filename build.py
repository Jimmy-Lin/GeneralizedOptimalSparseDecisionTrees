import pathlib
import platform
import subprocess
import sys
import os
import shutil

def setup(args):
    subprocess.run([sys.executable, "setup.py"] + args).check_returncode()


def delvewheel(args):
    subprocess.run([sys.executable, "-m", "delvewheel"] + args).check_returncode()


def remove_dir_if_exists(str):
    if os.path.exists(str):
        shutil.rmtree(str)


if __name__ == '__main__':
    try:
        print("Rebuilding the project from scratch...")
        remove_dir_if_exists("dist")
        remove_dir_if_exists("gosdt.egg-info")
        setup(["clean"])
        setup(["bdist_wheel", "--build-type=Release", "-G", "Ninja", "--", "--", "-j{}".format(os.cpu_count())])
        if platform.system() == "Windows":
            print("Adding required dynamic libraries to the wheel file...")
            vcpkg = pathlib.Path(os.getenv("VCPKG"))
            dlls = [str(vcpkg / "installed\\x64-windows\\bin\\tbb.dll"),
                    str(vcpkg / "installed\\x64-windows\\bin\\tbbmalloc.dll"),
                    str(vcpkg / "installed\\x64-windows\\bin\\gmp-10.dll")]
            wheels = os.listdir("dist")
            assert len(wheels) == 1, "The number of generated wheels is not 1. All wheels: {}.".format(wheels)
            delvewheel(["repair", "--add-dll", ";".join(dlls), "dist/{}".format(wheels[0]), "-w", "dist"])
        print("All done.")
        exit(0)
    except subprocess.CalledProcessError:
        exit(1)
