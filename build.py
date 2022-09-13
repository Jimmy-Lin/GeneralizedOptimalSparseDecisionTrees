import pathlib
import platform
import subprocess
import sys
import os


def setup(args):
    subprocess.run([sys.executable, "setup.py"] + args).check_returncode()


def delvewheel(args):
    subprocess.run([sys.executable, "-m", "delvewheel"] + args).check_returncode()


if __name__ == '__main__':
    try:
        print("Rebuilding the project from scratch...")
        setup(["clean"])
        setup(["bdist_wheel", "--build-type=Release", "-G Ninja", "--", "--", "-j{}".format(os.cpu_count())])
        if platform.system() == "Windows":
            print("Adding required dynamic libraries to the wheel file...")
            vcpkg = pathlib.Path(os.getenv("VCPKG"))
            dlls = [str(vcpkg / "installed\\x64-windows\\bin\\tbb.dll"),
                    str(vcpkg / "installed\\x64-windows\\bin\\tbbmalloc.dll"),
                    str(vcpkg / "installed\\x64-windows\\bin\\gmp-10.dll")]
            delvewheel(["repair", "--all-dll", ";".join(dlls), "dist/*.whl", "-w", "dist"])
        print("All done.")
        exit(0)
    except subprocess.CalledProcessError:
        exit(1)
    