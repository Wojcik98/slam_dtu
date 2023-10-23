from setuptools import find_packages, setup

package_name = "slam_dtu"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Michal Wojcik",
    maintainer_email="wojcikmichal98@gmail.com",
    description="Package for the SLAM project at the DTU",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["localization = slam_dtu.localization:main"],
    },
)
