from setuptools import find_packages, setup

package_name = "woki_step_detector"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="woki",
    maintainer_email="woki@example.com",
    description="WoKi D435 two-boundary step height detector ROS2 node",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "step_height_node = woki_step_detector.step_height_node:main",
        ],
    },
)
