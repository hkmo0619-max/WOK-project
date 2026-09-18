from setuptools import find_packages, setup


package_name = "woki_stm32_bridge"


setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "pyserial"],
    zip_safe=True,
    maintainer="woki",
    maintainer_email="woki@example.com",
    description="WoKi ROS2 STM32 serial transport bridge",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "stm32_bridge = woki_stm32_bridge.stm32_bridge:main",
        ],
    },
)
