from setuptools import find_packages, setup


package_name = "woki_web_controller"


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
    install_requires=["setuptools", "Flask"],
    zip_safe=True,
    maintainer="woki",
    maintainer_email="woki@example.com",
    description="WoKi ROS2 Web Controller intent interface",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "web_controller = woki_web_controller.web_controller:main",
        ],
    },
)
