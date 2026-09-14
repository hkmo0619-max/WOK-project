from setuptools import setup

package_name = 'lidar_avoidance'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='geonmo',
    maintainer_email='geonmo@example.com',
    description='LiDAR obstacle avoidance package',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'avoid = lidar_avoidance.avoid:main',
    ],
},    
)
