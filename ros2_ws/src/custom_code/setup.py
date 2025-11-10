from setuptools import find_packages, setup
import os
from glob import glob


package_name = 'custom_code'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='',
    maintainer_email='',
    description='',
    license='',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "my_tf = custom_code.my_tf:main",
            "likelihood = custom_code.likelihood:main",
            "localization = custom_code.my_localization:main",
            "ekf = custom_code.my_ekf:main",
            "features = custom_code.features:main",
            "feature_finder = custom_code.feature_finder:main",
            'ekf_prediction = custom_code.ekf_prediction:main',
            'ekf_correction = custom_code.ekf_correction:main',
            'fastslam = custom_code.my_fastslam:main',
            "fastslam_node = custom_code.fastslam_correct:main"
        ],
    },
)
