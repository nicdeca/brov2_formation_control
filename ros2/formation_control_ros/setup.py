from glob import glob
from setuptools import find_packages, setup

package_name = "formation_control_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "numpy"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Nicola De Carli",
    maintainer_email="nico.decarli@live.it",
    description=(
        "ROS 2 adapter around the ROS-independent formation_control core."
    ),
    license="MIT",
    entry_points={
        "console_scripts": [
            (
                "follower_controller = "
                "formation_control_ros.follower_controller_node:main"
            ),
            (
                "leader_controller = "
                "formation_control_ros.leader_controller_node:main"
            ),
            (
                "offboard_heartbeat_wrench = "
                "formation_control_ros.offboard_heartbeat_wrench:main"
            ),
            (
                "experiment_phase_manager = "
                "formation_control_ros.experiment_phase_manager:main"
            ),
        ],
    },
)
