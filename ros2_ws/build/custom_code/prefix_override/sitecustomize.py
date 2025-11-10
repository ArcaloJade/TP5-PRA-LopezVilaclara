import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/alumno1/Documents/TP4-PRA-LopezVilaclara/ros2_ws/install/custom_code'
