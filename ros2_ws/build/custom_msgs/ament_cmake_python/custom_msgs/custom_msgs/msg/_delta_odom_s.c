// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from custom_msgs:msg/DeltaOdom.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "custom_msgs/msg/detail/delta_odom__struct.h"
#include "custom_msgs/msg/detail/delta_odom__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool custom_msgs__msg__delta_odom__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[38];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("custom_msgs.msg._delta_odom.DeltaOdom", full_classname_dest, 37) == 0);
  }
  custom_msgs__msg__DeltaOdom * ros_message = _ros_message;
  {  // dr1
    PyObject * field = PyObject_GetAttrString(_pymsg, "dr1");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->dr1 = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // dr2
    PyObject * field = PyObject_GetAttrString(_pymsg, "dr2");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->dr2 = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // dt
    PyObject * field = PyObject_GetAttrString(_pymsg, "dt");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->dt = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * custom_msgs__msg__delta_odom__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of DeltaOdom */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("custom_msgs.msg._delta_odom");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "DeltaOdom");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  custom_msgs__msg__DeltaOdom * ros_message = (custom_msgs__msg__DeltaOdom *)raw_ros_message;
  {  // dr1
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->dr1);
    {
      int rc = PyObject_SetAttrString(_pymessage, "dr1", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // dr2
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->dr2);
    {
      int rc = PyObject_SetAttrString(_pymessage, "dr2", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // dt
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->dt);
    {
      int rc = PyObject_SetAttrString(_pymessage, "dt", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
