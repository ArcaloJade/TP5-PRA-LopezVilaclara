// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from custom_msgs:msg/Belief.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__BELIEF__STRUCT_H_
#define CUSTOM_MSGS__MSG__DETAIL__BELIEF__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'mu'
#include "geometry_msgs/msg/detail/pose2_d__struct.h"

/// Struct defined in msg/Belief in the package custom_msgs.
typedef struct custom_msgs__msg__Belief
{
  geometry_msgs__msg__Pose2D mu;
  /// row-major 3x3
  double covariance[9];
} custom_msgs__msg__Belief;

// Struct for a sequence of custom_msgs__msg__Belief.
typedef struct custom_msgs__msg__Belief__Sequence
{
  custom_msgs__msg__Belief * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} custom_msgs__msg__Belief__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CUSTOM_MSGS__MSG__DETAIL__BELIEF__STRUCT_H_
