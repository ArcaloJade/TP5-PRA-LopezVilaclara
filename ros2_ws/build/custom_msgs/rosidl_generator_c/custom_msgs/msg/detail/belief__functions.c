// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from custom_msgs:msg/Belief.idl
// generated code does not contain a copyright notice
#include "custom_msgs/msg/detail/belief__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `mu`
#include "geometry_msgs/msg/detail/pose2_d__functions.h"

bool
custom_msgs__msg__Belief__init(custom_msgs__msg__Belief * msg)
{
  if (!msg) {
    return false;
  }
  // mu
  if (!geometry_msgs__msg__Pose2D__init(&msg->mu)) {
    custom_msgs__msg__Belief__fini(msg);
    return false;
  }
  // covariance
  return true;
}

void
custom_msgs__msg__Belief__fini(custom_msgs__msg__Belief * msg)
{
  if (!msg) {
    return;
  }
  // mu
  geometry_msgs__msg__Pose2D__fini(&msg->mu);
  // covariance
}

bool
custom_msgs__msg__Belief__are_equal(const custom_msgs__msg__Belief * lhs, const custom_msgs__msg__Belief * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // mu
  if (!geometry_msgs__msg__Pose2D__are_equal(
      &(lhs->mu), &(rhs->mu)))
  {
    return false;
  }
  // covariance
  for (size_t i = 0; i < 9; ++i) {
    if (lhs->covariance[i] != rhs->covariance[i]) {
      return false;
    }
  }
  return true;
}

bool
custom_msgs__msg__Belief__copy(
  const custom_msgs__msg__Belief * input,
  custom_msgs__msg__Belief * output)
{
  if (!input || !output) {
    return false;
  }
  // mu
  if (!geometry_msgs__msg__Pose2D__copy(
      &(input->mu), &(output->mu)))
  {
    return false;
  }
  // covariance
  for (size_t i = 0; i < 9; ++i) {
    output->covariance[i] = input->covariance[i];
  }
  return true;
}

custom_msgs__msg__Belief *
custom_msgs__msg__Belief__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  custom_msgs__msg__Belief * msg = (custom_msgs__msg__Belief *)allocator.allocate(sizeof(custom_msgs__msg__Belief), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(custom_msgs__msg__Belief));
  bool success = custom_msgs__msg__Belief__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
custom_msgs__msg__Belief__destroy(custom_msgs__msg__Belief * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    custom_msgs__msg__Belief__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
custom_msgs__msg__Belief__Sequence__init(custom_msgs__msg__Belief__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  custom_msgs__msg__Belief * data = NULL;

  if (size) {
    data = (custom_msgs__msg__Belief *)allocator.zero_allocate(size, sizeof(custom_msgs__msg__Belief), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = custom_msgs__msg__Belief__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        custom_msgs__msg__Belief__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
custom_msgs__msg__Belief__Sequence__fini(custom_msgs__msg__Belief__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      custom_msgs__msg__Belief__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

custom_msgs__msg__Belief__Sequence *
custom_msgs__msg__Belief__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  custom_msgs__msg__Belief__Sequence * array = (custom_msgs__msg__Belief__Sequence *)allocator.allocate(sizeof(custom_msgs__msg__Belief__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = custom_msgs__msg__Belief__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
custom_msgs__msg__Belief__Sequence__destroy(custom_msgs__msg__Belief__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    custom_msgs__msg__Belief__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
custom_msgs__msg__Belief__Sequence__are_equal(const custom_msgs__msg__Belief__Sequence * lhs, const custom_msgs__msg__Belief__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!custom_msgs__msg__Belief__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
custom_msgs__msg__Belief__Sequence__copy(
  const custom_msgs__msg__Belief__Sequence * input,
  custom_msgs__msg__Belief__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(custom_msgs__msg__Belief);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    custom_msgs__msg__Belief * data =
      (custom_msgs__msg__Belief *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!custom_msgs__msg__Belief__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          custom_msgs__msg__Belief__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!custom_msgs__msg__Belief__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
