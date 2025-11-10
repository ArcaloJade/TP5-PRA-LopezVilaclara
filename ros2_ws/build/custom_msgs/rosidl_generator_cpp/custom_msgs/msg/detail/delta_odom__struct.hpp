// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from custom_msgs:msg/DeltaOdom.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__DELTA_ODOM__STRUCT_HPP_
#define CUSTOM_MSGS__MSG__DETAIL__DELTA_ODOM__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__custom_msgs__msg__DeltaOdom __attribute__((deprecated))
#else
# define DEPRECATED__custom_msgs__msg__DeltaOdom __declspec(deprecated)
#endif

namespace custom_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct DeltaOdom_
{
  using Type = DeltaOdom_<ContainerAllocator>;

  explicit DeltaOdom_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->dr1 = 0.0;
      this->dr2 = 0.0;
      this->dt = 0.0;
    }
  }

  explicit DeltaOdom_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->dr1 = 0.0;
      this->dr2 = 0.0;
      this->dt = 0.0;
    }
  }

  // field types and members
  using _dr1_type =
    double;
  _dr1_type dr1;
  using _dr2_type =
    double;
  _dr2_type dr2;
  using _dt_type =
    double;
  _dt_type dt;

  // setters for named parameter idiom
  Type & set__dr1(
    const double & _arg)
  {
    this->dr1 = _arg;
    return *this;
  }
  Type & set__dr2(
    const double & _arg)
  {
    this->dr2 = _arg;
    return *this;
  }
  Type & set__dt(
    const double & _arg)
  {
    this->dt = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    custom_msgs::msg::DeltaOdom_<ContainerAllocator> *;
  using ConstRawPtr =
    const custom_msgs::msg::DeltaOdom_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<custom_msgs::msg::DeltaOdom_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<custom_msgs::msg::DeltaOdom_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      custom_msgs::msg::DeltaOdom_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<custom_msgs::msg::DeltaOdom_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      custom_msgs::msg::DeltaOdom_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<custom_msgs::msg::DeltaOdom_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<custom_msgs::msg::DeltaOdom_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<custom_msgs::msg::DeltaOdom_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__custom_msgs__msg__DeltaOdom
    std::shared_ptr<custom_msgs::msg::DeltaOdom_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__custom_msgs__msg__DeltaOdom
    std::shared_ptr<custom_msgs::msg::DeltaOdom_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const DeltaOdom_ & other) const
  {
    if (this->dr1 != other.dr1) {
      return false;
    }
    if (this->dr2 != other.dr2) {
      return false;
    }
    if (this->dt != other.dt) {
      return false;
    }
    return true;
  }
  bool operator!=(const DeltaOdom_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct DeltaOdom_

// alias to use template instance with default allocator
using DeltaOdom =
  custom_msgs::msg::DeltaOdom_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace custom_msgs

#endif  // CUSTOM_MSGS__MSG__DETAIL__DELTA_ODOM__STRUCT_HPP_
