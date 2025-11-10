// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from custom_msgs:msg/Belief.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__BELIEF__STRUCT_HPP_
#define CUSTOM_MSGS__MSG__DETAIL__BELIEF__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'mu'
#include "geometry_msgs/msg/detail/pose2_d__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__custom_msgs__msg__Belief __attribute__((deprecated))
#else
# define DEPRECATED__custom_msgs__msg__Belief __declspec(deprecated)
#endif

namespace custom_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Belief_
{
  using Type = Belief_<ContainerAllocator>;

  explicit Belief_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : mu(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 9>::iterator, double>(this->covariance.begin(), this->covariance.end(), 0.0);
    }
  }

  explicit Belief_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : mu(_alloc, _init),
    covariance(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 9>::iterator, double>(this->covariance.begin(), this->covariance.end(), 0.0);
    }
  }

  // field types and members
  using _mu_type =
    geometry_msgs::msg::Pose2D_<ContainerAllocator>;
  _mu_type mu;
  using _covariance_type =
    std::array<double, 9>;
  _covariance_type covariance;

  // setters for named parameter idiom
  Type & set__mu(
    const geometry_msgs::msg::Pose2D_<ContainerAllocator> & _arg)
  {
    this->mu = _arg;
    return *this;
  }
  Type & set__covariance(
    const std::array<double, 9> & _arg)
  {
    this->covariance = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    custom_msgs::msg::Belief_<ContainerAllocator> *;
  using ConstRawPtr =
    const custom_msgs::msg::Belief_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<custom_msgs::msg::Belief_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<custom_msgs::msg::Belief_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      custom_msgs::msg::Belief_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<custom_msgs::msg::Belief_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      custom_msgs::msg::Belief_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<custom_msgs::msg::Belief_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<custom_msgs::msg::Belief_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<custom_msgs::msg::Belief_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__custom_msgs__msg__Belief
    std::shared_ptr<custom_msgs::msg::Belief_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__custom_msgs__msg__Belief
    std::shared_ptr<custom_msgs::msg::Belief_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Belief_ & other) const
  {
    if (this->mu != other.mu) {
      return false;
    }
    if (this->covariance != other.covariance) {
      return false;
    }
    return true;
  }
  bool operator!=(const Belief_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Belief_

// alias to use template instance with default allocator
using Belief =
  custom_msgs::msg::Belief_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace custom_msgs

#endif  // CUSTOM_MSGS__MSG__DETAIL__BELIEF__STRUCT_HPP_
