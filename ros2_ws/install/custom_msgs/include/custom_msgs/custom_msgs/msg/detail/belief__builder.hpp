// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from custom_msgs:msg/Belief.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__BELIEF__BUILDER_HPP_
#define CUSTOM_MSGS__MSG__DETAIL__BELIEF__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "custom_msgs/msg/detail/belief__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace custom_msgs
{

namespace msg
{

namespace builder
{

class Init_Belief_covariance
{
public:
  explicit Init_Belief_covariance(::custom_msgs::msg::Belief & msg)
  : msg_(msg)
  {}
  ::custom_msgs::msg::Belief covariance(::custom_msgs::msg::Belief::_covariance_type arg)
  {
    msg_.covariance = std::move(arg);
    return std::move(msg_);
  }

private:
  ::custom_msgs::msg::Belief msg_;
};

class Init_Belief_mu
{
public:
  Init_Belief_mu()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Belief_covariance mu(::custom_msgs::msg::Belief::_mu_type arg)
  {
    msg_.mu = std::move(arg);
    return Init_Belief_covariance(msg_);
  }

private:
  ::custom_msgs::msg::Belief msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::custom_msgs::msg::Belief>()
{
  return custom_msgs::msg::builder::Init_Belief_mu();
}

}  // namespace custom_msgs

#endif  // CUSTOM_MSGS__MSG__DETAIL__BELIEF__BUILDER_HPP_
