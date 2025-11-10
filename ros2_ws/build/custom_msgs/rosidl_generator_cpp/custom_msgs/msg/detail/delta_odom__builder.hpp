// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from custom_msgs:msg/DeltaOdom.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__DELTA_ODOM__BUILDER_HPP_
#define CUSTOM_MSGS__MSG__DETAIL__DELTA_ODOM__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "custom_msgs/msg/detail/delta_odom__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace custom_msgs
{

namespace msg
{

namespace builder
{

class Init_DeltaOdom_dt
{
public:
  explicit Init_DeltaOdom_dt(::custom_msgs::msg::DeltaOdom & msg)
  : msg_(msg)
  {}
  ::custom_msgs::msg::DeltaOdom dt(::custom_msgs::msg::DeltaOdom::_dt_type arg)
  {
    msg_.dt = std::move(arg);
    return std::move(msg_);
  }

private:
  ::custom_msgs::msg::DeltaOdom msg_;
};

class Init_DeltaOdom_dr2
{
public:
  explicit Init_DeltaOdom_dr2(::custom_msgs::msg::DeltaOdom & msg)
  : msg_(msg)
  {}
  Init_DeltaOdom_dt dr2(::custom_msgs::msg::DeltaOdom::_dr2_type arg)
  {
    msg_.dr2 = std::move(arg);
    return Init_DeltaOdom_dt(msg_);
  }

private:
  ::custom_msgs::msg::DeltaOdom msg_;
};

class Init_DeltaOdom_dr1
{
public:
  Init_DeltaOdom_dr1()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_DeltaOdom_dr2 dr1(::custom_msgs::msg::DeltaOdom::_dr1_type arg)
  {
    msg_.dr1 = std::move(arg);
    return Init_DeltaOdom_dr2(msg_);
  }

private:
  ::custom_msgs::msg::DeltaOdom msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::custom_msgs::msg::DeltaOdom>()
{
  return custom_msgs::msg::builder::Init_DeltaOdom_dr1();
}

}  // namespace custom_msgs

#endif  // CUSTOM_MSGS__MSG__DETAIL__DELTA_ODOM__BUILDER_HPP_
