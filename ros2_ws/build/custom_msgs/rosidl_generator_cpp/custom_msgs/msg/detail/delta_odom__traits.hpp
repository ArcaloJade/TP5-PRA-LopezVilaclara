// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from custom_msgs:msg/DeltaOdom.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__DELTA_ODOM__TRAITS_HPP_
#define CUSTOM_MSGS__MSG__DETAIL__DELTA_ODOM__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "custom_msgs/msg/detail/delta_odom__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace custom_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const DeltaOdom & msg,
  std::ostream & out)
{
  out << "{";
  // member: dr1
  {
    out << "dr1: ";
    rosidl_generator_traits::value_to_yaml(msg.dr1, out);
    out << ", ";
  }

  // member: dr2
  {
    out << "dr2: ";
    rosidl_generator_traits::value_to_yaml(msg.dr2, out);
    out << ", ";
  }

  // member: dt
  {
    out << "dt: ";
    rosidl_generator_traits::value_to_yaml(msg.dt, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const DeltaOdom & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: dr1
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "dr1: ";
    rosidl_generator_traits::value_to_yaml(msg.dr1, out);
    out << "\n";
  }

  // member: dr2
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "dr2: ";
    rosidl_generator_traits::value_to_yaml(msg.dr2, out);
    out << "\n";
  }

  // member: dt
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "dt: ";
    rosidl_generator_traits::value_to_yaml(msg.dt, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const DeltaOdom & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace custom_msgs

namespace rosidl_generator_traits
{

[[deprecated("use custom_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const custom_msgs::msg::DeltaOdom & msg,
  std::ostream & out, size_t indentation = 0)
{
  custom_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use custom_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const custom_msgs::msg::DeltaOdom & msg)
{
  return custom_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<custom_msgs::msg::DeltaOdom>()
{
  return "custom_msgs::msg::DeltaOdom";
}

template<>
inline const char * name<custom_msgs::msg::DeltaOdom>()
{
  return "custom_msgs/msg/DeltaOdom";
}

template<>
struct has_fixed_size<custom_msgs::msg::DeltaOdom>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<custom_msgs::msg::DeltaOdom>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<custom_msgs::msg::DeltaOdom>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CUSTOM_MSGS__MSG__DETAIL__DELTA_ODOM__TRAITS_HPP_
