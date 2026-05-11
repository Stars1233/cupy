#ifndef STRUCT_VIEW_H_
#define STRUCT_VIEW_H_

#include "cupy/carray.cuh"


namespace cupy {

// Forward declaration for SFINAE
template<typename, typename...>
class StructView;

// Trait to detect StructView
template<typename T>
struct is_struct_view {
  static constexpr bool value = false;
};

template<typename StorageType, typename... Fields>
struct is_struct_view<StructView<StorageType, Fields...>> {
  static constexpr bool value = true;
};

template<size_t Size, size_t Align>
struct alignas(Align) raw_structview_storage {
  static_assert(Align != 0);
  char _data[Size];
};

// Field descriptor: type and offset only
template<typename T, size_t Offset>
struct Field {
  using type = T;
  static constexpr size_t offset = Offset;
};

// Metaprogramming helper: get field type by index
template<size_t Index, typename... Fields>
struct FieldAtImpl;
template<typename First, typename... Rest>
struct FieldAtImpl<0, First, Rest...> { using type = First; };
template<size_t Index, typename First, typename... Rest>
struct FieldAtImpl<Index, First, Rest...> { using type = typename FieldAtImpl<Index - 1, Rest...>::type; };

// StructView represents a NumPy structured dtype as a C struct.
// Generally, NumPy structured dtypes operate by field index `.at<0>()`
// will fetch the first index and casts/assignment only use fields.
template<typename StorageType, typename... Fields>
class StructView {
  template<size_t Index>
  using FieldAt = typename FieldAtImpl<Index, Fields...>::type;

public:
  using storage_type = StorageType;
  static constexpr size_t size = sizeof(storage_type);
  static constexpr size_t alignment = alignof(storage_type);
  static constexpr size_t field_count = sizeof...(Fields);

  __device__ StructView() : data_{} {
    init<0>();
  }

  __device__ StructView(
      const StructView<storage_type, Fields...>& other) : data_{} {
    assign<0>(other);  // non-trivial to try and honor "holes"
  }

  // Construct from other struct
  template<typename OtherStorageType, typename... OtherFields>
  explicit __device__ StructView(
      const StructView<OtherStorageType, OtherFields...>& other)
      : data_{} {
    static_assert(sizeof...(Fields) == sizeof...(OtherFields), "Field count must match");
    assign<0>(other);
  }

  // Construct from "scalar" by broadcasting.
  template<typename T>
  explicit __device__ StructView(const T& value) : data_{} {
    assign_broadcast<0>(value);
  }

  // Index-based field access
  template<size_t Index>
  __device__ auto& at() {
    using FT = FieldAt<Index>;
    using T = typename FT::type;
    return *reinterpret_cast<T*>(reinterpret_cast<char*>(&data_) + FT::offset);
  }

  template<size_t Index>
  __device__ const auto& at() const {
    using FT = FieldAt<Index>;
    using T = typename FT::type;
    return *reinterpret_cast<const T*>(
        reinterpret_cast<const char*>(&data_) + FT::offset);
  }

  // Cross-type equality (only requires operator== on field types)
  // Note: Like NumPy structured dtypes, only == and != are supported
  template<typename OtherStorageType, typename... OtherFields>
  __device__ bool operator==(
      const StructView<OtherStorageType, OtherFields...>& other) const {
    static_assert(sizeof...(Fields) == sizeof...(OtherFields), "Field count must match");
    return compare_eq<0>(other);
  }

  template<typename OtherStorageType, typename... OtherFields>
  __device__ bool operator!=(
      const StructView<OtherStorageType, OtherFields...>& other) const {
    return !(*this == other);
  }

  // Same as constructor (but more interesting as we actually need to omit holes).
  __device__ StructView& operator=(
      const StructView<storage_type, Fields...>& other) {
    assign<0>(other);
    return *this;
  }

  template<typename OtherStorageType, typename... OtherFields>
  __device__ StructView& operator=(
      const StructView<OtherStorageType, OtherFields...>& other) {
    static_assert(sizeof...(Fields) == sizeof...(OtherFields), "Field count must match");
    assign<0>(other);
    return *this;
  }

  // Broadcast assignment - assign single value to all fields.
  // NOTE(seberg): Can omitting enable_if lead to ambiguity?
  template<typename T>
  __device__ StructView& operator=(const T& value) {
    assign_broadcast<0>(value);
    return *this;
  }

private:
  storage_type data_;

  // Equality comparison helper (only requires operator==)
  template<size_t Index, typename OtherStorageType, typename... OtherFields>
  __device__ bool compare_eq(
      const StructView<OtherStorageType, OtherFields...>& other) const {
    if constexpr (Index < sizeof...(Fields)) {
      if (!(at<Index>() == other.template at<Index>())) return false;
      return compare_eq<Index + 1>(other);
    }
    return true;
  }

  // Assignment helper (field-by-field from another StructView)
  template<size_t Index, typename OtherStorageType, typename... OtherFields>
  __device__ void assign(
      const StructView<OtherStorageType, OtherFields...>& other) {
    if constexpr (Index < sizeof...(Fields)) {
      at<Index>() = other.template at<Index>();
      assign<Index + 1>(other);
    }
  }

  // Initialization helper (field-by-field init)
  template<size_t Index>
  __device__ void init() {
    if constexpr (Index < sizeof...(Fields)) {
      using FT = FieldAt<Index>;
      using T = typename FT::type;
      at<Index>() = T{};
      init<Index + 1>();
    }
  }

  // Broadcast assignment helper (assign same value to all fields)
  template<size_t Index, typename T>
  __device__ void assign_broadcast(const T& value) {
    if constexpr (Index < sizeof...(Fields)) {
      at<Index>() = value;
      assign_broadcast<Index + 1>(value);
    }
  }
};

} // namespace cupy

#endif // STRUCT_VIEW_H_
