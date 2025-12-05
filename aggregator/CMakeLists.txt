cmake_minimum_required(VERSION 3.15)
project(FederatedAggregator CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)

# Protocol Buffers
set(PROTO_PATH "${CMAKE_SOURCE_DIR}/../proto")
set(PROTO_FILE "${PROTO_PATH}/coordinator.proto")
set(GENERATED_PROTOBUF_PATH "${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY ${GENERATED_PROTOBUF_PATH})

# Generate protobuf and gRPC files
set(PROTO_SRCS "${GENERATED_PROTOBUF_PATH}/coordinator.pb.cc")
set(PROTO_HDRS "${GENERATED_PROTOBUF_PATH}/coordinator.pb.h")
set(GRPC_SRCS "${GENERATED_PROTOBUF_PATH}/coordinator.grpc.pb.cc")
set(GRPC_HDRS "${GENERATED_PROTOBUF_PATH}/coordinator.grpc.pb.h")

add_custom_command(
    OUTPUT "${PROTO_SRCS}" "${PROTO_HDRS}" "${GRPC_SRCS}" "${GRPC_HDRS}"
    COMMAND protobuf::protoc
    ARGS --grpc_out "${GENERATED_PROTOBUF_PATH}"
         --cpp_out "${GENERATED_PROTOBUF_PATH}"
         -I "${PROTO_PATH}"
         --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
         "${PROTO_FILE}"
    DEPENDS "${PROTO_FILE}"
)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${GENERATED_PROTOBUF_PATH}
    ${EIGEN3_INCLUDE_DIR}
)

# Source files
set(SOURCES
    src/main.cpp
    src/aggregator.cpp
    src/byzantine_detector.cpp
    src/compression.cpp
    ${PROTO_SRCS}
    ${GRPC_SRCS}
)

# Executable
add_executable(aggregator_server ${SOURCES})

# Link libraries
target_link_libraries(aggregator_server
    gRPC::grpc++
    gRPC::grpc++_reflection
    protobuf::libprotobuf
    Eigen3::Eigen
)

# OpenMP support
if(OpenMP_CXX_FOUND)
    target_link_libraries(aggregator_server OpenMP::OpenMP_CXX)
endif()

# Compiler flags
target_compile_options(aggregator_server PRIVATE
    -Wall
    -Wextra
    -O3
    -march=native
)

# Install
install(TARGETS aggregator_server DESTINATION bin)

# Print configuration
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Protobuf Version: ${Protobuf_VERSION}")
message(STATUS "gRPC Found: ${gRPC_FOUND}")
message(STATUS "Eigen3 Found: ${Eigen3_FOUND}")
message(STATUS "OpenMP Found: ${OpenMP_CXX_FOUND}")
