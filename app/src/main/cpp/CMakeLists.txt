cmake_minimum_required(VERSION 3.22.1)
project("smartqrcode")

set(ZXING_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ZXING_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(ZXING_BLACKBOX_TESTS OFF CACHE BOOL "" FORCE)
set(ZXING_PYTHON_MODULE OFF CACHE BOOL "" FORCE)
set(ZXING_C_API OFF CACHE BOOL "" FORCE)
set(ZXING_WRITERS OFF CACHE STRING "" FORCE)
set(ZXING_READERS ON CACHE BOOL "" FORCE)

add_compile_definitions(ZXING_EXPERIMENTAL_API)

add_subdirectory(third_party/zxing-cpp EXCLUDE_FROM_ALL)

add_library(
    smartqrcode_native
    SHARED
    native_decoder.cpp
)

find_library(log-lib log)

target_link_libraries(
    smartqrcode_native
    ZXing::ZXing
    ${log-lib}
)
