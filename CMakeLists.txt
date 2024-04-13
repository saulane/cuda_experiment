cmake_minimum_required(VERSION 3.10)  # Specify the minimum version of CMake.

project(MyProject)  # Specify the project name and the language used.
set(CMAKE_C_STANDARD 99)  # Set the C standard to C99.

add_executable(test_dl test_dl.c)  # Create an executable named MyExecutable from main.c.

# Link the math library to the test_dl target
target_link_libraries(test_dl m)